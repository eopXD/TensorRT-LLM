# KV Cache Regions Developer Guide

Model and attention-backend code declares the persistent memory it needs as **regions**. The KV
cache manager lays every region out on KV cache manager V2 storage, keeps the map of which region
owns each storage layer, and derives from the declarations everything the shared runtime needs:
allocation, sizing, disaggregated transfer, prefix-reuse alignment and executor gates. Shared code
never asks what a model or backend is.

Code: `tensorrt_llm/_torch/pyexecutor/kv_cache/regions.py` (declarations and layout),
`KVCacheManagerV2` (declaration and layout hooks), `MambaHybridCacheManagerV2` (the hybrid
consumers). Kimi K3 is the reference user.

## Vocabulary

| Term | Meaning | Type |
|---|---|---|
| Plane | One kind of data a region stores per model layer, under one buffer role | `Plane` |
| Region layer | The planes a region stores for one rank-local model layer, plus its per-token cost | `RegionLayer` |
| Life cycle | When the region's memory is allocated, released early and handed off to the prefix index | `RegionLifeCycle` |
| Region | Planes over a set of model layers with one life cycle | `Region` |
| Storage index | Position of a region's storage among the regions covering the same model layers | `Region.storage_index` |
| Provider | Model- or backend-side code that declares regions and handles their hooks | `RegionProvider` |
| Layout | Which region owns each storage layer of the manager's V2 layer list | `RegionLayout` |

## How a region maps onto V2 storage

V2 storage offers two allocation shapes on the same paged pools. The life cycle selects one:

| Allocation unit | Hand-off | Early release | V2 layer config |
|---|---|---|---|
| `TOKEN_BLOCK` | `BLOCK_BOUNDARIES` | outside `window`, or none | `AttentionLayerConfig(sliding_window_size=window, reuse_policy=REQUIRED)` |
| `TOKEN_BLOCK` | `NEVER` | outside `window`, or none | `AttentionLayerConfig(..., reuse_policy=PRIVATE)` |
| `REQUEST` | `DECLARED_POINTS` | all but the latest state | `SsmLayerConfig(reuse_policy=REQUIRED)` |
| `REQUEST` | `NEVER` | all but the latest state | `SsmLayerConfig(reuse_policy=PRIVATE)` |

A token-block region is a span of slots, one per block, of which the window stays live. A
per-request region is one slot per request; earlier states leave it only as snapshots taken at the
request's declared points.

**Storage ids.** Storage at index 0 takes the id of the model layer it covers, so model-layer views
(`pp_layers`, per-layer head counts and windows) index it directly. Storage at a higher index is
appended after the model layers in declaration order and never indexes a per-model-layer table.
Two regions may not store the same model layer at the same index.

**Transfer ids.** A storage layer's disaggregated transfer id is `stride * L + index`, where `L` is
the global model layer and `stride` is one plus the highest declared storage index. The stride
depends only on the declarations, so ids agree across pipeline splits.

## Provider hooks

`KVCacheManagerV2` and `MambaHybridCacheManagerV2` call a provider's hooks in this order:

| Hook | When | Use |
|---|---|---|
| `configure(manager, kv_cache_config)` | Before the manager builds anything | Validate and adjust the cache config; record geometry the declarations need |
| `declare_regions()` | After the per-layer views exist, before the memory budget is costed | Return the regions; cost and sizing read them before any storage id exists |
| `on_layout(layout)` | Once storage ids are final, after empty layers are compacted | Record the ids the provider's own accessors address |
| `finalize(manager)` | After the pools exist | Validate pool geometry, initialize storage |
| `iter_cache_views()` | During the invalid-value scan | Views covering the provider's storage |
| `context_only` | Executor and warmup gates | Whether this worker omits storage only generation steps read |

`configure` and `finalize` run in the hybrid manager's constructor, which creates the providers in
`_create_region_providers(context)`; `RegionContext` carries the manager's construction arguments.
The manager class that owns a model assembles its providers there, including backend providers it
composes.

## What the core derives from the declarations

| Runtime need | Declaration it reads | Where |
|---|---|---|
| V2 layer configs and the ownership map | layers, planes, life cycle, storage index | `KVCacheManagerV2._lay_out_regions` |
| Minimum-snapshot commits | presence of per-request storage | `KVCacheManagerV2._lay_out_regions` |
| Disagg mapper kinds, rebuilt (untransferred) roles, role layouts | `Region.transfer` | `MambaHybridCacheManagerV2.get_disagg_*` |
| Disagg transfer ids | storage index | `MambaHybridCacheManagerV2.get_disagg_global_layer_ids` |
| Whether a layer group is slot storage | allocation unit | `is_slot_storage`, disagg page-table builder |
| Attention bytes per token | `RegionLayer.bytes_per_token` of token-block regions | `_attention_cache_bytes_per_token` |
| Class-level cost estimate, before a manager exists | `_attention_region_cost` of the manager class that assembles the providers | `MambaHybridCacheManagerV2.get_cache_size_per_token` |
| Primary pool capacity | first token-block region at index 0 | `blocks_in_primary_pool` |
| Recurrent bytes per slot and typed views | planes of per-request regions | `_mamba_state_bytes_per_slot`, `_build_cache_config` |
| Snapshot points on block boundaries | `resumes_at_block_boundaries` | `_align_snapshot_points_to_blocks` |
| History length after every chunk | `positioned_by_history` | `update_context_resources` |
| Context-only worker gate | `RegionProvider.context_only` | `context_only_cache`, `is_context_only_kv_cache` |

A manager that declares token-block regions costs a token by its region storage only, so every
attention layer of such a manager must be covered by a region at index 0.

## Kimi K3

`KimiK3HybridCacheManager` (`tensorrt_llm/_torch/modules/kimi_kda/cache_manager.py`) inherits
`Fp4MlaHybridCacheManager` (`tensorrt_llm/_torch/attention/backends/fp4_mla/regions.py`), which
adds `Fp4MlaRegionProvider` when the KV cache dtype selects the FP4 MLA layout, and adds its own
`KdaRegionProvider`.

| Region | Provider | Life cycle | Index | Transfer | Properties |
|---|---|---|---|---|---|
| `fp4_mla_latent` | FP4 MLA | token block, block hand-off, model window | 0 | K, K scales NHD; V scales and packed V rebuilt | resumes at block boundaries |
| `fp8_mla_latent` | FP4 MLA | token block, block hand-off, never-evicting window | 0 | K NHD | — |
| `fp4_mla_hp_ring` | FP4 MLA | token block, block hand-off, window = 16 + max rewind | 1 | replicated ring | positioned by history |
| `kda_recurrent_state` | KDA | per request, snapshots at declared points | 0 | replay buffers rebuilt | context-only without replay buffers |

Three declarations encode a storage constraint rather than the region's own timing:

- The FP8 latent declares a window larger than any sequence to obtain a pool of its own. Equal life
  cycles share one V2 pool group, and the FP4 MLA kernels address the FP8 latent through a separate
  pool. A windowless declaration also changes page eviction priority and windowed pool sizing.
- The HP ring hands off at block boundaries, as every required attention life cycle does.
- The KDA replay buffers are planes of the recurrent-state region, so the KDA kernels address them
  through the recurrent state's slot index; they are held and snapshotted with it.

## Rules for shared code

| Rule | Instead of |
|---|---|
| Ask who owns a storage layer and what its region declared | Asking whether a layer is a Mamba, KDA or MLA layer |
| Let the manager class that owns the model assemble its providers | Detecting a backend from the dtype or config in shared code |
| Read cost from the declarations | Passing backend-specific overrides into shared estimators |
| Declare the intended life cycle and grouping | Inventing layers or windows to obtain a behaviour |
| Give every life cycle its own region | Placing one region's planes inside another's life cycle |
| Take transfer ids, mapper kinds and layouts from the owning region | Encoding backend layouts in shared transfer code |
| Reach providers through the provider hooks | Attaching provider methods to the manager or probing it with `getattr` |
| Raise generic events and properties from the executor | Gates and hooks named after one model family |

The FP4 MLA provider still attaches its accessors to the manager (`install_manager_capabilities`),
because its attention backend reaches its pools through `metadata.kv_cache_manager`.

## Migrating a model to regions

1. Write a provider next to the model or backend code that owns the memory. Declare one region per
   life cycle, the planes per model layer, the per-token cost, the transfer declarations and the
   read/write properties.
2. Assemble the provider in the model's manager class through `_create_region_providers`.
3. Delete the shared-code branches the declarations replace.
4. Show the refactor changes nothing: for each configuration the model supports, compare before and
   after the built `KVCacheManagerConfig`, `pool_group_descs`, the page-table tensors
   (`kv_cache_pool_pointers` by owning pool, `kv_cache_pool_mapping`, `index_scales`), the class
   cost estimate (`get_cache_size_per_token`), the disagg page table
   (`build_page_table_from_manager`) and the backend's kernel-facing views. Raw device addresses
   differ between processes; compare them by the pool and offset they fall in.
