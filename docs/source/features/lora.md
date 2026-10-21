# LoRA (Low-Rank Adaptation)

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that enables adapting large language models to specific tasks without modifying the original model weights. Instead of fine-tuning all parameters, LoRA introduces small trainable rank decomposition matrices that are added to existing weights during inference.

## Table of Contents
1. [Background](#background)
2. [Basic Usage](#basic-usage)
   - [Single LoRA Adapter](#single-lora-adapter)
   - [Multi-LoRA Support](#multi-lora-support)
3. [Advanced Usage](#advanced-usage)
   - [LoRA with Quantization](#lora-with-quantization)
   - [NeMo LoRA Format](#nemo-lora-format)
   - [Routed-Expert MoE LoRA](#routed-expert-moe-lora)
   - [Multimodal Component LoRA](#multimodal-component-lora)
   - [Cache Management](#cache-management)
4. [TRTLLM serve with LoRA](#trtllm-serve-with-lora)
   - [YAML Configuration](#yaml-configuration)
   - [Starting the Server](#starting-the-server)
   - [Client Usage](#client-usage)
5. [TRTLLM bench with LoRA](#trtllm-bench-with-lora)
   - [YAML Configuration](#yaml-configuration)
   - [Run trtllm-bench](#run-trtllm-bench)

## Background

The PyTorch backend provides LoRA support, allowing you to:
- Load and apply multiple LoRA adapters simultaneously
- Switch between different adapters for different requests
- Use LoRA with quantized models
- Support both HuggingFace and NeMo LoRA formats

## Basic Usage

### Single LoRA Adapter

```python
from tensorrt_llm import LLM
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.sampling_params import SamplingParams

# Configure LoRA
lora_config = LoraConfig(
    lora_dir=["/path/to/lora/adapter"],
    max_lora_rank=8,
    max_loras=1,
    max_cpu_loras=1
)

# Initialize LLM with LoRA support
llm = LLM(
    model="/path/to/base/model",
    lora_config=lora_config
)

# Create LoRA request
lora_request = LoRARequest("my-lora-task", 0, "/path/to/lora/adapter")

# Generate with LoRA
prompts = ["Hello, how are you?"]
sampling_params = SamplingParams(max_tokens=50)

outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=[lora_request]
)
```

### Multi-LoRA Support

```python
# Configure for multiple LoRA adapters
lora_config = LoraConfig(
    lora_target_modules=['attn_q', 'attn_k', 'attn_v'],
    max_lora_rank=8,
    max_loras=4,
    max_cpu_loras=8
)

llm = LLM(model="/path/to/base/model", lora_config=lora_config)

# Create multiple LoRA requests
lora_req1 = LoRARequest("task-1", 0, "/path/to/adapter1")
lora_req2 = LoRARequest("task-2", 1, "/path/to/adapter2")

prompts = [
    "Translate to French: Hello world",
    "Summarize: This is a long document..."
]

# Apply different LoRAs to different prompts
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=[lora_req1, lora_req2]
)
```

## Advanced Usage

### LoRA with Quantization

```python
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

# Configure quantization
quant_config = QuantConfig(
    quant_algo=QuantAlgo.FP8,
    kv_cache_quant_algo=QuantAlgo.FP8
)

# LoRA works with quantized models
llm = LLM(
    model="/path/to/model",
    quant_config=quant_config,
    lora_config=lora_config
)
```

#### Native FP8 adapter support

The base model's quantization and the LoRA adapter's data type are independent. Dense LoRA modules can keep
FP8 E4M3 adapter weights and execute them with native FP8 grouped GEMM kernels on the following architectures:

| GPU architecture | Native FP8 adapter support |
|---|---|
| Hopper (SM90) | Yes |
| Blackwell B200 (SM100) | Yes |
| Blackwell (SM103/SM107) | No |
| Blackwell (SM120/SM121) | No |

If native FP8 LoRA kernels are unavailable for the current device or were excluded from the TensorRT-LLM build,
the adapter weights are converted to the model compute data type. Native FP8 adapter weights are not supported
for routed-expert MoE LoRA modules; see [Routed-Expert MoE LoRA](#routed-expert-moe-lora).

### NeMo LoRA Format

```python
# For NeMo-format LoRA checkpoints
lora_config = LoraConfig(
    lora_dir=["/path/to/nemo/lora"],
    lora_ckpt_source="nemo",
    max_lora_rank=8
)

lora_request = LoRARequest(
    "nemo-task",
    0,
    "/path/to/nemo/lora",
    lora_ckpt_source="nemo"
)
```

### Routed-Expert MoE LoRA

LoRA can be applied to the routed-expert projections of a Mixture-of-Experts (MoE) layer in addition to the attention modules. The PyTorch backend's Cutlass MoE kernel fuses the LoRA application into the MoE forward pass, so multi-adapter batches run without an extra GEMM pass per layer.

#### Supported configuration

| Aspect | Supported |
|---|---|
| MoE backend | `CUTLASS` only (other backends raise an error at construction). |
| Base-weight dtype | bf16 / fp16. Quantized base weights (FP8, NVFP4, INT4, INT8) are not yet supported. |
| Adapter modules | `moe_h_to_4h` (gate side of SwiGLU), `moe_gate` (up side), `moe_4h_to_h` (down). `moe_h_to_4h` and `moe_4h_to_h` must both be present; `moe_gate` is optional (gated activations only). |
| Adapter layout | Per-expert (stacked `[num_experts, ...]`). |
| Multi-LoRA in flight | Yes. Reuses the existing slot manager. |
| Execution modes | Eager and CUDA-graph capture/replay (see below). Both feed the same grouped-GEMM LoRA core. |
| min-latency mode | Not supported with MoE LoRA. |
| Alltoall (WideEP) | Not supported with MoE LoRA. |
| DoRA on MoE modules | Not supported (and rejected at load time). |
| `register_to_config` + `torch.compile` | Not supported with MoE LoRA. |

#### Enabling routed-expert MoE LoRA

```python
from tensorrt_llm import LLM
from tensorrt_llm._torch.peft.lora.config import LoraConfig

lora_config = LoraConfig(
    lora_target_modules=[
        "attn_q", "attn_k", "attn_v",  # optional: standard attention LoRA
        "moe_h_to_4h",                 # gate/SiLU projection (required for MoE LoRA)
        "moe_4h_to_h",                 # down projection (required for MoE LoRA)
        "moe_gate",                    # up/linear projection (optional; gated activations)
    ],
    max_lora_rank=16,
    max_loras=8,
    max_cpu_loras=8,
)

llm = LLM(
    model="/path/to/moe_base_model",
    lora_config=lora_config,
    # The MoE LoRA path requires the Cutlass backend. Other moe_backend values
    # raise ValueError at construction.
    moe_backend="CUTLASS",
)
```

Adapters are loaded via `LoRARequest` exactly like attention-only LoRA; no API change.

#### Adapter layout

Each routed expert has its own `(A, B)` matrices, stored as stacked `[num_experts, rank, in_dim]` and `[num_experts, out_dim, rank]` tensors. This is the standard HuggingFace PEFT export shape for MoE LoRA; the MoE kernel reads each expert's slice at offset `expert_index * dim * rank`.

A helper for assembling synthetic per-expert adapters (for unit tests and experimentation) is provided at `tensorrt_llm._torch.peft.lora.moe_layout`:

```python
from tensorrt_llm._torch.peft.lora.moe_layout import make_per_expert_lora

fc1_adapter = make_per_expert_lora(
    num_experts=8, rank=16, in_dim=2048, out_dim=5632,
    dtype=torch.bfloat16,
)
# fc1_adapter["A"].shape == (8, 16, 2048)  -- independent per expert
# fc1_adapter["B"].shape == (8, 5632, 16)  -- independent per expert
```

#### Execution modes

Routed-expert MoE LoRA always runs through a single capture-safe grouped-GEMM core (on-stream pointer expansion, problem building, and grouped GEMMs). Two input schemas feed that core, and both produce identical results:

- **Eager** uses the per-request input schema: the op expands the per-request adapter tables into per-token `(rank, A, B)` arrays before launching the core. This mode handles both context (prefill) and decode batches.
- **CUDA-graph decode** uses the slot-indexed input schema: `CudaGraphLoraManager` maintains stable per-slot adapter tables and a `token_to_slot` map, and the slot→token expansion runs entirely on the stream. Because the tables live at stable addresses and are refreshed in place, reassigning a slot's adapter is reflected on replay without re-capture. CUDA-graph decode is generation-only.

The per-request schema is not itself CUDA-graph capturable (its host-side adapter expansion would be frozen at capture time), so under capture the op uses the slot-indexed schema; supplying per-request inputs while capturing raises a clear error.

#### What is rejected, and where

If you supply MoE LoRA on a non-Cutlass backend or with quantization, `create_moe` raises at construction with a message pointing at the offending setting. At runtime, the fused MoE op also rejects min-latency mode + LoRA, alltoall + LoRA, and per-request (eager-schema) inputs under CUDA-graph capture (use the slot-indexed schema for capture).

### Multimodal Component LoRA

Set `lora_config.enable_multimodal_lora: true` to apply one request-selected HF
adapter to declared encoder and connector projections as well as the decoder.
The default is `false`, which keeps decoder-only ownership and rejects encoder
or connector weights. The existing `LoRARequest` API selects one checkpoint per
request; that checkpoint can contain any supported subset of components.

The runtime accepts these native, unquantized BF16 model configurations:

| Model configuration | Vision encoder | Audio encoder | Connectors | Tensor parallelism |
| --- | --- | --- | --- | --- |
| Nemotron Nano-Omni-30B (`NemotronH_Nano_Omni_Reasoning_V3`) | RADIO-H Q/K/V, fused QKV, output projection | Parakeet Q/K/V and output projection | Both linear projections in the vision and audio connectors | TP1 |
| Nemotron Super 3.5 VL (`NemotronH_Omni_Reasoning_V3`, vision-only, 4096-wide 88-block decoder) | RADIO-H Q/K/V, fused QKV, output projection | Not available | Both linear projections in the vision connector | TP4 on one node |
| Mistral Small 3.1 (`Mistral3ForConditionalGeneration`, dense decoder) | Pixtral Q/K/V and output projection | Not available | `linear_1` and `linear_2` in the multimodal projector | TP1 |

Other variants of these architectures require their own component declarations.
Encoder MLPs, patch embeddings, audio convolutions, and the Mistral patch merger
are outside the enrolled targets. Packed and split versions of the same projection
cannot coexist in one checkpoint.

```yaml
lora_config:
  enable_multimodal_lora: true
  lora_dir:
    - /path/to/composite-adapter
  max_lora_rank: 8
  max_loras: 4
  max_cpu_loras: 4
```

Component checkpoint module names are matched against the model's declared
paths before normalization. The optional `base_model.model.` prefix and
`.lora_A.weight` / `.lora_B.weight` suffix surround these paths:

| Component | Accepted module path examples |
| --- | --- |
| Nemotron vision | `vision_model.radio_model.model.blocks.0.attn.q_proj`, `vision_model.encoder.layer.0.attention.query`, `vision_model.encoder.layer.0.attention.attention.query` |
| Nano-Omni audio | `sound_encoder.encoder.layers.0.self_attn.q_proj` |
| Nemotron vision connector | `mlp1.1`, `mlp1.3`, `vision_projector.mlp1.linear1`, `vision_projector.mlp1.linear2` |
| Nano-Omni audio connector | `sound_projection.linear1`, `sound_projection.linear2` |
| Mistral vision | `vision_tower.transformer.layers.0.attention.q_proj`, `model.vision_tower.transformer.layers.0.attention.q_proj` |
| Mistral connector | `multi_modal_projector.linear_1`, `model.multi_modal_projector.linear_1` |

Flat and language-model-prefixed decoder paths retain their existing meaning.
An unsupported component, undeclared projection, duplicate destination, invalid
dimension, or excessive rank fails before the adapter is registered. Checkpoints
containing encoder or connector weights require one rank across their targeted
components, use the shared BF16 cache dtype, and support standard LoRA or rsLoRA.
DoRA, `alpha_pattern`, and `fan_in_fan_out` are not supported for these composite
checkpoints. The component policy accepts HF checkpoints only. With the policy
disabled, decoder-only checkpoints retain their existing format rules.

Use full, unsharded HF adapter weights for Super VL. The runtime shards RADIO
attention and decoder adapters across four ranks and replicates the vision
connector adapters on every rank. Adapter checkpoints cannot contain audio
targets for this vision-only configuration.

Component LoRA requires the model's listed TP size on one node, PP1/CP1/EP1,
local encoder and decoder execution, and eager encoders. Attention DP, DWDP,
MoE clusters, and component TP subgroups are unsupported. Decoder CUDA graphs
remain available. Encoder
CUDA graphs, prefill CUDA graphs, `torch.compile`, disaggregated serving,
encoder-only workers, encoder side-stream prefetch, and EVS video pruning are
rejected when this policy is enabled.
Item scheduling supports `DEFAULT` and `DISABLED`; `EAGER` is rejected because
it can request adapter residency before LLM admission releases cache capacity.
For Nano Omni, set `video_pruning_rate` to `null` or `0.0` to keep video pruning
disabled. Set it explicitly to `0.0` for Super VL, whose checkpoint can enable
pruning by default. Leave this option unset for Mistral; its input processor does
not accept an explicit pruning rate.

Different requests in a batch can select different adapters or no adapter.
Encoder mappings follow physical patch/frame rows, including audio padding;
connector mappings follow the rows after projection preparation. Base-only rows
receive no adapter contribution. Encoder-embedding reuse includes the relevant
encoder and connector weight fingerprints, so identical media can be reused
across decoder-only adapter changes. Changed encoder or connector weights produce
separate cache entries, including after an adapter reload. Precomputed final
embeddings require matching component provenance when component adaptation is
active.

The worker hashes checkpoint contents on each opt-in adapter selection and keeps
an immutable copy of each distinct revision until shutdown. These checkpoint
copies use disk space outside the `max_cpu_loras` weight-cache capacity.

### Cache Management

```python
from tensorrt_llm.llmapi.llm_args import PeftCacheConfig

# Fine-tune cache sizes
peft_cache_config = PeftCacheConfig(
    host_cache_size=1024*1024*1024,  # 1GB CPU cache
    device_cache_percent=0.1          # 10% of free GPU memory
)

llm = LLM(
    model="/path/to/model",
    lora_config=lora_config,
    peft_cache_config=peft_cache_config
)
```

## TRTLLM serve with LoRA

### YAML Configuration

```{eval-rst}
.. include:: ../_includes/note_sections.rst
   :start-after: .. start-note-config-flag-alias
   :end-before: .. end-note-config-flag-alias
```

Create a `config.yaml` file:

```yaml
lora_config:
  lora_target_modules: ['attn_q', 'attn_k', 'attn_v']
  max_lora_rank: 8
```
### Starting the Server
```bash
python -m tensorrt_llm.commands.serve
     /path/to/model \
    --config config.yaml
```

### Client Usage

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.completions.create(
    model="/path/to/model",
    prompt="What is the capital city of France?",
    max_tokens=20,
    extra_body={
        "lora_request": {
            "lora_name": "lora-example-0",
            "lora_int_id": 0,
            "lora_path": "/path/to/lora_adapter"
        }
    },
)
```

## TRTLLM bench with LoRA

### YAML Configuration

```{eval-rst}
.. include:: ../_includes/note_sections.rst
   :start-after: .. start-note-config-flag-alias
   :end-before: .. end-note-config-flag-alias
```

Create a `config.yaml` file:

```yaml
lora_config:
  lora_dir:
    - /workspaces/tensorrt_llm/loras/0
  max_lora_rank: 64
  max_loras: 8
  max_cpu_loras: 8
  lora_target_modules:
    - attn_q
    - attn_k
    - attn_v
  trtllm_modules_to_hf_modules:
    attn_q: q_proj
    attn_k: k_proj
    attn_v: v_proj
```
### Run trtllm-bench
```bash
trtllm-bench --model $model_path throughput --dataset $dataset_path --config config.yaml --num_requests 64 --concurrency 16
```
