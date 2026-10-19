# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for immutable adapter snapshots, without importing the GPU runtime."""

import importlib.util
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[4] / "tensorrt_llm/_torch/peft/lora/adapter_revision.py"
_SPEC = importlib.util.spec_from_file_location("_adapter_revision_cpu_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
adapter_revision = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = adapter_revision
_SPEC.loader.exec_module(adapter_revision)

pytestmark = pytest.mark.cpu_only


def _checkpoint(path: Path, filename: str = "adapter_model.safetensors") -> Path:
    path.mkdir()
    (path / "adapter_config.json").write_bytes(b'{"r":4,"lora_alpha":8}')
    (path / filename).write_bytes(b"adapter-weights-A")
    return path


@pytest.mark.parametrize("filename", ["adapter_model.safetensors", "adapter_model.bin"])
def test_unchanged_checkpoint_reuses_snapshot_without_copying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    source = _checkpoint(tmp_path / "adapter", filename)
    with adapter_revision.AdapterRevisionRegistry() as registry:
        first = registry.resolve(7, str(source))

        def unexpected_copy(*_args: object) -> None:
            raise AssertionError("Unchanged source must reuse its snapshot")

        monkeypatch.setattr(registry, "_copy_source", unexpected_copy)
        assert registry.resolve(7, str(source)) is first
        assert 0 < first.task_id < 2**63
        assert Path(first.path, filename).read_bytes() == b"adapter-weights-A"
        with pytest.raises(FrozenInstanceError):
            first.revision = "changed"


@pytest.mark.parametrize("changed_file", ["adapter_config.json", "adapter_model.safetensors"])
def test_rewrite_creates_new_identity_and_retains_previous_bytes(
    tmp_path: Path, changed_file: str
) -> None:
    source = _checkpoint(tmp_path / "adapter")
    with adapter_revision.AdapterRevisionRegistry() as registry:
        first = registry.resolve(7, str(source))
        before = Path(first.path, changed_file).read_bytes()
        (source / changed_file).write_bytes(before + b" changed")
        second = registry.resolve(7, str(source))

        assert first.task_id != second.task_id
        assert first.revision != second.revision
        assert first.path != second.path
        assert Path(first.path, changed_file).read_bytes() == before
        assert Path(second.path, changed_file).read_bytes() == before + b" changed"
        assert registry.resolve(7, str(source)) is second


def test_same_content_and_adapter_id_have_stable_identity_across_registries(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "adapter")
    with adapter_revision.AdapterRevisionRegistry() as first_registry:
        first = first_registry.resolve(11, str(source))
        with adapter_revision.AdapterRevisionRegistry() as second_registry:
            second = second_registry.resolve(11, str(source))
            assert (first.task_id, first.revision) == (second.task_id, second.revision)
            assert first.path != second.path


def test_changed_stat_with_identical_content_reuses_existing_revision(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "adapter")
    with adapter_revision.AdapterRevisionRegistry() as registry:
        first = registry.resolve(7, str(source))
        weights = source / "adapter_model.safetensors"
        previous = weights.stat()
        os.utime(weights, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000))
        assert registry.resolve(7, str(source)) is first


def test_restoring_previous_content_reuses_its_retained_snapshot(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "adapter")
    weights = source / "adapter_model.safetensors"
    original = weights.read_bytes()
    with adapter_revision.AdapterRevisionRegistry() as registry:
        first = registry.resolve(7, str(source))
        weights.write_bytes(b"adapter-weights-B")
        second = registry.resolve(7, str(source))
        weights.write_bytes(original)
        assert registry.resolve(7, str(source)) is first
        assert Path(second.path, weights.name).read_bytes() == b"adapter-weights-B"


def _coarsen_stat_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    def signature_without_timestamps(
        cls: type[adapter_revision._FileSignature], value: os.stat_result
    ) -> adapter_revision._FileSignature:
        return cls(value.st_dev, value.st_ino, value.st_size, 0, 0)

    monkeypatch.setattr(
        adapter_revision._FileSignature,
        "from_stat",
        classmethod(signature_without_timestamps),
    )


@pytest.mark.parametrize("changed_file", ["adapter_config.json", "adapter_model.safetensors"])
def test_same_stat_rewrites_are_identified_by_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_file: str
) -> None:
    source = _checkpoint(tmp_path / "adapter")
    changed_path = source / changed_file
    original = changed_path.read_bytes()
    replacement = (
        original.replace(b"8", b"9") if changed_file.endswith("json") else original[:-1] + b"B"
    )
    assert len(replacement) == len(original) and replacement != original
    _coarsen_stat_timestamps(monkeypatch)

    with adapter_revision.AdapterRevisionRegistry() as registry:
        first = registry.resolve(7, str(source))
        signature = adapter_revision._source_files(source)
        changed_path.write_bytes(replacement)
        assert adapter_revision._source_files(source) == signature
        second = registry.resolve(7, str(source))
        assert second.task_id != first.task_id
        assert second.revision != first.revision
        assert Path(second.path, changed_file).read_bytes() == replacement

        changed_path.write_bytes(original)
        assert adapter_revision._source_files(source) == signature
        assert registry.resolve(7, str(source)) is first
        assert Path(first.path, changed_file).read_bytes() == original
        assert Path(second.path, changed_file).read_bytes() == replacement


def test_same_stat_change_between_hash_and_copy_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _checkpoint(tmp_path / "adapter")
    _coarsen_stat_timestamps(monkeypatch)
    original_copy = adapter_revision.AdapterRevisionRegistry._copy_source

    def change_before_copy(
        source_file: adapter_revision._SourceFile,
        destination: Path,
        digest: adapter_revision._Digest,
    ) -> None:
        if source_file.path.name == "adapter_model.safetensors":
            source_file.path.write_bytes(b"adapter-weights-B")
        original_copy(source_file, destination, digest)

    with adapter_revision.AdapterRevisionRegistry() as registry:
        monkeypatch.setattr(registry, "_copy_source", change_before_copy)
        with pytest.raises(RuntimeError, match="changed while snapshotting"):
            registry.resolve(7, str(source))
        assert registry._snapshots == {}
        assert registry._content_paths == {}
        assert list(registry._root.iterdir()) == []


def test_distinct_external_ids_do_not_alias_internal_task_ids(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "adapter")
    with adapter_revision.AdapterRevisionRegistry() as registry:
        first = registry.resolve(7, str(source))
        second = registry.resolve(8, str(source))
        assert first.task_id != second.task_id
        assert first.revision == second.revision
        assert first.path == second.path


def test_task_id_collision_is_rejected_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _checkpoint(tmp_path / "adapter")
    monkeypatch.setattr(adapter_revision, "_task_id", lambda *_args: 17)
    with adapter_revision.AdapterRevisionRegistry() as registry:
        first = registry.resolve(7, str(source))
        with pytest.raises(RuntimeError, match="task ID collision"):
            registry.resolve(8, str(source))
        assert list(registry._snapshots.values()) == [first]
        assert registry.resolve(7, str(source)) is first


def test_config_replaced_during_weight_copy_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _checkpoint(tmp_path / "adapter")
    original_copy = adapter_revision.AdapterRevisionRegistry._copy_source

    def copy_with_source_change(
        source_file: adapter_revision._SourceFile,
        destination: Path,
        digest: adapter_revision._Digest,
    ) -> None:
        original_copy(source_file, destination, digest)
        if source_file.path.name == "adapter_model.safetensors":
            config = source / "adapter_config.json"
            old_stat = config.stat()
            replacement = source / "replacement.json"
            replacement.write_bytes(b'{"r":4,"lora_alpha":9}')
            os.utime(replacement, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
            replacement.replace(config)

    with adapter_revision.AdapterRevisionRegistry() as registry:
        monkeypatch.setattr(registry, "_copy_source", copy_with_source_change)
        with pytest.raises(RuntimeError, match="changed while snapshotting"):
            registry.resolve(7, str(source))
        assert registry._snapshots == {}
        assert registry._content_paths == {}
        assert list(registry._root.iterdir()) == []

        monkeypatch.setattr(registry, "_copy_source", original_copy)
        snapshot = registry.resolve(7, str(source))
        assert (
            Path(snapshot.path, "adapter_config.json").read_bytes()
            == (source / "adapter_config.json").read_bytes()
        )


def test_failed_copy_does_not_publish_or_remove_existing_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _checkpoint(tmp_path / "adapter")
    with adapter_revision.AdapterRevisionRegistry() as registry:
        first = registry.resolve(7, str(source))
        (source / "adapter_model.safetensors").write_bytes(b"changed adapter bytes")

        def fail_copy(*_args: object) -> None:
            raise OSError("copy failed")

        monkeypatch.setattr(registry, "_copy_source", fail_copy)
        with pytest.raises(OSError, match="copy failed"):
            registry.resolve(7, str(source))
        assert list(registry._snapshots.values()) == [first]
        assert Path(first.path, "adapter_model.safetensors").read_bytes() == b"adapter-weights-A"
        assert list(registry._root.iterdir()) == [Path(first.path)]


def test_safetensors_is_preferred_and_unselected_files_are_not_copied(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "adapter")
    (source / "adapter_model.bin").write_bytes(b"unselected checkpoint")
    with adapter_revision.AdapterRevisionRegistry() as registry:
        snapshot = registry.resolve(7, str(source))
        assert {path.name for path in Path(snapshot.path).iterdir()} == {
            "adapter_config.json",
            "adapter_model.safetensors",
        }


def test_preferred_file_added_during_bin_snapshot_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _checkpoint(tmp_path / "adapter", "adapter_model.bin")
    original_copy = adapter_revision.AdapterRevisionRegistry._copy_source

    def copy_and_add_preferred_file(
        source_file: adapter_revision._SourceFile,
        destination: Path,
        digest: adapter_revision._Digest,
    ) -> None:
        original_copy(source_file, destination, digest)
        if source_file.path.name == "adapter_model.bin":
            (source / "adapter_model.safetensors").write_bytes(b"preferred revision")

    with adapter_revision.AdapterRevisionRegistry() as registry:
        monkeypatch.setattr(registry, "_copy_source", copy_and_add_preferred_file)
        with pytest.raises(RuntimeError, match="changed while snapshotting"):
            registry.resolve(7, str(source))
        assert registry._snapshots == {}
        assert list(registry._root.iterdir()) == []


def test_missing_source_files_do_not_publish(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "adapter")
    (source / "adapter_model.safetensors").unlink()
    with adapter_revision.AdapterRevisionRegistry() as registry:
        with pytest.raises(FileNotFoundError, match="No adapter_model"):
            registry.resolve(7, str(source))
        assert registry._snapshots == {}
        assert list(registry._root.iterdir()) == []


def test_concurrent_resolution_publishes_one_snapshot(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "adapter")
    with adapter_revision.AdapterRevisionRegistry() as registry:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: registry.resolve(7, str(source)), range(8)))
        assert all(result is results[0] for result in results)
        assert len(list(registry._root.iterdir())) == 1


def test_close_releases_all_revisions_and_rejects_further_resolution(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "adapter")
    registry = adapter_revision.AdapterRevisionRegistry()
    first = registry.resolve(7, str(source))
    (source / "adapter_model.safetensors").write_bytes(b"changed adapter bytes")
    second = registry.resolve(7, str(source))
    registry.close()
    registry.close()
    assert not Path(first.path).exists()
    assert not Path(second.path).exists()
    with pytest.raises(RuntimeError, match="registry is closed"):
        registry.resolve(7, str(source))
