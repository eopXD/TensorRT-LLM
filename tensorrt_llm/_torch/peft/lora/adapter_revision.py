# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable checkpoint files and task identities for runtime adapter revisions."""

import hashlib
import os
import shutil
import stat
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Protocol

_COPY_CHUNK_BYTES = 1024 * 1024
_MAX_TASK_ID = (1 << 63) - 1


class _Digest(Protocol):
    def update(self, data: bytes) -> None: ...


@dataclass(frozen=True)
class AdapterSnapshot:
    """An adapter incarnation whose files remain available until registry cleanup."""

    task_id: int
    revision: str
    path: str


@dataclass(frozen=True)
class _FileSignature:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_FileSignature":
        return cls(value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


@dataclass(frozen=True)
class _SourceFile:
    path: Path
    signature: _FileSignature


@dataclass(frozen=True)
class _SourceFiles:
    config: _SourceFile
    weights: _SourceFile


def _source_file(path: Path) -> _SourceFile:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"LoRA checkpoint file is not a regular file: {path}")
    return _SourceFile(path, _FileSignature.from_stat(info))


def _source_files(directory: Path) -> _SourceFiles:
    config = _source_file(directory / "adapter_config.json")
    for filename in ("adapter_model.safetensors", "adapter_model.bin"):
        try:
            weights = _source_file(directory / filename)
        except FileNotFoundError:
            continue
        return _SourceFiles(config, weights)
    raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin in {directory}")


def _task_id(external_adapter_id: int, revision: str) -> int:
    digest = hashlib.sha256(
        f"trtllm.component-lora.task.v1\0{external_adapter_id}\0{revision}".encode()
    ).digest()
    return (int.from_bytes(digest[:8], "big") & _MAX_TASK_ID) or 1


def _check_open_file(source: _SourceFile, stream: BinaryIO) -> None:
    if _FileSignature.from_stat(os.fstat(stream.fileno())) != source.signature:
        raise RuntimeError(f"LoRA checkpoint changed while snapshotting: {source.path}")


def _hash_file_header(source: _SourceFile, digest: _Digest) -> None:
    filename = source.path.name.encode()
    digest.update(len(filename).to_bytes(8, "big"))
    digest.update(filename)
    digest.update(source.signature.size.to_bytes(8, "big"))


class AdapterRevisionRegistry:
    """Bind external adapter IDs to immutable, same-host checkpoint snapshots.

    Published revisions remain available for active requests and cache eviction reloads.
    Close the registry only after its worker has stopped using those paths.
    """

    def __init__(self) -> None:
        self._directory = tempfile.TemporaryDirectory(prefix="trtllm-component-lora-")
        self._root = Path(self._directory.name)
        self._lock = threading.Lock()
        self._closed = False
        self._content_paths: dict[str, Path] = {}
        self._snapshots: dict[tuple[int, str], AdapterSnapshot] = {}
        self._task_identities: dict[int, tuple[int, str]] = {}

    def resolve(self, external_adapter_id: int, path: str) -> AdapterSnapshot:
        """Hash the current HF checkpoint and reuse or snapshot that exact content."""
        if not isinstance(external_adapter_id, int) or isinstance(external_adapter_id, bool):
            raise TypeError("external_adapter_id must be an integer")
        with self._lock:
            if self._closed:
                raise RuntimeError("Adapter revision registry is closed")
            source_directory = Path(path).expanduser().resolve()
            sources = _source_files(source_directory)
            # Same-size rewrites can share timestamps on coarse-grained filesystems.
            # Content, rather than stat equality, determines the adapter revision.
            digest = hashlib.sha256(b"trtllm.component-lora.snapshot.v1\0")
            for source in (sources.config, sources.weights):
                _hash_file_header(source, digest)
                self._hash_source(source, digest)
            if _source_files(source_directory) != sources:
                raise RuntimeError(
                    f"LoRA checkpoint changed while snapshotting: {source_directory}"
                )
            revision = digest.hexdigest()
            self._check_task_identity(external_adapter_id, revision)
            if revision in self._content_paths:
                return self._publish(external_adapter_id, revision)

            pending = Path(tempfile.mkdtemp(prefix="pending-", dir=self._root))
            try:
                digest = hashlib.sha256(b"trtllm.component-lora.snapshot.v1\0")
                for source in (sources.config, sources.weights):
                    _hash_file_header(source, digest)
                    self._copy_source(source, pending / source.path.name, digest)
                if _source_files(source_directory) != sources or digest.hexdigest() != revision:
                    raise RuntimeError(
                        f"LoRA checkpoint changed while snapshotting: {source_directory}"
                    )
                published = self._root / revision
                pending.rename(published)
                self._content_paths[revision] = published
                return self._publish(external_adapter_id, revision)
            finally:
                if pending.exists():
                    shutil.rmtree(pending)

    @staticmethod
    def _hash_source(source: _SourceFile, digest: _Digest) -> None:
        with source.path.open("rb") as incoming:
            _check_open_file(source, incoming)
            while chunk := incoming.read(_COPY_CHUNK_BYTES):
                digest.update(chunk)
            _check_open_file(source, incoming)

    @staticmethod
    def _copy_source(source: _SourceFile, destination: Path, digest: _Digest) -> None:
        with source.path.open("rb") as incoming, destination.open("xb") as outgoing:
            _check_open_file(source, incoming)
            while chunk := incoming.read(_COPY_CHUNK_BYTES):
                outgoing.write(chunk)
                digest.update(chunk)
            _check_open_file(source, incoming)
        destination.chmod(0o444)

    def _check_task_identity(self, external_adapter_id: int, revision: str) -> int:
        identity = (external_adapter_id, revision)
        task_id = _task_id(*identity)
        existing = self._task_identities.get(task_id)
        if existing is not None and existing != identity:
            raise RuntimeError(
                f"LoRA task ID collision for external adapter {external_adapter_id}, "
                f"revision {revision}: task {task_id} already identifies {existing}"
            )
        return task_id

    def _publish(self, external_adapter_id: int, revision: str) -> AdapterSnapshot:
        identity = (external_adapter_id, revision)
        snapshot = self._snapshots.get(identity)
        if snapshot is None:
            task_id = self._check_task_identity(*identity)
            snapshot = AdapterSnapshot(task_id, revision, str(self._content_paths[revision]))
            self._task_identities[task_id] = identity
            self._snapshots[identity] = snapshot
        return snapshot

    def close(self) -> None:
        """Release all snapshots after outstanding requests have finished."""
        with self._lock:
            if not self._closed:
                self._directory.cleanup()
                self._content_paths.clear()
                self._snapshots.clear()
                self._task_identities.clear()
                self._closed = True

    def __enter__(self) -> "AdapterRevisionRegistry":
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()
