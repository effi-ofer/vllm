# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDS (GPU Direct Storage) offloading worker.

Handles FS->GPU transfers using NIXL's GDS backend with VRAM descriptors,
bypassing CPU memory entirely.
"""

import os
import time
from typing import TYPE_CHECKING, NamedTuple

import torch

from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.tiering.gds.common import GDSLoadStoreSpec

if TYPE_CHECKING:
    from nixl._api import nixl_prepped_dlist_handle, nixl_xfer_handle

    from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

logger = init_logger(__name__)

NIXL_READ = "READ"
NIXL_PROC = "PROC"
NIXL_DONE = "DONE"


class _GDSTransferEntry(NamedTuple):
    xfer_handle: "nixl_xfer_handle"
    file_reg: object
    file_handle: "nixl_prepped_dlist_handle"
    fds: list[int]


class GDSOffloadingHandler:
    """Handles FS->GPU transfers using NIXL GDS backend + VRAM descriptors.

    Initialized in the worker process which owns the GPU context. Registers
    GPU KV cache memory as VRAM descriptors and uses the GDS backend to
    read files directly into VRAM.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
    ):
        agent_config = nixl_agent_config(backends=["GDS"])
        self._agent = NixlWrapper("GDSAgent", agent_config)

        self._block_size_factor = block_size_factor
        self._transfers: dict[int, _GDSTransferEntry] = {}
        self._pending_results: list[TransferResult] = []
        self._next_file_dev_id: int = 1

        self._register_vram(kv_caches)

    def _register_vram(self, kv_caches: CanonicalKVCaches) -> None:
        """Register GPU KV cache tensors as VRAM with NIXL."""

        self._gpu_tensors: list[torch.Tensor] = []
        self._gpu_page_sizes: list[int] = []

        for tensor_info in kv_caches.tensors:
            gpu_page_size = tensor_info.page_size_bytes
            t = tensor_info.tensor.view(torch.int8).view(-1, gpu_page_size)
            self._gpu_tensors.append(t)
            self._gpu_page_sizes.append(gpu_page_size)

        # Register full VRAM regions
        vram_data = []
        for t in self._gpu_tensors:
            base_addr = t.data_ptr()
            total_bytes = t.shape[0] * t.shape[1]
            device_id = t.device.index
            vram_data.append((base_addr, total_bytes, device_id, ""))

        self._vram_reg = self._agent.register_memory(vram_data, "VRAM")

        # Build per-block VRAM descriptors for transfer.
        # Each block in the GPU tensor is one descriptor.
        # With block_size_factor > 1, one offloaded block maps to multiple
        # GPU blocks, so we use the offloaded block stride.
        blocks_data: list[tuple[int, int, int]] = []
        self._num_gpu_blocks = self._gpu_tensors[0].shape[0]
        device_id = self._gpu_tensors[0].device.index

        for t in self._gpu_tensors:
            gpu_page_size = t.shape[1]
            offloaded_page_size = gpu_page_size * self._block_size_factor
            num_offloaded_blocks = t.shape[0] // self._block_size_factor
            for block_id in range(num_offloaded_blocks):
                offset = block_id * offloaded_page_size
                addr = t.data_ptr() + offset
                blocks_data.append((addr, offloaded_page_size, device_id))

        descs = self._agent.get_xfer_descs(blocks_data, "VRAM")
        self._vram_prepped_handle: nixl_prepped_dlist_handle = (
            self._agent.prep_xfer_dlist("NIXL_INIT_AGENT", descs)
        )
        self._blocks_per_tensor = (
            self._gpu_tensors[0].shape[0] // self._block_size_factor
        )

    def submit_load(
        self,
        job_id: int,
        gds_spec: GDSLoadStoreSpec,
        gpu_spec: GPULoadStoreSpec,
    ) -> bool:
        """Initiate FS->GPU transfer via NIXL GDS."""
        logger.debug(
            "GDS submit_load job=%d, %d blocks, %d bytes each",
            job_id,
            len(gds_spec.file_paths),
            gds_spec.block_size_bytes,
        )

        # Open files and register with NIXL using file descriptors.
        # The GDS backend requires actual fds, not path strings.
        fds = [os.open(path, os.O_RDONLY) for path in gds_spec.file_paths]
        file_descs = [(0, gds_spec.block_size_bytes, fd, "") for fd in fds]

        file_reg = self._agent.register_memory(file_descs, "FILE")
        if file_reg is None:
            logger.warning("GDS register_memory failed for job %d", job_id)
            for fd in fds:
                os.close(fd)
            self._pending_results.append(TransferResult(job_id=job_id, success=False))
            return False

        file_handle = self._agent.prep_xfer_dlist("GDSAgent", file_reg.trim())
        if not file_handle:
            logger.warning("GDS prep_xfer_dlist failed for job %d", job_id)
            self._agent.deregister_memory(file_reg)
            for fd in fds:
                os.close(fd)
            self._pending_results.append(TransferResult(job_id=job_id, success=False))
            return False

        num_files = len(gds_spec.file_paths)
        vram_ids = self._compute_vram_ids(gpu_spec, num_files)
        file_ids = list(range(num_files))

        xfer_handle = self._agent.make_prepped_xfer(
            NIXL_READ,
            self._vram_prepped_handle,
            vram_ids,
            file_handle,
            file_ids,
        )
        if not xfer_handle:
            logger.warning("GDS make_prepped_xfer failed for job %d", job_id)
            self._agent.release_dlist_handle(file_handle)
            self._agent.deregister_memory(file_reg)
            for fd in fds:
                os.close(fd)
            self._pending_results.append(TransferResult(job_id=job_id, success=False))
            return False

        state = self._agent.transfer(xfer_handle)
        if state == "ERR":
            logger.warning("GDS transfer failed for job %d", job_id)
            self._agent.release_xfer_handle(xfer_handle)
            self._agent.release_dlist_handle(file_handle)
            self._agent.deregister_memory(file_reg)
            for fd in fds:
                os.close(fd)
            self._pending_results.append(TransferResult(job_id=job_id, success=False))
            return False

        self._transfers[job_id] = _GDSTransferEntry(
            xfer_handle, file_reg, file_handle, fds
        )
        return True

    def _compute_vram_ids(
        self, gpu_spec: GPULoadStoreSpec, num_files: int
    ) -> list[int]:
        """Map GPU block IDs to VRAM descriptor indices.

        With block_size_factor > 1, multiple GPU blocks correspond to one
        offloaded block. The VRAM descriptors are at offloaded-block
        granularity. We deduplicate and return unique offloaded block indices,
        matching the number of files being transferred.
        """
        block_ids = gpu_spec.block_ids
        if self._block_size_factor == 1:
            return block_ids[:num_files].tolist()

        # Map GPU block IDs to offloaded block IDs and deduplicate
        offloaded_ids = block_ids // self._block_size_factor
        seen: set[int] = set()
        result: list[int] = []
        for oid in offloaded_ids:
            oid_int = int(oid)
            if oid_int not in seen:
                seen.add(oid_int)
                result.append(oid_int)
                if len(result) == num_files:
                    break
        return result

    def get_finished(self) -> list[TransferResult]:
        """Poll pending GDS transfers for completion."""
        self._poll_active_transfers()
        results = self._pending_results
        self._pending_results = []
        return results

    def _poll_active_transfers(self) -> None:
        for job_id, entry in list(self._transfers.items()):
            try:
                state = self._agent.check_xfer_state(entry.xfer_handle)
            except Exception as exc:
                success = False
                logger.warning(
                    "GDS check_xfer_state raised for job %d: %s", job_id, exc
                )
            else:
                if state == NIXL_PROC:
                    continue
                elif state == NIXL_DONE:
                    success = True
                else:
                    success = False
                    logger.warning("GDS transfer failed job=%d state=%s", job_id, state)
            del self._transfers[job_id]
            self._agent.release_xfer_handle(entry.xfer_handle)
            self._agent.release_dlist_handle(entry.file_handle)
            self._agent.deregister_memory(entry.file_reg)
            for fd in entry.fds:
                os.close(fd)
            self._pending_results.append(TransferResult(job_id=job_id, success=success))

    def has_pending(self) -> bool:
        return bool(self._transfers)

    def wait(self, job_ids: set[int]) -> None:
        """Block until specified GDS jobs complete."""
        while any(jid in self._transfers for jid in job_ids):
            self._poll_active_transfers()
            if any(jid in self._transfers for jid in job_ids):
                time.sleep(0.001)

    def shutdown(self) -> None:
        import contextlib

        for job_id, entry in self._transfers.items():
            with contextlib.suppress(Exception):
                self._agent.release_xfer_handle(entry.xfer_handle)
            with contextlib.suppress(Exception):
                self._agent.release_dlist_handle(entry.file_handle)
            with contextlib.suppress(Exception):
                self._agent.deregister_memory(entry.file_reg)
            for fd in entry.fds:
                with contextlib.suppress(Exception):
                    os.close(fd)
        self._transfers.clear()
        if self._vram_prepped_handle is not None:
            with contextlib.suppress(Exception):
                self._agent.release_dlist_handle(self._vram_prepped_handle)
        if self._vram_reg is not None:
            with contextlib.suppress(Exception):
                self._agent.deregister_memory(self._vram_reg)


class GDSCapableOffloadingWorker(OffloadingWorker):
    """Composite worker: dispatches to CPU or GDS handler based on spec type."""

    def __init__(
        self,
        cpu_worker: "CPUOffloadingWorker",
        gds_handler: GDSOffloadingHandler,
    ):
        self._cpu = cpu_worker
        self._gds = gds_handler

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        return self._cpu.submit_store(job_id, src_spec, dst_spec)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        if isinstance(src_spec, GDSLoadStoreSpec):
            return self._gds.submit_load(job_id, src_spec, dst_spec)
        return self._cpu.submit_load(job_id, src_spec, dst_spec)

    def get_finished(self) -> list[TransferResult]:
        return self._cpu.get_finished() + self._gds.get_finished()

    def wait(self, job_ids: set[int]) -> None:
        self._cpu.wait(job_ids)
        self._gds.wait(job_ids)

    def shutdown(self) -> None:
        self._cpu.shutdown()
        self._gds.shutdown()
