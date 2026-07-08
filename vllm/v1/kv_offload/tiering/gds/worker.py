# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDS (GPU Direct Storage) offloading worker.

Handles FS->GPU transfers using NIXL's GDS backend with a bounce buffer,
bypassing CPU memory entirely. Files are read into a contiguous GPU bounce
buffer via GDS, then scattered to the individual KV cache tensors.
"""

import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch

from vllm import _custom_ops as ops
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
    from nixl._api import nixl_xfer_handle

    from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

logger = init_logger(__name__)

NIXL_READ = "READ"
NIXL_PROC = "PROC"
NIXL_DONE = "DONE"

_DEFAULT_BOUNCE_SLOTS = 64


class _TransferPhase(Enum):
    GDS_IN_PROGRESS = auto()
    SCATTER_IN_PROGRESS = auto()


@dataclass
class _GDSTransferEntry:
    job_id: int
    phase: _TransferPhase
    # GDS phase
    xfer_handle: "nixl_xfer_handle | None" = None
    file_reg: object = None
    fds: list[int] = field(default_factory=list)
    slot_indices: list[int] = field(default_factory=list)
    # Block IDs for scatter (one per file, at offloaded-block granularity)
    offloaded_block_ids: list[int] = field(default_factory=list)
    # Scatter phase
    scatter_event: "torch.Event | None" = None


class GDSOffloadingHandler:
    """Handles FS->GPU transfers using NIXL GDS backend + bounce buffer.

    Files are read into a contiguous GPU bounce buffer via GDS (one
    descriptor pair per file). After the GDS read completes, data is
    scattered from the bounce buffer to the individual KV cache tensors
    using GPU-to-GPU copies.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        num_bounce_slots: int = _DEFAULT_BOUNCE_SLOTS,
    ):
        self._block_size_factor = block_size_factor
        self._transfers: dict[int, _GDSTransferEntry] = {}
        self._pending_results: list[TransferResult] = []

        # Store GPU tensor references for scatter
        self._gpu_tensors: list[torch.Tensor] = []
        self._offloaded_page_sizes: list[int] = []
        for tensor_info in kv_caches.tensors:
            gpu_page_size = tensor_info.page_size_bytes
            t = tensor_info.tensor.view(torch.int8).view(-1, gpu_page_size)
            self._gpu_tensors.append(t)
            self._offloaded_page_sizes.append(gpu_page_size * block_size_factor)

        self._num_tensors = len(self._gpu_tensors)
        self._block_size_bytes = sum(self._offloaded_page_sizes)
        self._device = self._gpu_tensors[0].device

        logger.info(
            "GDS bounce buffer: %d tensors, block_size_bytes=%d, "
            "num_slots=%d, total_buffer=%.0f MB",
            self._num_tensors,
            self._block_size_bytes,
            num_bounce_slots,
            num_bounce_slots * self._block_size_bytes / (1 << 20),
        )

        # Allocate contiguous GPU bounce buffer
        self._num_slots = num_bounce_slots
        self._bounce_buffer = torch.empty(
            self._num_slots * self._block_size_bytes,
            dtype=torch.int8,
            device=self._device,
        )
        self._free_slots: list[int] = list(range(self._num_slots))

        # NIXL setup — register only the bounce buffer
        agent_config = nixl_agent_config(
            backends=["GDS"],
            custom_params={"batch_limit": "128", "batch_pool_size": "64"},
        )
        self._agent = NixlWrapper("GDSAgent", agent_config)

        device_id = self._device.index
        vram_data = [
            (
                self._bounce_buffer.data_ptr(),
                self._bounce_buffer.numel(),
                device_id,
                "",
            )
        ]
        self._vram_reg = self._agent.register_memory(vram_data, "VRAM")

        self._device_id = device_id

        # CUDA stream for scatter operations
        self._scatter_stream = torch.cuda.Stream(device=self._device)

    def submit_load(
        self,
        job_id: int,
        gds_spec: GDSLoadStoreSpec,
        gpu_spec: GPULoadStoreSpec,
    ) -> bool:
        """Initiate FS->GPU transfer via GDS bounce buffer."""
        num_files = len(gds_spec.file_paths)
        logger.debug(
            "GDS submit_load job=%d, %d files, %d bytes each",
            job_id,
            num_files,
            gds_spec.block_size_bytes,
        )

        # Compute destination offloaded block IDs
        offloaded_block_ids = self._compute_offloaded_block_ids(gpu_spec, num_files)

        # Acquire bounce buffer slots
        if len(self._free_slots) < num_files:
            logger.warning(
                "GDS job=%d needs %d slots but only %d free, deferring",
                job_id,
                num_files,
                len(self._free_slots),
            )
            self._pending_results.append(TransferResult(job_id=job_id, success=False))
            return False

        slots = [self._free_slots.pop() for _ in range(num_files)]

        # Open files and register with NIXL
        fds = [os.open(path, os.O_RDONLY | os.O_DIRECT) for path in gds_spec.file_paths]

        file_descs = [(0, gds_spec.block_size_bytes, fd, "") for fd in fds]
        file_reg = self._agent.register_memory(file_descs, "FILE")
        assert file_reg is not None, (
            f"GDS register_memory(FILE) failed for job {job_id}"
        )

        # Build VRAM descriptors for just the slots being used
        vram_descs_data: list[tuple[int, int, int]] = []
        for slot in slots:
            addr = self._bounce_buffer.data_ptr() + slot * self._block_size_bytes
            vram_descs_data.append((addr, self._block_size_bytes, self._device_id))
        vram_descs = self._agent.get_xfer_descs(vram_descs_data, "VRAM")

        # Use initialize_xfer (matches descriptors 1:1)
        file_xfer_descs = file_reg.trim()

        logger.debug(
            "GDS submit_load: slots=%s, vram_descs_data=%s, "
            "file_descs=%s, num_vram=%d, num_file=%d",
            slots[:5],
            vram_descs_data[:3],
            file_descs[:3],
            len(vram_descs_data),
            len(file_descs),
        )

        xfer_handle = self._agent.initialize_xfer(
            NIXL_READ, vram_descs, file_xfer_descs, "GDSAgent"
        )

        logger.debug(
            "GDS submit_load: xfer_handle=%s, vram_descs=%s, file_xfer_descs=%s",
            xfer_handle,
            vram_descs,
            file_xfer_descs,
        )
        assert xfer_handle, f"GDS initialize_xfer failed for job {job_id}"

        logger.debug("   -------------------- before transfer")
        state = self._agent.transfer(xfer_handle)
        assert state != "ERR", f"GDS transfer failed for job {job_id}"
        logger.debug("   -------------------- after transfer")

        self._transfers[job_id] = _GDSTransferEntry(
            job_id=job_id,
            phase=_TransferPhase.GDS_IN_PROGRESS,
            xfer_handle=xfer_handle,
            file_reg=file_reg,
            fds=fds,
            slot_indices=slots,
            offloaded_block_ids=offloaded_block_ids,
        )
        return True

    def _compute_offloaded_block_ids(
        self, gpu_spec: GPULoadStoreSpec, num_files: int
    ) -> list[int]:
        """Map GPU block IDs to offloaded block indices."""
        block_ids = gpu_spec.block_ids
        if self._block_size_factor == 1:
            return block_ids[:num_files].tolist()

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

    def _scatter_to_tensors(
        self, slots: list[int], offloaded_block_ids: list[int]
    ) -> torch.Event:
        """Scatter data from bounce buffer slots to KV cache tensors.

        Returns a CUDA event that signals when the scatter is complete.
        """
        num_copies = len(slots) * self._num_tensors
        src_ptrs = torch.empty(num_copies, dtype=torch.int64, pin_memory=True)
        dst_ptrs = torch.empty(num_copies, dtype=torch.int64, pin_memory=True)
        sizes = torch.empty(num_copies, dtype=torch.int64, pin_memory=True)

        idx = 0
        for slot, block_id in zip(slots, offloaded_block_ids):
            bounce_base = self._bounce_buffer.data_ptr() + slot * self._block_size_bytes
            tensor_offset_in_file = 0
            for tensor_idx in range(self._num_tensors):
                page_size = self._offloaded_page_sizes[tensor_idx]
                src_addr = bounce_base + tensor_offset_in_file
                dst_addr = (
                    self._gpu_tensors[tensor_idx].data_ptr() + block_id * page_size
                )

                src_ptrs[idx] = src_addr
                dst_ptrs[idx] = dst_addr
                sizes[idx] = page_size
                tensor_offset_in_file += page_size
                idx += 1

        event = torch.Event(enable_timing=False)
        with torch.cuda.stream(self._scatter_stream):
            ops.swap_blocks_batch(
                src_ptrs, dst_ptrs, sizes, is_src_access_order_any=True
            )
            event.record(self._scatter_stream)
        return event

    def get_finished(self) -> list[TransferResult]:
        """Poll pending GDS transfers for completion."""
        self._poll_active_transfers()
        results = self._pending_results
        self._pending_results = []
        return results

    def _poll_active_transfers(self) -> None:
        for job_id, entry in list(self._transfers.items()):
            if entry.phase == _TransferPhase.GDS_IN_PROGRESS:
                self._poll_gds_phase(job_id, entry)
            elif entry.phase == _TransferPhase.SCATTER_IN_PROGRESS:
                self._poll_scatter_phase(job_id, entry)

    def _poll_gds_phase(self, job_id: int, entry: _GDSTransferEntry) -> None:
        try:
            state = self._agent.check_xfer_state(entry.xfer_handle)
        except Exception as exc:
            raise RuntimeError(f"GDS check_xfer_state raised for job {job_id}") from exc

        if state == NIXL_PROC:
            return

        if state != NIXL_DONE:
            raise RuntimeError(f"GDS transfer failed job={job_id} state={state}")

        # GDS read complete — clean up NIXL resources
        self._agent.release_xfer_handle(entry.xfer_handle)
        self._agent.deregister_memory(entry.file_reg)
        for fd in entry.fds:
            os.close(fd)
        entry.xfer_handle = None
        entry.file_reg = None
        entry.fds = []

        # Start scatter phase
        event = self._scatter_to_tensors(entry.slot_indices, entry.offloaded_block_ids)
        entry.scatter_event = event
        entry.phase = _TransferPhase.SCATTER_IN_PROGRESS

    def _poll_scatter_phase(self, job_id: int, entry: _GDSTransferEntry) -> None:
        assert entry.scatter_event is not None
        if not entry.scatter_event.query():
            return

        # Scatter complete — release slots and report done
        self._free_slots.extend(entry.slot_indices)
        del self._transfers[job_id]
        self._pending_results.append(TransferResult(job_id=job_id, success=True))

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

        for entry in self._transfers.values():
            if entry.xfer_handle is not None:
                with contextlib.suppress(Exception):
                    self._agent.release_xfer_handle(entry.xfer_handle)
            if entry.file_reg is not None:
                with contextlib.suppress(Exception):
                    self._agent.deregister_memory(entry.file_reg)
            for fd in entry.fds:
                with contextlib.suppress(Exception):
                    os.close(fd)
        self._transfers.clear()

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
