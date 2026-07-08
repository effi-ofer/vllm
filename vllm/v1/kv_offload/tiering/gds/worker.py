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
        logger.info(
            "GDS _register_vram: %d tensors, page_sizes=%s, bsf=%d",
            len(kv_caches.tensors),
            [t.page_size_bytes for t in kv_caches.tensors],
            self._block_size_factor,
        )

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
        # Each offloaded block on disk contains data for ALL tensors
        # concatenated. We register one VRAM descriptor per (tensor, block)
        # pair. Layout: tensor0_block0, tensor0_block1, ..., tensor1_block0, ...
        blocks_data: list[tuple[int, int, int]] = []
        self._num_gpu_blocks = self._gpu_tensors[0].shape[0]
        device_id = self._gpu_tensors[0].device.index
        self._num_tensors = len(self._gpu_tensors)

        # Per-tensor offloaded page sizes (for file offset calculation)
        self._offloaded_page_sizes: list[int] = []
        for t in self._gpu_tensors:
            gpu_page_size = t.shape[1]
            offloaded_page_size = gpu_page_size * self._block_size_factor
            self._offloaded_page_sizes.append(offloaded_page_size)
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

    # TODO: handle GDS failures rather than assert
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
        # Each file contains data for ALL tensors concatenated. We register
        # one FILE descriptor per (file, tensor) pair with the correct offset.
        fds = [os.open(path, os.O_RDONLY) for path in gds_spec.file_paths]
        num_files = len(gds_spec.file_paths)

        # Build FILE descriptors: for each file, one per tensor at the
        # correct offset within the file.
        file_descs = []
        for fd in fds:
            file_offset = 0
            for tensor_page_size in self._offloaded_page_sizes:
                file_descs.append((file_offset, tensor_page_size, fd, ""))
                file_offset += tensor_page_size

        file_reg = self._agent.register_memory(file_descs, "FILE")
        assert file_reg is not None, f"GDS register_memory failed for job {job_id}"

        file_handle = self._agent.prep_xfer_dlist("GDSAgent", file_reg.trim())
        assert file_handle, f"GDS prep_xfer_dlist failed for job {job_id}"

        # Build matched ID lists: for each file, pair each tensor's FILE
        # descriptor with the corresponding VRAM descriptor.
        vram_ids = self._compute_vram_ids(gpu_spec, num_files)
        file_ids = list(range(num_files * self._num_tensors))
        # Expand vram_ids: for each offloaded block, add IDs for all tensors.
        # VRAM layout: tensor0 blocks [0..N-1], tensor1 blocks [N..2N-1], ...
        expanded_vram_ids = []
        for block_id in vram_ids:
            for tensor_idx in range(self._num_tensors):
                expanded_vram_ids.append(
                    tensor_idx * self._blocks_per_tensor + block_id
                )
        vram_ids = expanded_vram_ids

        xfer_handle = self._agent.make_prepped_xfer(
            NIXL_READ,
            self._vram_prepped_handle,
            vram_ids,
            file_handle,
            file_ids,
        )
        assert xfer_handle, f"GDS make_prepped_xfer failed for job {job_id}"

        state = self._agent.transfer(xfer_handle)
        assert state != "ERR", f"GDS transfer failed for job {job_id}"

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
                raise RuntimeError(
                    f"GDS check_xfer_state raised for job {job_id}"
                ) from exc
            else:
                if state == NIXL_PROC:
                    continue
                elif state == NIXL_DONE:
                    success = True
                else:
                    raise RuntimeError(
                        f"GDS transfer failed job={job_id} state={state}"
                    )
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
