# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDS offloading spec — factory for GDS manager and worker."""

from typing_extensions import override

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
    OffloadingWorker,
)
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.gds.manager import GDSOffloadingManager
from vllm.v1.kv_offload.gds.worker import GDSOffloadingWorker


class GDSOffloadingSpec(CPUOffloadingSpec):
    """Spec for GPU Direct Storage offloading via cuFile async API."""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        # GDS doesn't need cpu_bytes_to_use but CPUOffloadingSpec requires it.
        # Inject a dummy value so the parent __init__ can compute block sizing.
        assert vllm_config.kv_transfer_config is not None
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        if "cpu_bytes_to_use" not in extra_config:
            extra_config["cpu_bytes_to_use"] = 1

        super().__init__(vllm_config, kv_cache_config)

        self.gds_root_dir: str = self.extra_config.get("gds_root_dir", "/tmp/vllm_gds")
        self._gds_worker: GDSOffloadingWorker | None = None

    def _get_file_mapper(self) -> FileMapper:
        return FileMapper.from_offloading_spec(
            root_dir=self.gds_root_dir,
            offloading_spec=self,
        )

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            self._manager = GDSOffloadingManager(
                file_mapper=self._get_file_mapper(),
                enable_events=self.kv_events_config.enable_kv_cache_events,
            )
        return self._manager

    @override
    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        raise NotImplementedError("GDS uses get_worker() directly")

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if not self._gds_worker:
            if not current_platform.is_cuda_alike():
                raise RuntimeError("GDS offloading requires a CUDA-capable GPU.")
            file_mapper = FileMapper.from_offloading_spec(
                root_dir=self.gds_root_dir,
                offloading_spec=self,
            )
            self._gds_worker = GDSOffloadingWorker(
                kv_caches=kv_caches,
                block_size_factor=self.block_size_factor,
                file_mapper=file_mapper,
            )
        return self._gds_worker
