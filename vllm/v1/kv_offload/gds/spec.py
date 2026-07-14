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
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.gds.manager import GDSOffloadingManager
from vllm.v1.kv_offload.gds.worker import GDSOffloadingWorker


class GDSOffloadingSpec(OffloadingSpec):
    """Spec for GPU Direct Storage offloading via cuFile async API."""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        self.gds_root_dir: str = self.extra_config.get("gds_root_dir", "/tmp/vllm_gds")

        self._manager: GDSOffloadingManager | None = None
        self._worker: GDSOffloadingWorker | None = None

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            self._manager = GDSOffloadingManager(
                enable_events=self.kv_events_config.enable_kv_cache_events,
            )
        return self._manager

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if not self._worker:
            if not current_platform.is_cuda_alike():
                raise RuntimeError("GDS offloading requires a CUDA-capable GPU.")
            file_mapper = FileMapper.from_offloading_spec(
                root_dir=self.gds_root_dir,
                offloading_spec=self,
            )
            self._worker = GDSOffloadingWorker(
                kv_caches=kv_caches,
                block_size_factor=self.block_size_factor,
                file_mapper=file_mapper,
            )
        return self._worker
