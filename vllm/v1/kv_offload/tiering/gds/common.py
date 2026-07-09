# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDS (GPU Direct Storage) LoadStoreSpec for filesystem-to-GPU transfers."""

from dataclasses import dataclass

from typing_extensions import override

from vllm.v1.kv_offload.base import LoadStoreSpec


@dataclass
class GDSLoadStoreSpec(LoadStoreSpec):
    """Spec for loading KV blocks directly from filesystem to GPU via GDS.

    The scheduler produces this when GDS is available and blocks are in the
    FS tier. The worker receives it and uses NIXL GDS backend + VRAM to
    execute the FS->GPU transfer, bypassing CPU entirely.
    """

    file_paths: list[str]
    block_size_bytes: int
    slot_indices: list[int]

    @staticmethod
    @override
    def medium() -> str:
        return "GDS"
