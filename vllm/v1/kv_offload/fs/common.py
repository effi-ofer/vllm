# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing_extensions import override

from vllm.v1.kv_offload.base import LoadStoreSpec, OffloadKey


class FSLoadStoreSpec(LoadStoreSpec):
    """Spec for loading/storing KV blocks to a filesystem.

    Carries OffloadKeys that the worker resolves to file paths
    via FileMapper.
    """

    def __init__(self, keys: list[OffloadKey]):
        self.keys = keys

    @staticmethod
    @override
    def medium() -> str:
        return "FS"
