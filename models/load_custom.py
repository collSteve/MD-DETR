# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Loading of Deformable DETR's CUDA kernels"""
from pathlib import Path


def _resolve_kernel_root() -> Path:
    candidates = [
        Path(__file__).resolve().parent.parent / "kernels" / "deformable_detr",
        Path(__file__).resolve().parent.parent.parent / "kernels" / "deformable_detr",
    ]
    try:
        import transformers
        candidates.append(Path(transformers.__file__).resolve().parent / "kernels" / "deta")
    except ImportError:
        pass

    for c in candidates:
        if (c / "vision.cpp").exists():
            return c
    raise FileNotFoundError(
        f"Deformable attention kernel sources not found. Tried: {[str(c) for c in candidates]}"
    )


def load_cuda_kernels():
    from torch.utils.cpp_extension import load

    root = _resolve_kernel_root()
    src_files = [
        root / "vision.cpp",
        root / "cpu" / "ms_deform_attn_cpu.cpp",
        root / "cuda" / "ms_deform_attn_cuda.cu",
    ]

    load(
        "MultiScaleDeformableAttention",
        src_files,
        with_cuda=True,
        extra_include_paths=[str(root)],
        extra_cflags=["-DWITH_CUDA=1"],
        extra_cuda_cflags=[
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],
    )

    import MultiScaleDeformableAttention as MSDA

    return MSDA
