"""Case-8 cuBLASLt Linear2 wrapper with an unchanged BF16 output boundary."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional

import torch


M = 8192
K = 1024
N = 1024
WORKSPACE_CAP_BYTES = 16 * 1024 * 1024

_CUDA_R_32F = 0
_CUDA_R_16BF = 14
_CUBLAS_COMPUTE_32F = 68
_CUBLAS_OP_N = 0
_CUBLAS_OP_T = 1
_CUBLASLT_MATMUL_DESC_TRANSA = 3
_CUBLASLT_MATMUL_DESC_TRANSB = 4
_CUBLASLT_MATMUL_DESC_EPILOGUE = 7
_CUBLASLT_MATMUL_DESC_BIAS_POINTER = 8
_CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = 26
_CUBLASLT_EPILOGUE_BIAS = 4
_CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1

_STATUS_NAMES = {
    0: "CUBLAS_STATUS_SUCCESS",
    1: "CUBLAS_STATUS_NOT_INITIALIZED",
    3: "CUBLAS_STATUS_ALLOC_FAILED",
    7: "CUBLAS_STATUS_INVALID_VALUE",
    8: "CUBLAS_STATUS_ARCH_MISMATCH",
    11: "CUBLAS_STATUS_MAPPING_ERROR",
    13: "CUBLAS_STATUS_EXECUTION_FAILED",
    14: "CUBLAS_STATUS_INTERNAL_ERROR",
    15: "CUBLAS_STATUS_NOT_SUPPORTED",
}


class _Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 8)]


class _HeuristicResult(ctypes.Structure):
    _fields_ = [
        ("algo", _Algo),
        ("workspace_size", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("waves_count", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    ]


def _check(status: int, operation: str) -> None:
    if status != 0:
        name = _STATUS_NAMES.get(status, "unknown status")
        raise RuntimeError(f"{operation} failed: {name} ({status})")


def _library_path() -> Path:
    site_packages = Path(torch.__file__).resolve().parent.parent
    candidates = sorted(site_packages.glob("nvidia/cu*/lib/libcublasLt.so.*"))
    if not candidates:
        raise RuntimeError("the pinned environment does not contain libcublasLt")
    return candidates[-1]


def _configure_library(library: ctypes.CDLL) -> None:
    pointer = ctypes.c_void_p
    library.cublasLtCreate.argtypes = [ctypes.POINTER(pointer)]
    library.cublasLtCreate.restype = ctypes.c_int
    library.cublasLtDestroy.argtypes = [pointer]
    library.cublasLtDestroy.restype = ctypes.c_int
    library.cublasLtMatmulDescCreate.argtypes = [
        ctypes.POINTER(pointer),
        ctypes.c_int,
        ctypes.c_int,
    ]
    library.cublasLtMatmulDescCreate.restype = ctypes.c_int
    library.cublasLtMatmulDescDestroy.argtypes = [pointer]
    library.cublasLtMatmulDescDestroy.restype = ctypes.c_int
    library.cublasLtMatmulDescSetAttribute.argtypes = [
        pointer,
        ctypes.c_int,
        pointer,
        ctypes.c_size_t,
    ]
    library.cublasLtMatmulDescSetAttribute.restype = ctypes.c_int
    library.cublasLtMatrixLayoutCreate.argtypes = [
        ctypes.POINTER(pointer),
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int64,
    ]
    library.cublasLtMatrixLayoutCreate.restype = ctypes.c_int
    library.cublasLtMatrixLayoutDestroy.argtypes = [pointer]
    library.cublasLtMatrixLayoutDestroy.restype = ctypes.c_int
    library.cublasLtMatmulPreferenceCreate.argtypes = [ctypes.POINTER(pointer)]
    library.cublasLtMatmulPreferenceCreate.restype = ctypes.c_int
    library.cublasLtMatmulPreferenceDestroy.argtypes = [pointer]
    library.cublasLtMatmulPreferenceDestroy.restype = ctypes.c_int
    library.cublasLtMatmulPreferenceSetAttribute.argtypes = [
        pointer,
        ctypes.c_int,
        pointer,
        ctypes.c_size_t,
    ]
    library.cublasLtMatmulPreferenceSetAttribute.restype = ctypes.c_int
    library.cublasLtMatmulAlgoGetHeuristic.argtypes = [
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        ctypes.c_int,
        ctypes.POINTER(_HeuristicResult),
        ctypes.POINTER(ctypes.c_int),
    ]
    library.cublasLtMatmulAlgoGetHeuristic.restype = ctypes.c_int
    library.cublasLtMatmul.argtypes = [
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        ctypes.POINTER(_Algo),
        pointer,
        ctypes.c_size_t,
        pointer,
    ]
    library.cublasLtMatmul.restype = ctypes.c_int


class CublasLtLinear:
    """One-shape BF16 GEMM+bias wrapper with a cached vendor heuristic."""

    def __init__(self) -> None:
        self._library = ctypes.CDLL(str(_library_path()))
        _configure_library(self._library)
        self._handle = ctypes.c_void_p()
        self._operation = ctypes.c_void_p()
        self._a_layout = ctypes.c_void_p()
        self._b_layout = ctypes.c_void_p()
        self._c_layout = ctypes.c_void_p()
        self._d_layout = ctypes.c_void_p()
        self._preference = ctypes.c_void_p()
        self._heuristic: Optional[_HeuristicResult] = None
        self._workspace: Optional[torch.Tensor] = None
        self._device_index: Optional[int] = None
        self._closed = False
        self._create_descriptors()

    def _set_operation_attribute(self, attribute: int, value: object) -> None:
        _check(
            self._library.cublasLtMatmulDescSetAttribute(
                self._operation,
                attribute,
                ctypes.byref(value),
                ctypes.sizeof(value),
            ),
            f"set matmul descriptor attribute {attribute}",
        )

    def _create_descriptors(self) -> None:
        _check(self._library.cublasLtCreate(ctypes.byref(self._handle)), "create")
        _check(
            self._library.cublasLtMatmulDescCreate(
                ctypes.byref(self._operation),
                _CUBLAS_COMPUTE_32F,
                _CUDA_R_32F,
            ),
            "create matmul descriptor",
        )
        self._set_operation_attribute(
            _CUBLASLT_MATMUL_DESC_TRANSA, ctypes.c_int(_CUBLAS_OP_T)
        )
        self._set_operation_attribute(
            _CUBLASLT_MATMUL_DESC_TRANSB, ctypes.c_int(_CUBLAS_OP_N)
        )
        self._set_operation_attribute(
            _CUBLASLT_MATMUL_DESC_EPILOGUE,
            ctypes.c_uint32(_CUBLASLT_EPILOGUE_BIAS),
        )
        self._set_operation_attribute(
            _CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, ctypes.c_int(_CUDA_R_16BF)
        )

        layouts = (
            (self._a_layout, K, N, K),
            (self._b_layout, K, M, K),
            (self._c_layout, N, M, N),
            (self._d_layout, N, M, N),
        )
        for layout, rows, columns, leading_dimension in layouts:
            _check(
                self._library.cublasLtMatrixLayoutCreate(
                    ctypes.byref(layout),
                    _CUDA_R_16BF,
                    rows,
                    columns,
                    leading_dimension,
                ),
                "create matrix layout",
            )

        _check(
            self._library.cublasLtMatmulPreferenceCreate(
                ctypes.byref(self._preference)
            ),
            "create matmul preference",
        )
        workspace_limit = ctypes.c_uint64(WORKSPACE_CAP_BYTES)
        _check(
            self._library.cublasLtMatmulPreferenceSetAttribute(
                self._preference,
                _CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                ctypes.byref(workspace_limit),
                ctypes.sizeof(workspace_limit),
            ),
            "set workspace preference",
        )

    @staticmethod
    def _check_tensor(
        name: str,
        tensor: torch.Tensor,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> None:
        if tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if tensor.dtype != torch.bfloat16 or tensor.device != device:
            raise ValueError(f"{name} must be BF16 on {device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.data_ptr() % 256 != 0:
            raise ValueError(f"{name} must be 256-byte aligned")

    def _select_algorithm(self, bias: torch.Tensor, device_index: int) -> None:
        if self._heuristic is not None:
            if self._device_index != device_index:
                raise RuntimeError("cached cuBLASLt wrapper cannot cross devices")
            return
        self._set_operation_attribute(
            _CUBLASLT_MATMUL_DESC_BIAS_POINTER,
            ctypes.c_void_p(bias.data_ptr()),
        )
        result = _HeuristicResult()
        returned = ctypes.c_int()
        _check(
            self._library.cublasLtMatmulAlgoGetHeuristic(
                self._handle,
                self._operation,
                self._a_layout,
                self._b_layout,
                self._c_layout,
                self._d_layout,
                self._preference,
                1,
                ctypes.byref(result),
                ctypes.byref(returned),
            ),
            "get matmul heuristic",
        )
        if returned.value != 1:
            raise RuntimeError("cuBLASLt returned no case-8 linear algorithm")
        _check(result.state, "cuBLASLt heuristic result")
        if result.workspace_size > WORKSPACE_CAP_BYTES:
            raise RuntimeError("cuBLASLt selected excessive workspace")
        self._heuristic = result
        self._device_index = device_index
        if result.workspace_size:
            self._workspace = torch.empty(
                result.workspace_size,
                device=bias.device,
                dtype=torch.uint8,
            )

    def __call__(
        self,
        activation: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("cuBLASLt wrapper is closed")
        device = activation.device
        if device.type != "cuda":
            raise ValueError("cuBLASLt linear requires CUDA")
        self._check_tensor("activation", activation, (64, 128, K), device)
        self._check_tensor("weight", weight, (N, K), device)
        self._check_tensor("bias", bias, (N,), device)

        with torch.cuda.device(device):
            self._select_algorithm(bias, torch.cuda.current_device())
            self._set_operation_attribute(
                _CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                ctypes.c_void_p(bias.data_ptr()),
            )
            output = torch.empty_like(activation)
            alpha = ctypes.c_float(1.0)
            beta = ctypes.c_float(0.0)
            stream = torch.cuda.current_stream(device).cuda_stream
            assert self._heuristic is not None
            workspace_pointer = (
                ctypes.c_void_p(self._workspace.data_ptr())
                if self._workspace is not None
                else ctypes.c_void_p()
            )
            _check(
                self._library.cublasLtMatmul(
                    self._handle,
                    self._operation,
                    ctypes.byref(alpha),
                    ctypes.c_void_p(weight.data_ptr()),
                    self._a_layout,
                    ctypes.c_void_p(activation.data_ptr()),
                    self._b_layout,
                    ctypes.byref(beta),
                    ctypes.c_void_p(output.data_ptr()),
                    self._c_layout,
                    ctypes.c_void_p(output.data_ptr()),
                    self._d_layout,
                    ctypes.byref(self._heuristic.algo),
                    workspace_pointer,
                    self._heuristic.workspace_size,
                    ctypes.c_void_p(stream),
                ),
                "cublasLtMatmul",
            )
        return output

    def close(self) -> None:
        if self._closed:
            return
        for descriptor, destroy in (
            (self._preference, self._library.cublasLtMatmulPreferenceDestroy),
            (self._a_layout, self._library.cublasLtMatrixLayoutDestroy),
            (self._b_layout, self._library.cublasLtMatrixLayoutDestroy),
            (self._c_layout, self._library.cublasLtMatrixLayoutDestroy),
            (self._d_layout, self._library.cublasLtMatrixLayoutDestroy),
            (self._operation, self._library.cublasLtMatmulDescDestroy),
            (self._handle, self._library.cublasLtDestroy),
        ):
            if descriptor.value:
                destroy(descriptor)
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
