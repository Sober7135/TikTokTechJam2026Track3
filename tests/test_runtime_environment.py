import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from transformer_benchmark.runtime_environment import (
    PYTORCH_CUDA_ALLOC_CONF,
    configure_pre_torch_environment,
)
from transformer_benchmark.runner import configure_triton_driver_path


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_configures_fragmentation_resistant_cuda_allocator(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            configure_pre_torch_environment()
            self.assertEqual(
                os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                PYTORCH_CUDA_ALLOC_CONF,
            )

    def test_preserves_explicit_cuda_allocator_configuration(self) -> None:
        with mock.patch.dict(
            os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64"}, clear=True
        ):
            configure_pre_torch_environment()
            self.assertEqual(
                os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                "max_split_size_mb:64",
            )

    def test_configures_existing_nixos_libcuda_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            driver_directory = Path(directory)
            driver_directory.joinpath("libcuda.so.1").touch()

            with mock.patch.dict(os.environ, {}, clear=True):
                configure_triton_driver_path(driver_directory)
                self.assertEqual(
                    os.environ.get("TRITON_LIBCUDA_PATH"), str(driver_directory)
                )

    def test_preserves_explicit_triton_libcuda_path(self) -> None:
        with mock.patch.dict(
            os.environ, {"TRITON_LIBCUDA_PATH": "/explicit/driver"}, clear=True
        ):
            configure_triton_driver_path(Path("/missing"))
            self.assertEqual(
                os.environ.get("TRITON_LIBCUDA_PATH"), "/explicit/driver"
            )


if __name__ == "__main__":
    unittest.main()
