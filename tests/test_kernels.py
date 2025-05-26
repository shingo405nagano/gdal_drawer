import numpy as np
import pytest

from gdal_drawer.kernels import Kernels, KernelSize
from gdal_drawer.utils.colors import dimensional_count

kernels = Kernels()


@pytest.mark.parametrize(
    "kernel_size, expected",
    [
        (3, 3),
        (4, 5),
        (5, 5),
        (6, 7),
        (7, 7),
        (8, 9),
        (9, 9),
        (10, 11),
    ]
)
def test__adjust_size_from_kernels_cls(kernel_size, expected):
    """Test the _adjust_size method from Kernels class."""
    class Test(Kernels):
        def __init__(self):
            super().__init__()
        @kernels._adjust_size(0, 'size')
        def dummy_function(self, size):
            return size

    test = Test()
    assert test.dummy_function(kernel_size) == expected
    assert test.dummy_function(size=kernel_size) == expected
    

@pytest.mark.parametrize(
    "distance, x_cell_size, expected",
    [
        (5, 1, KernelSize(5, 5)),
        (5, 0.5, KernelSize(10, 10)),
        (10, 2, KernelSize(5, 5)),
        (5, 0.4, KernelSize(13, 13)),
    ]
)
def test_distance_to_kernel_size_from_kernels_cls(distance, x_cell_size, expected):
    """Test the distance_to_kernel_size function."""
    kernel_size = kernels.distance_to_kernel_size(distance, x_cell_size)
    assert isinstance(kernel_size, KernelSize)
    assert kernel_size.x == expected.x
    assert kernel_size.y == expected.y


@pytest.mark.parametrize(
    "x_size",
    [i for i in range(3, 20, 1)]
)
def test_mean_kernel_from_kernels_cls(x_size):
    """Test the mean_kernel function."""
    kernel = kernels.mean_kernel(x_size)
    assert isinstance(kernel, np.ndarray)
    assert dimensional_count(kernel) == 2
    try:
        assert kernel.shape == (x_size, x_size)
    except:
        assert kernel.shape == (x_size + 1, x_size + 1)
    assert kernel.sum() == pytest.approx(1.0, rel=1e-2)


@pytest.mark.parametrize(
    "x_size",
    [i for i in range(3, 20, 1)]
)   
def test_doughnut_kernel_from_kernels_cls(x_size):
    """Test the doughnut_kernel function."""
    kernel = kernels.doughnut_kernel(x_size)
    assert isinstance(kernel, np.ndarray)
    assert dimensional_count(kernel) == 2
    outside = kernel[: 1].sum() + kernel[-1:].sum() + kernel[1: -1, : 1].sum() * 2
    assert outside == pytest.approx(1.0, rel=1e-2)
    assert kernel.sum() == pytest.approx(1.0, rel=1e-2)


@pytest.mark.parametrize(
    "sigma, coef",
    [
        (1, None),
        (1, 1),
        (2, 1),
        (2, 2)
    ]
)
def test_gaussian_kernel_from_kernels_cls(sigma, coef):
    """Test the gaussian_kernel function."""
    kernel = kernels.gaussian_kernel(sigma, coef)
    assert isinstance(kernel, np.ndarray)
    assert dimensional_count(kernel) == 2
    assert kernel.sum() == pytest.approx(1.0, rel=1e-2)
    rows, cols = kernel.shape
    center_val = kernel[rows // 2, cols // 2]
    outside_val = kernel[0, 0]
    assert outside_val < center_val


@pytest.mark.parametrize(
    "sigma, coef",
    [
        (1, None),
        (1, 1),
        (2, 1),
        (2, 2)
    ]
)
def test_inverse_gaussian_kernel_from_kernels_cls(sigma, coef):
    """Test the inverse_gaussian_kernel function."""
    kernel = kernels.inverse_gaussian_kernel(sigma, coef)
    assert isinstance(kernel, np.ndarray)
    assert dimensional_count(kernel) == 2
    assert kernel.sum() == pytest.approx(1.0, rel=1e-2)
    rows, cols = kernel.shape
    center_val = kernel[rows // 2, cols // 2]
    outside_val = kernel[0, 0]
    assert center_val < outside_val
    

@pytest.mark.parametrize(
    "x_size, coef",
    [
        (5, None),
        (5, 1),
        (10, 1)
    ]
)
def test_gaussian_kernel_from_size_from_kernels_cls(x_size, coef):
    """Test the gaussian_kernel_from_size function."""
    kernel = kernels.gaussian_kernel_from_size(x_size, coef=coef)
    assert isinstance(kernel, np.ndarray)
    assert dimensional_count(kernel) == 2
    assert kernel.sum() == pytest.approx(1.0, rel=1e-2)
    rows, cols = kernel.shape
    center_val = kernel[rows // 2, cols // 2]
    outside_val = kernel[0, 0]
    assert outside_val < center_val
    

@pytest.mark.parametrize(
    "x_size, coef",
    [
        (5, None),
        (5, 1),
        (10, 1)
    ]
)
def test_inverse_gaussian_kernel_from_size_from_kernels_cls(x_size, coef):
    """Test the inverse_gaussian_kernel_from_size function."""
    kernel = kernels.inverse_gaussian_kernel_from_size(x_size, coef=coef)
    assert isinstance(kernel, np.ndarray)
    assert dimensional_count(kernel) == 2
    assert kernel.sum() == pytest.approx(1.0, rel=1e-2)
    rows, cols = kernel.shape
    center_val = kernel[rows // 2, cols // 2]
    outside_val = kernel[0, 0]
    assert center_val < outside_val