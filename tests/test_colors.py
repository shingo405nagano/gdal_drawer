from matplotlib import pyplot as plt
import pytest

import numpy as np
from matplotlib.colors import to_hex

from gdal_drawer.utils.colors import (
    CustomCmap,
    Converter,
    dimensional_count,
    LinearColorMap,
)


@pytest.mark.parametrize(
    "array, expected",
    [
        ('array', 0),
        ({'key': 'value', 'key2': 'value2'}, 0),
        ([0, 1, 2], 1),
        ([[0, 1], [2, 3]], 2),
        ([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], 3),
        (np.array([0, 1, 2]), 1),
        (np.array([[0, 1], [2, 3]]), 2),
        (np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]), 3),
    ]
)
def test_dimensional_count(array, expected):
    """Test the dimensional_count function."""
    assert dimensional_count(array) == expected


@pytest.mark.parametrize(
    "color_list, expected",
    [
        (['red', 'green', 'blue'], ['#ff0000', '#008000', '#0000ff']),
        ([(255, 0, 0), (0, 255, 0), (0, 0, 255)], ['#ff0000', '#00ff00', '#0000ff']),
        ([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)], ['#ff0000', '#00ff00', '#0000ff']),
        ([(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)], ['#ff0000ff', '#00ff00ff', '#0000ffff']),
    ],
)
def test__check_color_list_from_converter_cls(color_list, expected):
    converter = Converter(color_list)
    hex_color_list = converter._check_color_list(color_list)
    for hex_color, expected_color in zip(hex_color_list, expected):
        assert hex_color == expected_color
    with pytest.raises(TypeError):
        converter._check_color_list('invalid_color_list')
    with pytest.raises(ValueError):        
        converter._check_color_list([])
        converter._check_color_list(['red', 'green', '#blue'])
        converter._check_color_list([(255, 0, 0), (0, 255, 0), (0, 0, -255)])
        converter._check_color_list([(255, 0, 0, 0), (0, 255, 0, 0), (0, 0, -255, 0)])

@pytest.mark.parametrize(
    "color_list, expected",
    [
        (
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)], 
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        (
            [(255, 0, 0), (0, 255, 0), (0, 0, 255)],
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        ),
        (
            [(65535, 0, 0), (0, 65535, 0), (0, 0, 65535)],
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        )
    ],
)
def test__normalize_from_converter_cls(color_list, expected):
    converter = Converter(color_list)
    normalized_color_list = converter._normalize(color_list)
    for normal_color, expected_color in zip(normalized_color_list, expected):
        for nc, ec in zip(normal_color, expected_color):
            assert nc == ec
        

@pytest.mark.parametrize(
    "return_type, expected",
    [
        ('hex', 'hex'),
        ('rgba', 'rgba'),
        ('rgb', 'rgb'),
        ('HEX', 'hex'),
        ('RGBA', 'rgba'),
        ('RGB', 'rgb')
    ]
)
def test__check_return_type_from_converter_cls(return_type, expected):
    converter = Converter(['red', 'green', 'blue'], return_type=return_type)
    assert converter._check_return_type(return_type) == expected
    with pytest.raises(ValueError):
        converter._check_return_type('invalid_return_type')
    with pytest.raises(TypeError):
        converter._check_return_type(123)
        

@pytest.mark.parametrize(
    "rgba, expected",
    [
        (False, [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]),
        (True, [(1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)]),
    ]
)
def test_to_mpl_color_list_from_custom_cmap_cls(rgba, expected):
    color_list = ['red', 'blue']
    custom_cmap = CustomCmap()
    result = custom_cmap.to_mpl_color_list(color_list, rgba=rgba)
    for res, exp in zip(result, expected):
        for r, e in zip(res, exp):
            assert r == e
            

def test_color_list_to_linear_map_from_custom_cmap_cls():
    color_list = ['red', 'green', 'blue']
    custom_cmap = CustomCmap()
    linear_color_map = custom_cmap.color_list_to_linear_cmap(color_list)
    assert isinstance(linear_color_map, LinearColorMap)
    

@pytest.mark.parametrize(
    "position, return_type, expected",
    [
        (256, 'rgb', (1.0, 1.0, 1.0, 0.0)),
        (0, 'rgb', (1.0, 0.0, 0.0)),
        (0, 'rgba', (1.0, 0.0, 0.0, 1.0)),
        (0, 'hex', '#ff0000'),
        (255, 'hex', '#0000ff'),
        (0, 'int', (255, 0, 0)),
        (0, 'inta', (255, 0, 0, 255)),
    ]
)
def test__get_from_linear_color_map_cls(position, return_type, expected):
    color_list = ['red', 'green', 'blue']
    custom_cmap = CustomCmap()
    linear_color_map = custom_cmap.color_list_to_linear_cmap(color_list)
    color = linear_color_map._get(position, return_type)
    for c, e in zip(color, expected):
        assert c == e
    
    with pytest.raises(ValueError):
        linear_color_map._get(0, 'invalid')
        

@pytest.mark.parametrize(
    "position, return_type, expected",
    [
        (0, 'hex', '#ff0000'),
        ([0, 255], 'hex', ['#ff0000', '#0000ff']),
        ([[0, 255]], 'hex', [['#ff0000', '#0000ff']])
    ]
)
def test_get_from_linear_color_map_cls(position, return_type, expected):
    color_list = ['red', 'green', 'blue']
    custom_cmap = CustomCmap()
    linear_color_map = custom_cmap.color_list_to_linear_cmap(color_list)
    result = linear_color_map.get(position, return_type)
    if isinstance(result, str):
        assert result == expected
    result = np.array(result)
    expected = np.array(expected)
    assert np.array_equal(result, expected)
    with pytest.raises(TypeError):
        linear_color_map.get([0, 255.0], 'hex')
        

def test_get_registered_color_from_linear_color_map_cls():
    color_list = ['red', 'green', 'blue']
    custom_cmap = CustomCmap()
    linear_color_map = custom_cmap.color_list_to_linear_cmap(color_list)
    # Test for registered color of RGB
    result = linear_color_map.get_registered_color('rgb')
    assert 2 == dimensional_count(result)
    assert 0.0 <= np.array(result).max() <= 1.0
    assert 0.0 <= np.array(result).min() <= 1.0
    # The for registered color of hex
    result = linear_color_map.get_registered_color('hex')
    assert 1 == dimensional_count(result)
    _ = [to_hex(color) for color in result]
    # The for registered color of int
    result = linear_color_map.get_registered_color('int')
    assert 2 == dimensional_count(result)
    assert 0 <= np.array(result).max() <= 255
    assert 0 <= np.array(result).min() <= 255
    
    
def test_generate_idx_for_retriveval_from_linear_color_map_cls():
    color_map = plt.get_cmap('viridis')
    linear_color_map = LinearColorMap(color_map)
    values = np.random.normal(0, 1, 100).reshape(10, 10)
    indices = linear_color_map.generate_idx_for_retrieval(values)
    assert isinstance(indices, np.ndarray)
    assert 0 <= indices.min() <= 255
    assert 0 <= indices.max() <= 255
    values[4, 3] = np.nan
    indices = linear_color_map.generate_idx_for_retrieval(values)
    assert isinstance(indices, np.ndarray)
    assert -1 == indices.min()
    

def test_values_to_img_from_linear_color_map_cls():
    color_map = plt.get_cmap('viridis')
    linear_color_map = LinearColorMap(color_map)
    values = np.random.normal(0, 1, 100).reshape(10, 10)
    raster = linear_color_map.values_to_img(values)
    assert isinstance(raster, np.ndarray)
    assert dimensional_count(raster) == 3
    assert 0 <= raster.min() <= 255
    assert 0 <= raster.max() <= 255