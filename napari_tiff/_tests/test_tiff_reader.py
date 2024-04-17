import numpy as np
import pytest
import zarr

from napari_tiff import napari_get_reader
from napari_tiff._tests.test_data import (
    example_data_filepath,
    example_data_imagej,
    example_data_multiresolution,
    example_data_ometiff,
    example_data_tiff,
    example_data_zipped_filepath,
)
from napari_tiff.napari_tiff_reader import (
    imagecodecs_reader,
    tifffile_reader,
    zip_reader,
)


def test_get_reader_pass():
    """Test None is returned if file format is not recognized."""
    reader = napari_get_reader("fake.file")
    assert reader is None


@pytest.mark.parametrize(
    "data_fixture, original_data",
    [
        (example_data_filepath, np.random.random((20, 20))),
        (example_data_zipped_filepath, np.random.random((20, 20))),
    ],
)
def test_reader(tmp_path, data_fixture, original_data):
    """Test tiff reader with example data filepaths."""

    my_test_file = data_fixture(tmp_path, original_data)

    # try to read it back in
    reader = napari_get_reader(my_test_file)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(my_test_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0

    # make sure it's the same as it started
    if data_fixture == example_data_zipped_filepath:  # zipfile has unsqueezed dimension
        np.testing.assert_allclose(original_data, layer_data_tuple[0][0])
    else:
        np.testing.assert_allclose(original_data, layer_data_tuple[0])


@pytest.mark.parametrize(
    "reader, data_fixture, original_data",
    [
        (imagecodecs_reader, example_data_filepath, np.random.random((20, 20))),
        (
            tifffile_reader,
            example_data_imagej,
            np.random.randint(0, 255, size=(20, 20)).astype(np.uint8),
        ),
        (
            tifffile_reader,
            example_data_tiff,
            np.random.randint(0, 255, size=(20, 20)).astype(np.uint8),
        ),
        (
            tifffile_reader,
            example_data_ometiff,
            np.random.randint(0, 255, size=(20, 20)).astype(np.uint8),
        ),
        (zip_reader, example_data_zipped_filepath, np.random.random((20, 20))),
    ],
)
def test_all_readers(reader, data_fixture, original_data, tmp_path):
    """Test each individual reader."""
    assert callable(reader)

    test_data = data_fixture(tmp_path, original_data)

    # make sure we're delivering the right format
    layer_data_list = reader(test_data)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0

    # make sure it's the same as it started
    if data_fixture == example_data_zipped_filepath:  # zipfile has unsqueezed dimension
        np.testing.assert_allclose(original_data, layer_data_tuple[0][0])
    else:
        np.testing.assert_allclose(original_data, layer_data_tuple[0])


def test_multiresolution_image(example_data_multiresolution):
    """Test opening a multi-resolution image."""
    assert example_data_multiresolution.series[0].is_pyramidal
    layer_data_list = tifffile_reader(example_data_multiresolution)
    layer_data_tuple = layer_data_list[0]
    layer_data = layer_data_tuple[0]
    assert len(layer_data) == 3
    assert layer_data[0].shape == (16, 512, 512, 3)
    assert layer_data[1].shape == (16, 256, 256, 3)
    assert layer_data[2].shape == (16, 128, 128, 3)
    assert all([isinstance(level, zarr.Array) for level in layer_data])
