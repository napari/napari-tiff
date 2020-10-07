import os
import zipfile

import numpy as np
from napari_tifffile_reader import napari_get_reader
from napari_tifffile_reader.napari_tifffile_reader import (imagecodecs_reader,
                                             imagej_reader,
                                             tifffile_reader,
                                             zip_reader)
import pytest
import tifffile


def example_data_filepath(tmp_path, original_data):
    example_data_filepath = str(tmp_path / "myfile.tif")
    tifffile.imwrite(example_data_filepath, original_data)
    return example_data_filepath


def example_data_tiff(tmp_path, original_data):
    example_data_filepath = str(tmp_path / "myfile.tif")
    tifffile.imwrite(example_data_filepath, original_data, imagej=True)
    return tifffile.TiffFile(example_data_filepath)


def example_data_zipped(tmp_path, original_data):
    example_tiff_filepath = str(tmp_path / "myfile.tif")
    tifffile.imwrite(example_tiff_filepath, original_data)
    example_zipped_filepath = str(tmp_path / "myfile.zip")
    with zipfile.ZipFile(example_zipped_filepath, 'w') as myzip:
        myzip.write(example_tiff_filepath)
    os.remove(example_tiff_filepath)  # not needed now the zip file is saved
    return example_zipped_filepath


def test_get_reader_pass():
    """Test None is returned if file format is not recognized."""
    reader = napari_get_reader("fake.file")
    assert reader is None


@pytest.mark.parametrize("data_fixture, original_data", [
    (example_data_filepath, np.random.random((20, 20))),
    (example_data_zipped, np.random.random((20, 20))),
    ])
def test_reader(tmp_path, data_fixture, original_data):
    """An example of how you might test your plugin."""

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
    if data_fixture == example_data_zipped:  # zipfile has unsqueezed dimension
        np.testing.assert_allclose(original_data, layer_data_tuple[0][0])
    else:
        np.testing.assert_allclose(original_data, layer_data_tuple[0])


@pytest.mark.parametrize("reader, data_fixture, original_data", [
    (imagecodecs_reader, example_data_filepath, np.random.random((20, 20))),
    (imagej_reader, example_data_tiff,  np.random.randint(0, 255, size=(20, 20)).astype(np.uint8)),
    (tifffile_reader, example_data_tiff, np.random.randint(0, 255, size=(20, 20)).astype(np.uint8)),
    (zip_reader, example_data_zipped, np.random.random((20, 20))),
    ])
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
    if data_fixture == example_data_zipped:  # zipfile has unsqueezed dimension
        np.testing.assert_allclose(original_data, layer_data_tuple[0][0])
    else:
        np.testing.assert_allclose(original_data, layer_data_tuple[0])
