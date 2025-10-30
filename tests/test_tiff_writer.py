from napari_tiff.napari_tiff_writter import images_layer_writer
from napari_tiff.napari_tiff_reader import reader_function

import pytest
import numpy as np
import numpy.testing as npt
import tifffile
from napari.layers import Image
from pathlib import Path


def test_simple_write(tmp_path: Path) -> None:
    layer = Image(np.random.randint(0, 255, size=(20, 20)).astype(np.uint8))
    writer = images_layer_writer(str(tmp_path / "data.tiff"), [layer.as_layer_data_tuple()])
    assert writer == [str(tmp_path / "data.tiff")]
    assert (tmp_path / "data.tiff").exists()
    npt.assert_equal(tifffile.imread(tmp_path / "data.tiff"), layer.data)

def test_simple_write3d(tmp_path: Path) -> None:
    layer = Image(np.random.randint(0, 255, size=(10, 20, 20)).astype(np.uint8))
    writer = images_layer_writer(str(tmp_path / "data.tiff"), [layer.as_layer_data_tuple()])
    assert writer == [str(tmp_path / "data.tiff")]
    assert (tmp_path / "data.tiff").exists()
    npt.assert_equal(tifffile.imread(tmp_path / "data.tiff"), layer.data)

def test_simple_write_two_images(tmp_path: Path) -> None:
    layer1 = Image(np.random.randint(0, 255, size=(20, 20)).astype(np.uint8))
    layer2 = Image(np.random.randint(0, 255, size=(20, 20)).astype(np.uint8))
    writer = images_layer_writer(str(tmp_path / "data.tiff"), [layer1.as_layer_data_tuple(), layer2.as_layer_data_tuple()])
    assert writer == [str(tmp_path / "data.tiff")]
    assert (tmp_path / "data.tiff").exists()
    save_data = tifffile.imread(tmp_path / "data.tiff")
    assert save_data.shape == (2, 20, 20)
    npt.assert_equal(save_data[0], layer1.data)
    npt.assert_equal(save_data[1], layer2.data)


def test_reading_saved_default_meta(tmp_path: Path) -> None:
    layer = Image(np.empty((20, 20), dtype=np.uint8), scale=(10,10))
    images_layer_writer(str(tmp_path / "data.tiff"), [layer.as_layer_data_tuple()])
    layer_data_list = reader_function(str(tmp_path / "data.tiff"))
    assert layer_data_list[0][1]["scale"] ==[10, 10]
    assert layer_data_list[0][1]["units"] == ['pixel', 'pixel']


def test_reading_saved_meta(tmp_path: Path) -> None:
    layer = Image(np.empty((20, 20), dtype=np.uint8), scale=(10,10), units=['mm', 'mm'])
    images_layer_writer(str(tmp_path / "data.tiff"), [layer.as_layer_data_tuple()])
    layer_data_list = reader_function(str(tmp_path / "data.tiff"))
    assert layer_data_list[0][1]["scale"] ==[10, 10]
    assert layer_data_list[0][1]["units"] == ['mm', 'mm']
    assert layer_data_list[0][1]["axis_labels"] == ["y", "x"]

def test_reading_saved_meta_time(tmp_path: Path) -> None:
    layer = Image(np.empty((20, 20, 20), dtype=np.uint8), scale=(3, 10,10), units=['s', 'mm', 'mm'], axis_labels=("t", "y", "x"))
    images_layer_writer(str(tmp_path / "data.tiff"), [layer.as_layer_data_tuple()])
    layer_data_list = reader_function(str(tmp_path / "data.tiff"))
    assert layer_data_list[0][1]["scale"] ==[3, 10, 10]
    assert layer_data_list[0][1]["units"] == ['s', 'mm', 'mm']
    assert layer_data_list[0][1]["axis_labels"] == ["t", "y", "x"]

def test_determine_axis_time(tmp_path: Path) -> None:
    layer = Image(np.empty((20, 20, 20), dtype=np.uint8), scale=(3, 10, 10), units=['s', 'mm', 'mm'])
    writer = images_layer_writer(str(tmp_path / "data.tiff"), [layer.as_layer_data_tuple()])
    assert writer == [str(tmp_path / "data.tiff")]
    assert (tmp_path / "data.tiff").exists()
    layer_data_list = reader_function(str(tmp_path / "data.tiff"))
    assert layer_data_list[0][1]["axis_labels"] == ["t", "y", "x"]
    assert layer_data_list[0][1]["scale"] ==[3, 10, 10]
    assert layer_data_list[0][1]["units"] == ['s', 'mm', 'mm']


def test_determine_axis_time_3d(tmp_path: Path) -> None:
    layer = Image(np.empty((5, 20, 20, 20), dtype=np.uint8), scale=(3, 5, 10, 10), units=['s', "cm", 'mm', 'mm'])
    writer = images_layer_writer(str(tmp_path / "data.tiff"), [layer.as_layer_data_tuple()])
    assert writer == [str(tmp_path / "data.tiff")]
    assert (tmp_path / "data.tiff").exists()
    layer_data_list = reader_function(str(tmp_path / "data.tiff"))
    assert layer_data_list[0][1]["axis_labels"] == ["t", "z", "y", "x"]
    assert layer_data_list[0][1]["scale"] ==[3, 5, 10, 10]
    assert layer_data_list[0][1]["units"] == ['s', 'cm', 'mm', 'mm']

def test_not_match_data_shape(tmp_path: Path) -> None:
    layer1 = Image(np.random.randint(0, 255, size=(20, 20)).astype(np.uint8))
    layer2 = Image(np.random.randint(0, 255, size=(2, 20, 20)).astype(np.uint8))
    with pytest.raises(ValueError, match="All images must have the same shape"):
        images_layer_writer(str(tmp_path / "data.tiff"), [layer1.as_layer_data_tuple(), layer2.as_layer_data_tuple()])

def test_not_match_data_dtype(tmp_path: Path) -> None:
    layer1 = Image(np.random.randint(0, 255, size=(20, 20)).astype(np.uint8))
    layer2 = Image(np.random.randint(0, 255, size=(20, 20)).astype(np.uint16))
    with pytest.raises(ValueError, match="All images must have the same dtype"):
        images_layer_writer(str(tmp_path / "data.tiff"), [layer1.as_layer_data_tuple(), layer2.as_layer_data_tuple()])

def test_not_match_layer_scale(tmp_path: Path) -> None:
    layer1 = Image(np.random.randint(0, 255, size=(20, 20)).astype(np.uint8))
    layer2 = Image(np.random.randint(0, 255, size=(20, 20)).astype(np.uint8), scale=(10,10))
    with pytest.raises(ValueError, match="All images must have the same scale"):
        images_layer_writer(str(tmp_path / "data.tiff"), [layer1.as_layer_data_tuple(), layer2.as_layer_data_tuple()])