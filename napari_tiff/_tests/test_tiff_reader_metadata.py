from napari.layers import Image
import numpy as np
import pytest
import tifffile

from napari_tiff import napari_get_reader


def generate_ometiff_file(tmp_path, filename, data):
    filepath = str(tmp_path / filename)
    tifffile.imwrite(filepath, data, ome=True)
    return filepath


@pytest.mark.parametrize("data_fixture, original_filename, original_data", [
    (generate_ometiff_file, "myfile.ome.tif", np.random.randint(0, 255, size=(20, 20, 3)).astype(np.uint8)),
    (None, "D:/slides/EM04573_01small.ome.tif", None),
    ])
def test_reader(tmp_path, data_fixture, original_filename, original_data):

    if data_fixture is not None:
        test_file = data_fixture(tmp_path, original_filename, original_data)
    else:
        test_file = original_filename

    # try to read it back in
    reader = napari_get_reader(test_file)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_datas = reader(test_file)
    assert isinstance(layer_datas, list) and len(layer_datas) > 0
    layer_data = layer_datas[0]
    assert isinstance(layer_data, tuple) and len(layer_data) > 0

    # make sure it's the same as it started
    data = layer_data[0]
    if original_data is not None:
        np.testing.assert_allclose(original_data, data)

    # test layer metadata
    metadata = layer_data[1]
    layer = Image(data, **metadata)
    assert isinstance(layer, Image)
