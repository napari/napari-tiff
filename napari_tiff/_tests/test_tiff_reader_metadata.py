import napari
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
    channel_axis = metadata.pop('channel_axis', None)
    if isinstance(metadata.get('blending'), (list, tuple)):
        pass
        # unravel layered data
        #for channeli, blending in enumerate(metadata.get('blending')):
        #    metadata1 = get_list_dict(metadata, channeli)
        #    layer = Image(data, **metadata1)
        #    assert isinstance(layer, Image)
    else:
        layer = Image(data, **metadata)
        assert isinstance(layer, Image)


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# you don't need to import it, as long as napari is installed
# in your testing environment
def test_reader_metadata():
    path = "D:/slides/EM04573_01small.ome.tif"
    viewer = napari.Viewer()
    layer_datas = napari_get_reader(path)(path)
    for layer_data in layer_datas:
        data = layer_data[0]
        metadata = layer_data[1]
        layer = viewer.add_image(data, **metadata)
        assert layer is not None


def get_list_dict(dct, index):
    dct1 = {}
    for key, value in dct.items():
        if isinstance(value, (list, tuple)):
            if index < len(value):
                dct1[key] = value[index]
    return dct1
