from glob import glob
from napari.layers import Layer, Image
import numpy as np
import pytest
import tifffile

from napari_tiff import napari_get_reader


viewer = None


def generate_ometiff_file(tmp_path, filename, data, metadata):
    filepath = str(tmp_path / filename)
    tifffile.imwrite(filepath, data, ome=True, metadata=metadata)
    return filepath


def get_files(tmp_path, path, data, metadata):
    # TODO download files instead; path can be URL
    filepaths = glob(path)
    return filepaths


@pytest.mark.parametrize("data_fixture, original_filename, original_data, original_metadata", [
    (generate_ometiff_file, "single_channel.ome.tif", np.random.randint(0, 255, size=(16, 16)).astype(np.uint8), None),
    (generate_ometiff_file, "multi_channel.ome.tif", np.random.randint(0, 65535, size=(2, 16, 16)).astype(np.uint16),
     {'Channel': [{'Name': 'WF', 'Color': '-1'}, {'Name': 'Fluor', 'Color': '16711935'}]}),
    (generate_ometiff_file, "rgb.ome.tif", np.random.randint(0, 255, size=(16, 16, 3)).astype(np.uint8), None),
    (get_files, "D:/slides/test/*", None, None),
    ])
def test_reader(data_fixture, original_filename, original_data, original_metadata, tmp_path):
    global viewer

    test_files = data_fixture(tmp_path, original_filename, original_data, original_metadata)
    if not isinstance(test_files, list):
        test_files = [test_files]

    for test_file in test_files:
        # try to read it back in
        reader = napari_get_reader(test_file)
        assert callable(reader)

        # make sure we're delivering the right format
        layer_datas = reader(test_file)
        assert isinstance(layer_datas, list) and len(layer_datas) > 0

        for layer_data in layer_datas:
            assert isinstance(layer_data, tuple) and len(layer_data) > 0

            data = layer_data[0]
            metadata = layer_data[1]

            if original_data is not None:
                # make sure the data is the same as it started
                np.testing.assert_allclose(original_data, data)
            else:
                # test pixel data
                if isinstance(data, list):
                    data0 = data[0]
                else:
                    data0 = data
                assert data0.size > 0
                slicing = tuple([0] * data0.ndim)
                value = np.array(data0[slicing])
                assert value is not None and value.size > 0

            # test layer metadata
            if 'channel_axis' in metadata:
                if viewer is None:
                    from napari import Viewer
                    viewer = Viewer()
                layers = viewer.add_image(data, **metadata)
                if not isinstance(layers, list):
                    layers = [layers]
                for layer in layers:
                    assert isinstance(layer, Image)
            else:
                layer = Layer.create(*layer_data)  # incompatible with channel_axis
                assert isinstance(layer, Image)

                layer = Image(data, **metadata)    # incompatible with channel_axis
                assert isinstance(layer, Image)
