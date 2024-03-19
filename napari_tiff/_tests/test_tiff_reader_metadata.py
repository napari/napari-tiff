from glob import glob
from napari.layers import Layer, Image
import numpy as np
import pytest
import tifffile

from napari_tiff import napari_get_reader


viewer = None


def generate_ometiff_file(tmp_path, filename, data):
    filepath = str(tmp_path / filename)
    tifffile.imwrite(filepath, data, ome=True)
    return filepath


@pytest.mark.parametrize("data_fixture, original_filename, original_data", [
    (generate_ometiff_file, "single_channel.ome.tif", np.random.randint(0, 255, size=(16, 16)).astype(np.uint8)),
    (generate_ometiff_file, "multi_channel.ome.tif", np.random.randint(0, 255, size=(2, 16, 16)).astype(np.uint8)),
    (generate_ometiff_file, "rgb.ome.tif", np.random.randint(0, 255, size=(16, 16, 3)).astype(np.uint8)),
    (None, "D:/slides/test/*", None),
    ])
def test_reader(tmp_path, data_fixture, original_filename, original_data):
    global viewer

    if data_fixture is not None:
        test_files = [data_fixture(tmp_path, original_filename, original_data)]
    else:
        test_files = glob(original_filename)

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
