import numpy
import pytest
from tifffile import imwrite, TiffFile

from napari_tiff.napari_tiff_reader import imagej_reader


@pytest.fixture(scope="session")
def imagej_hyperstack_image(tmp_path_factory):
    """ImageJ hyperstack tiff image.

    Write a 10 fps time series of volumes with xyz voxel size 2.6755x2.6755x3.9474
    micron^3 to an ImageJ hyperstack formatted TIFF file:
    """
    filename = tmp_path_factory.mktemp("data") / "imagej_hyperstack.tif"

    volume = numpy.random.randn(6, 57, 256, 256).astype('float32')
    image_labels = [f'{i}' for i in range(volume.shape[0] * volume.shape[1])]
    metadata = {
            'spacing': 3.947368,
            'unit': 'um',
            'finterval': 1/10,
            'fps': 10.0,
            'axes': 'TZYX',
            'Labels': image_labels,
        }
    imwrite(
        filename,
        volume,
        imagej=True,
        resolution=(1./2.6755, 1./2.6755),
        metadata=metadata,
    )
    return (filename, metadata)


def test_imagej_hyperstack_metadata(imagej_hyperstack_image):
    """Test metadata from imagej hyperstack tiff is passed to napari layer."""
    imagej_hyperstack_filename, expected_metadata = imagej_hyperstack_image

    with TiffFile(imagej_hyperstack_filename) as tif:
        layer_data_list = imagej_reader(tif)

    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 3

    napari_layer_metadata = layer_data_tuple[1]
    assert napari_layer_metadata.get('scale') == (1.0, 3.947368, 2.675500000484335, 2.675500000484335)
    assert layer_data_tuple[0].shape == (6, 57, 256, 256)  # image volume shape

    napari_layer_imagej_metadata = napari_layer_metadata.get('metadata').get('imagej_metadata')
    assert napari_layer_imagej_metadata.get('slices') == 57  # calculated automatically when file is written
    assert napari_layer_imagej_metadata.get('frames') == 6   # calculated automatically when file is written
    expected_metadata.pop('axes')  # 'axes' is stored as a tiff series attribute, not in the imagej_metadata property
    for (key, val) in expected_metadata.items():
        assert key in napari_layer_imagej_metadata
        assert napari_layer_imagej_metadata.get(key) == val
