import numpy as np
import pytest
from tifffile import xml2dict

from napari_tiff.napari_tiff_reader import tifffile_reader
from napari_tiff.napari_tiff_metadata import get_extra_metadata
from napari_tiff._tests.test_data import example_data_imagej, example_data_ometiff, example_data_multiresolution


@pytest.mark.parametrize("data_fixture, original_data, metadata_type", [
    (example_data_ometiff, np.random.randint(0, 255, size=(20, 20)).astype(np.uint8), 'ome_metadata'),
    (example_data_imagej, np.random.randint(0, 255, size=(20, 20)).astype(np.uint8), 'imagej_metadata'),
    ])
def test_metadata_dict(tmp_path, data_fixture, original_data, metadata_type):
    """Check the 'metadata' dict stored with the layer data contains expected values."""
    test_data = data_fixture(tmp_path, original_data)
    result_metadata = tifffile_reader(test_data)[0][1]
    # check metadata against TiffFile source metadata
    expected_metadata = getattr(test_data, metadata_type)
    if isinstance(expected_metadata, str):
        expected_metadata = xml2dict(expected_metadata)
    assert result_metadata.get('metadata').get(metadata_type) == expected_metadata
    # check metadata in layer is identical to the extra metadata dictionary result
    extra_metadata_dict = get_extra_metadata(test_data)
    assert result_metadata.get('metadata') == extra_metadata_dict
