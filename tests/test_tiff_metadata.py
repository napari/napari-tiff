import numpy as np
import pytest
from tifffile import TiffFile, xml2dict

from base_data import (
    example_data_imagej,
    example_data_ometiff,
    imagej_hyperstack_image,
)
from napari_tiff.napari_tiff_metadata import get_extra_metadata, get_scale_translate_and_units_from_ome
from napari_tiff.napari_tiff_reader import tifffile_reader


@pytest.mark.parametrize(
    "data_fixture, original_data, metadata_type",
    [
        (
            example_data_ometiff,
            np.random.randint(0, 255, size=(20, 20)).astype(np.uint8),
            "ome_metadata",
        ),
        (
            example_data_imagej,
            np.random.randint(0, 255, size=(20, 20)).astype(np.uint8),
            "imagej_metadata",
        ),
    ],
)
def test_metadata_dict(tmp_path, data_fixture, original_data, metadata_type):
    """Check the 'metadata' dict stored with the layer data contains expected values."""
    test_data = data_fixture(tmp_path, original_data)
    result_metadata = tifffile_reader(test_data)[0][1]
    # check metadata against TiffFile source metadata
    expected_metadata = getattr(test_data, metadata_type)
    if isinstance(expected_metadata, str):
        expected_metadata = xml2dict(expected_metadata)
    assert result_metadata.get("metadata").get(metadata_type) == expected_metadata
    # check metadata in layer is identical to the extra metadata dictionary result
    extra_metadata_dict = get_extra_metadata(test_data)
    assert result_metadata.get("metadata") == extra_metadata_dict


def test_imagej_hyperstack_metadata(imagej_hyperstack_image):
    """Test metadata from imagej hyperstack tiff is passed to napari layer."""
    imagej_hyperstack_filename, expected_metadata = imagej_hyperstack_image

    with TiffFile(imagej_hyperstack_filename) as tif:
        layer_data_list = tifffile_reader(tif)

    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 3

    napari_layer_metadata = layer_data_tuple[1]
    assert napari_layer_metadata.get("scale") == (
        0.1,
        3.947368,
        2.675500000484335,
        2.675500000484335,
    )
    assert layer_data_tuple[0].shape == (6, 57, 256, 256)  # image volume shape

    napari_layer_imagej_metadata = napari_layer_metadata.get("metadata").get(
        "imagej_metadata"
    )
    assert (
        napari_layer_imagej_metadata.get("slices") == 57
    )  # calculated automatically when file is written
    assert (
        napari_layer_imagej_metadata.get("frames") == 6
    )  # calculated automatically when file is written
    expected_metadata.pop(
        "axes"
    )  # 'axes' is stored as a tiff series attribute, not in the imagej_metadata property
    for key, val in expected_metadata.items():
        assert key in napari_layer_imagej_metadata
        assert napari_layer_imagej_metadata.get(key) == val


@pytest.mark.parametrize(
    "data,expected",
    [
        (dict(pixels={"PhysicalSizeX": "0.1", "PhysicalSizeXUnit": "mm", "PhysicalSizeY": "0.2", "PhysicalSizeYUnit": "um"}, axes="xy", shape=(10, 10)), ([0.1, 0.2], [0, 0], ["mm", "um"])),
        (dict(pixels={"PhysicalSizeX": "0.1", "PhysicalSizeXUnit": "mm", "PhysicalSizeY": "0.2", "PhysicalSizeYUnit": "um"}, axes="zxy", shape=(1, 10, 10)), ([1, 0.1, 0.2], [0, 0, 0], ["pixel", "mm", "um"])),
        (dict(pixels={"PhysicalSizeX": "0.1", "PhysicalSizeXUnit": "mm", "PhysicalSizeY": "0.2", "PhysicalSizeYUnit": "um"}, axes="zyx", shape=(2, 10, 10)), ([1.0, 0.2, 0.1], [0, 0, 0], ['pixel', 'um', 'mm'])),
        (dict(pixels={"PhysicalSizeX": "0.1", "PhysicalSizeXUnit": "mm", "PhysicalSizeY": "0.2", "PhysicalSizeYUnit": "um"}, axes="txy", shape=(1, 10, 10)), ([1, 0.1, 0.2], [0, 0, 0], ["pixel", "mm", "um"])),
        (dict(pixels={"PhysicalSizeX": "0.1", "PhysicalSizeXUnit": "mm", "PhysicalSizeY": "0.2", "PhysicalSizeYUnit": "um"}, axes="tyx", shape=(2, 10, 10)), ([1.0, 0.2, 0.1], [0, 0, 0], ['pixel', 'um', 'mm'])),
        (dict(pixels={"PhysicalSizeX": "0.1", "PhysicalSizeXUnit": "mm", "PhysicalSizeY": "0.2", "PhysicalSizeYUnit": "um", "TimeIncrement": "10", "TimeIncrementUnit": "s"}, axes="txy", shape=(2, 10, 10)), ([10, 0.1, 0.2], [0, 0, 0], ["s", "mm", "um"])),
        (dict(pixels={"PhysicalSizeX": "0.1", "PhysicalSizeXUnit": "mm", "PhysicalSizeY": "0.2", "PhysicalSizeYUnit": "um", "TimeIncrement": "10"}, axes="txy", shape=(2, 10, 10)), ([10, 0.1, 0.2], [0, 0, 0], ["pixel", "mm", "um"])),
        (dict(pixels={"PhysicalSizeX": "0.1", "PhysicalSizeXUnit": "mm", "PhysicalSizeY": "0.2", "PhysicalSizeYUnit": "um", "PhysicalSizeZ": "10"}, axes="zxy", shape=(2, 10, 10)), ([10, 0.1, 0.2], [0, 0, 0], ["pixel", "mm", "um"])),
    ]
)
def test_get_scale_and_units_from_ome(data, expected):
    assert get_scale_translate_and_units_from_ome(**data) == expected