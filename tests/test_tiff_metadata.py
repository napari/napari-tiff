import numpy as np
import pytest
from tifffile import imwrite, TiffFile, TiffWriter, xml2dict
from numpy import testing as npt

from base_data import (
    example_data_imagej,
    example_data_ometiff,
    imagej_hyperstack_image,
)
from napari_tiff.napari_tiff_metadata import (
    get_extra_metadata,
    get_scale_and_units_from_ome,
    get_scale_and_units_from_tiff,
)
from napari_tiff.napari_tiff_colormaps import qpi_color_to_rgba
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
    npt.assert_array_almost_equal(napari_layer_metadata.get("scale"), (
        0.1,
        3.947368,
        2.675500,
        2.675500,
    ))
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
        (dict(pixels={"PhysicalSizeX": "0.1", "PhysicalSizeXUnit": "mm", "PhysicalSizeY": "0.2", "PhysicalSizeYUnit": "um"}, axes="xy", shape=(10, 10)), ([100, 0.2], ["µm", "µm"])),
        (dict(pixels={"PhysicalSizeX": "0.1", "PhysicalSizeXUnit": "mm", "PhysicalSizeY": "0.2", "PhysicalSizeYUnit": "um"}, axes="zxy", shape=(1, 10, 10)), ([1, 100, 0.2], ["pixel", "µm", "µm"])),
        (dict(pixels={"PhysicalSizeX": "0.1", "PhysicalSizeXUnit": "mm", "PhysicalSizeY": "0.2", "PhysicalSizeYUnit": "um"}, axes="zyx", shape=(2, 10, 10)), ([1.0, 0.2, 100.0], ['pixel', 'µm', 'µm'])),
        (dict(pixels={"PhysicalSizeX": "0.1", "PhysicalSizeXUnit": "mm", "PhysicalSizeY": "0.2", "PhysicalSizeYUnit": "um"}, axes="txy", shape=(1, 10, 10)), ([1, 100, 0.2], ["pixel", "µm", "µm"])),
        (dict(pixels={"PhysicalSizeX": "0.1", "PhysicalSizeXUnit": "mm", "PhysicalSizeY": "0.2", "PhysicalSizeYUnit": "um"}, axes="tyx", shape=(2, 10, 10)), ([1.0, 0.2, 100.0], ['pixel', 'µm', 'µm'])),
        (dict(pixels={"PhysicalSizeX": "0.1", "PhysicalSizeXUnit": "mm", "PhysicalSizeY": "0.2", "PhysicalSizeYUnit": "um", "TimeIncrement": "10", "TimeIncrementUnit": "s"}, axes="txy", shape=(2, 10, 10)), ([10, 100, 0.2], ["s", "µm", "µm"])),
        (dict(pixels={"PhysicalSizeX": "0.1", "PhysicalSizeXUnit": "mm", "PhysicalSizeY": "0.2", "PhysicalSizeYUnit": "um", "TimeIncrement": "10"}, axes="txy", shape=(2, 10, 10)), ([10, 100, 0.2], ["pixel", "µm", "µm"])),
        (dict(pixels={"PhysicalSizeX": "0.1", "PhysicalSizeXUnit": "mm", "PhysicalSizeY": "0.2", "PhysicalSizeYUnit": "um", "PhysicalSizeZ": "10"}, axes="zxy", shape=(2, 10, 10)), ([10, 100, 0.2], ["pixel", "µm", "µm"])),
    ]
)
def test_get_scale_and_units_from_ome(data, expected):
    assert get_scale_and_units_from_ome(**data) == expected


@pytest.mark.parametrize(
    # XResolution = 100
    # YResolution = 200
    "resolution_unit, expected_units, expected_scale",
    [
        (1, ("pixel", "pixel"), (1 / 200, 1 / 100)),
        # ResolutionUnit 2 = inches
        (2, ("µm", "µm"), (25400 / 200, 25400 / 100)),
        # ResolutionUnit 3 = cm
        (3, ("µm", "µm"), (10000 / 200, 10000 / 100)),
    ],
)
def test_tifffile_reader_2d_resolution(
    tmp_path, resolution_unit, expected_units, expected_scale
):
    """Test tifffile_reader with 2D images with different resolution units."""
    data = np.zeros((10, 20), dtype=np.uint8)
    filepath = tmp_path / "test.tiff"

    imwrite(
        filepath,
        data,
        resolution=(100, 200),
        resolutionunit=resolution_unit,
    )

    with TiffFile(filepath) as tif:
        layer_data_list = tifffile_reader(tif)
        metadata = layer_data_list[0][1]
        scale = metadata.get("scale")
        units = metadata.get("units")

        assert np.allclose(scale, expected_scale)
        assert units == expected_units


def test_tifffile_reader_3d_resolution(tmp_path):
    """Test tifffile_reader with 3D image with resolution unit."""
    data = np.zeros((5, 10, 20), dtype=np.uint8)
    filepath = tmp_path / "test.tiff"

    imwrite(filepath, data, resolution=(100, 200), resolutionunit=2)

    with TiffFile(filepath) as tif:

        layer_data_list = tifffile_reader(tif)
        metadata = layer_data_list[0][1]
        scale = metadata.get("scale")
        units = metadata.get("units")

        expected_scale = (1.0, 25400 / 200, 25400 / 100)
        assert np.allclose(scale, expected_scale)
        assert units == ("pixel", "µm", "µm")


def test_svs_resolution_units(tmp_path):
    """Test tifffile_reader with SVS microns-per-pixel metadata."""
    data = np.zeros((10, 20), dtype=np.uint8)
    filepath = tmp_path / "test.svs"

    imwrite(filepath, data, description="Aperio |MPP = 1.234", tile=(16, 16))

    with TiffFile(filepath) as tif:
        assert tif.is_svs
        layer_data_list = tifffile_reader(tif)
        metadata = layer_data_list[0][1]
        scale = metadata.get("scale")
        units = metadata.get("units")

        expected_scale = (1.234, 1.234)
        assert np.allclose(scale, expected_scale)
        assert units == ("µm", "µm")


@pytest.mark.parametrize("nchannels", [2, 3, 39])
def test_tifffile_reader_splits_channels(tmp_path, nchannels):
    """Every channel becomes its own layer, however many there are.
    """
    data = np.zeros((nchannels, 10, 20), dtype=np.uint8)
    filepath = tmp_path / f"test_{nchannels}_channels.tiff"

    imwrite(
        filepath,
        data,
        photometric="minisblack",
        metadata={"axes": "CYX"},
        resolution=(100, 200),
        resolutionunit=2,
    )

    with TiffFile(filepath) as tif:
        assert tif.series[0].axes == "CYX"
        metadata = tifffile_reader(tif)[0][1]

    assert metadata.get("channel_axis") == 0
    assert metadata.get("name") == [f"Channel {i}" for i in range(nchannels)]
    # napari cycles its own colormaps when given None, and raises an IndexError
    # if it is handed fewer colormaps than there are channels
    assert metadata.get("colormap") is None
    # the channel axis is consumed by the split, so it is not a layer dimension
    assert np.allclose(metadata.get("scale"), (25400 / 200, 25400 / 100))
    assert metadata.get("units") == ("µm", "µm")


QPI_DESCRIPTION = (
    '<?xml version="1.0" encoding="utf-16"?>'
    "<PerkinElmer-QPI-ImageDescription>"
    "<Name>{name}</Name>"
    "<Color>{color}</Color>"
    "<Biomarker>{biomarker}</Biomarker>"
    "<ImageType>FullResolution</ImageType>"
    "</PerkinElmer-QPI-ImageDescription>"
)

QPI_CHANNELS = [
    # fluorophore, biomarker, colour
    ("DAPI", "DAPI", "0,0,255"),
    ("FITC", "CD8", "255,0,0"),
    ("Cy5", "CD4", "0,255,0"),
]


def write_qptiff(filepath, channels=QPI_CHANNELS, **kwargs):
    """Write a minimal PerkinElmer/Akoya QPTIFF, one page per channel."""
    # tifffile identifies a QPTIFF by its Software tag and reads the channels
    # back as a single CYX series; `metadata=None` keeps tifffile from writing
    # its own 'shaped' description, which would take priority over that
    with TiffWriter(filepath) as writer:
        for index, (name, biomarker, color) in enumerate(channels):
            writer.write(
                np.full((10, 20), index, dtype=np.uint8),
                photometric="minisblack",
                software="PerkinElmer-QPI",
                description=QPI_DESCRIPTION.format(
                    name=name, biomarker=biomarker, color=color
                ),
                metadata=None,
                contiguous=False,
                **kwargs,
            )


def test_qptiff_channel_names_and_colormaps(tmp_path):
    """QPTIFF channels are named and coloured from their own page metadata."""
    filepath = tmp_path / "test.qptiff"
    write_qptiff(filepath, resolution=(100, 200), resolutionunit=2)

    with TiffFile(filepath) as tif:
        assert tif.is_qpi
        assert tif.series[0].axes == "CYX"
        metadata = tifffile_reader(tif)[0][1]

    assert metadata.get("channel_axis") == 0
    # the biomarker is the stain, which is more useful than the fluorophore
    assert metadata.get("name") == ["DAPI", "CD8", "CD4"]
    assert metadata.get("colormap") == [
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 1.0),
    ]
    assert metadata.get("units") == ("µm", "µm")
    # each layer carries its own page's metadata, not page 0's
    assert [m["qpi_metadata"]["Biomarker"] for m in metadata["metadata"]] == [
        "DAPI",
        "CD8",
        "CD4",
    ]


def test_qptiff_falls_back_without_channel_metadata(tmp_path):
    """A QPTIFF missing per-channel names/colours keeps the generic metadata."""
    filepath = tmp_path / "test_bare.qptiff"
    write_qptiff(filepath, channels=[("", "", ""), ("", "", "")])

    with TiffFile(filepath) as tif:
        assert tif.is_qpi
        metadata = tifffile_reader(tif)[0][1]

    assert metadata.get("channel_axis") == 0
    assert metadata.get("name") == ["Channel 0", "Channel 1"]
    assert metadata.get("colormap") is None


def test_qptiff_single_channel(tmp_path):
    """A single channel QPTIFF has no channel axis to name or colour."""
    filepath = tmp_path / "test_one.qptiff"
    write_qptiff(filepath, channels=QPI_CHANNELS[:1])

    with TiffFile(filepath) as tif:
        assert tif.is_qpi
        metadata = tifffile_reader(tif)[0][1]

    assert metadata.get("channel_axis") is None
    assert metadata.get("name") is None


@pytest.mark.parametrize(
    "color, expected",
    [
        ("0,0,255", (0.0, 0.0, 1.0, 1.0)),
        ((255, 128, 0), (1.0, 128 / 255, 0.0, 1.0)),
        # unusable values fall back to letting napari pick the colormaps
        ("", None),
        ("0,0", None),
        ("r,g,b", None),
        (None, None),
    ],
)
def test_qpi_color_to_rgba(color, expected):
    assert qpi_color_to_rgba(color) == expected


@pytest.mark.parametrize(
    "description",
    [
        pytest.param(None, id="no description tag"),
        pytest.param("not xml at all", id="not xml"),
    ],
)
def test_qptiff_unreadable_page_description(tmp_path, description):
    """An unreadable page description falls back instead of raising.
    """
    filepath = tmp_path / "test_bad_description.qptiff"
    with TiffWriter(filepath) as writer:
        for index, page_description in enumerate([QPI_DESCRIPTION.format(
            name="DAPI", biomarker="DAPI", color="0,0,255"
        ), description]):
            writer.write(
                np.full((10, 20), index, dtype=np.uint8),
                photometric="minisblack",
                software="PerkinElmer-QPI",
                metadata=None,
                contiguous=False,
                **({} if page_description is None else {"description": page_description}),
            )

    with TiffFile(filepath) as tif:
        assert tif.is_qpi
        metadata = tifffile_reader(tif)[0][1]

    # both channels are still there; valid metadata from page 0 is preserved
    assert metadata.get("channel_axis") == 0
    assert metadata.get("name") == ["DAPI", "Channel 1"]
    assert metadata.get("colormap") is None


def test_qptiff_nested_biomarker_name(tmp_path):
    """A structured biomarker produces a useful channel name."""
    filepath = tmp_path / "test_biomarker.qptiff"
    write_qptiff(
        filepath,
        channels=[
            ("DAPI", "<Name>DAPI</Name>", "0,0,255"),
            ("FITC", "CD8", "255,0,0"),
        ],
    )

    with TiffFile(filepath) as tif:
        metadata = tifffile_reader(tif)[0][1]

    assert metadata.get("name") == ["DAPI", "CD8"]

