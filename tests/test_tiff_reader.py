import numpy as np
import os
import pytest
import uuid
import dask.array as da

import pint
import tifffile
from napari.layers import Image
from napari.components import ViewerModel
from napari.types import LayerDataTuple
from numpy.testing import assert_array_equal
from pint.testing import assert_allclose

from napari_tiff import napari_get_reader
from base_data import (
    example_data_filepath,
    example_data_imagej,
    example_data_multiresolution,
    example_data_ometiff,
    example_data_shaped_singleton,
    example_data_tiff,
    example_data_zipped_filepath,
)
from napari_tiff.napari_tiff_reader import (
    imagecodecs_reader,
    tifffile_reader,
    zip_reader,
    reader_function,
)


def assert_units_allclose(first, second, msg: str | None = None) -> None:
    for f, s in zip(first, second):
        pint.testing.assert_allclose(f, s, msg=msg)

def add_images_to_viewer(viewer: ViewerModel, layer_data_list: list[LayerDataTuple]) -> None:
    for layer_data in layer_data_list:
        viewer.add_image(layer_data[0], **layer_data[1])

def test_get_reader_pass():
    """Test None is returned if file format is not recognized."""
    reader = napari_get_reader("fake.file")
    assert reader is None


@pytest.mark.parametrize(
    "data_fixture, original_data",
    [
        (example_data_filepath, np.random.random((20, 20))),
        (example_data_zipped_filepath, np.random.random((20, 20))),
    ],
)
def test_reader(tmp_path, data_fixture, original_data):
    """Test tiff reader with example data filepaths."""

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
    if data_fixture == example_data_zipped_filepath:  # zipfile has unsqueezed dimension
        np.testing.assert_allclose(original_data, layer_data_tuple[0][0])
    else:
        np.testing.assert_allclose(original_data, layer_data_tuple[0])


@pytest.mark.parametrize(
    "reader, data_fixture, original_data",
    [
        (imagecodecs_reader, example_data_filepath, np.random.random((20, 20))),
        (
            tifffile_reader,
            example_data_imagej,
            np.random.randint(0, 255, size=(20, 20)).astype(np.uint8),
        ),
        (
            tifffile_reader,
            example_data_tiff,
            np.random.randint(0, 255, size=(20, 20)).astype(np.uint8),
        ),
        (
            tifffile_reader,
            example_data_ometiff,
            np.random.randint(0, 255, size=(20, 20)).astype(np.uint8),
        ),
        (zip_reader, example_data_zipped_filepath, np.random.random((20, 20))),
    ],
)
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
    if data_fixture == example_data_zipped_filepath:  # zipfile has unsqueezed dimension
        np.testing.assert_allclose(original_data, layer_data_tuple[0][0])
    else:
        np.testing.assert_allclose(original_data, layer_data_tuple[0])


def test_multiresolution_image(example_data_multiresolution):
    """Test opening a multi-resolution image."""
    assert example_data_multiresolution.series[0].is_pyramidal
    layer_data_list = tifffile_reader(example_data_multiresolution)
    layer_data_tuple = layer_data_list[0]
    layer_data = layer_data_tuple[0]
    assert len(layer_data) == 3
    assert layer_data[0].shape == (16, 512, 512, 3)
    assert layer_data[1].shape == (16, 256, 256, 3)
    assert layer_data[2].shape == (16, 128, 128, 3)
    assert all([isinstance(level, da.core.Array) for level in layer_data])


def test_shaped_singleton_image(example_data_shaped_singleton):
    """Test a plain, 'shaped' tiff with a leading singleton dimension.

    Regression test for the recorded 5D shape ``[1, 2, 14, 32, 32]``. Data and
    metadata must both come from ``series[0]`` so their dimensionality agrees,
    otherwise adding the layer to a viewer raises
    ``ValueError: units=... must have length ndim=...``.

    The assertions are deliberately version-agnostic: tifffile >=2026.5.2
    squeezes the leading singleton in the series view (data is 4D), while older
    versions keep it (data is 5D). Either way, data and metadata must match.
    """
    tif = example_data_shaped_singleton

    layer_data_list = tifffile_reader(tif)
    data, kwargs, _ = layer_data_list[0]

    # data always comes from series[0]
    assert data.shape == tif.series[0].shape

    # scale and units must match the data dimensionality
    assert len(kwargs["scale"]) == data.ndim
    assert len(kwargs["units"]) == data.ndim

    # the actual failure mode was on layer creation
    viewer = ViewerModel()
    add_images_to_viewer(viewer, layer_data_list)
    assert len(viewer.layers) == 1
    assert viewer.layers[0].ndim == data.ndim


@pytest.mark.parametrize("file_name", ['test_imagej.tiff', 'test_ome.tiff', 'test_imagej_with_time.tiff', 'test_imagej_greyscale.tiff', 'test_ome_with_time.tiff'])
def test_read_tiff_metadata_colormap(data_dir, file_name):
    """Test opening an ImageJ tiff."""
    viewer = ViewerModel()
    layer_data_list = reader_function(data_dir / file_name)
    add_images_to_viewer(viewer, layer_data_list)
    assert len(viewer.layers) == 2
    assert isinstance(viewer.layers[0], Image)
    if file_name == 'test_imagej_greyscale.tiff':
        # from issue #66 https://github.com/napari/napari-tiff/issues/66
        assert_array_equal(viewer.layers[0].colormap.colors[-1], (1, 1, 1, 1))
        assert_array_equal(viewer.layers[1].colormap.colors[-1], (1, 1, 1, 1))
    else:
        assert_array_equal(viewer.layers[0].colormap.colors[-1], (1, 0, 0, 1))
        assert_array_equal(viewer.layers[1].colormap.colors[-1], (0, 0, 1, 1))

@pytest.mark.parametrize("file_name", ['test_imagej.tiff', 'test_ome.tiff'])
def test_read_tiff_metadata_units(data_dir, file_name):
    viewer = ViewerModel()
    layer_data_list = reader_function(data_dir / file_name)
    add_images_to_viewer(viewer, layer_data_list)
    nm = pint.get_application_registry()['nm']
    layer_scale = [x*y for x, y in zip(viewer.layers[0].scale, viewer.layers[0].units)]
    assert_units_allclose(layer_scale, [210 * nm, 77.5224855 * nm, 77.5224855 * nm])


@pytest.mark.parametrize("file_name", ['test_imagej_with_time.tiff', 'test_ome_with_time.tiff'])
def test_read_tiff_metadata_units_with_time(data_dir, file_name):
    viewer = ViewerModel()
    layer_data_list = reader_function(data_dir / file_name)
    add_images_to_viewer(viewer, layer_data_list)
    nm = pint.get_application_registry()['nm']
    s = pint.get_application_registry()['s']
    layer_scale = [x*y for x, y in zip(viewer.layers[0].scale[1:], viewer.layers[0].units[1:])]
    pint.testing.assert_equal(viewer.layers[0].scale[0] * viewer.layers[0].units[0], 5 * s)
    assert_units_allclose(layer_scale, [77.5224855 * nm, 77.5224855 * nm])

def write_tiffs(tmp_path, images) -> list[str]:
    """Write each image to its own plain tiff file and return the paths."""
    paths = [str(tmp_path / f"image_{i}.tif") for i in range(len(images))]
    for path, image in zip(paths, images):
        tifffile.imwrite(path, image)
    return paths


def test_get_reader_list_of_paths(tmp_path):
    """A list of tiff paths (napari "Open Files as Stack") gets a reader."""
    paths = write_tiffs(tmp_path, [np.zeros((4, 4), np.uint8)] * 2)
    assert callable(napari_get_reader(paths))


def test_reader_stacks_list_of_paths(tmp_path):
    """Reading a list of tiff paths stacks the images along a new first axis."""
    images = [np.random.randint(0, 255, size=(20, 20)).astype(np.uint8) for _ in range(3)]
    paths = write_tiffs(tmp_path, images)

    reader = napari_get_reader(paths)
    layer_data_list = reader(paths)

    assert len(layer_data_list) == 1
    data, kwargs, layer_type = layer_data_list[0]
    assert layer_type == "image"
    np.testing.assert_array_equal(data, np.stack(images))

    viewer = ViewerModel()
    add_images_to_viewer(viewer, layer_data_list)
    assert viewer.layers[0].ndim == 3


@pytest.mark.parametrize("file_name", ['test_imagej.tiff', 'test_ome.tiff', 'test_imagej_with_time.tiff', 'test_imagej_greyscale.tiff', 'test_ome_with_time.tiff'])
def test_reader_stack_keeps_metadata(data_dir, file_name):
    """Stacking keeps channels, colormaps, scale and units of the files."""
    path = data_dir / file_name
    single = ViewerModel()
    add_images_to_viewer(single, reader_function(path))

    stacked = ViewerModel()
    add_images_to_viewer(stacked, reader_function([path, path, path]))

    assert len(stacked.layers) == len(single.layers)
    for layer, single_layer in zip(stacked.layers, single.layers):
        assert layer.data.shape == (3, *single_layer.data.shape)
        np.testing.assert_array_equal(layer.data[1], single_layer.data)
        assert_array_equal(layer.colormap.colors, single_layer.colormap.colors)
        # the new stack axis has unit scale and comes first
        assert layer.scale[0] == 1
        np.testing.assert_allclose(layer.scale[1:], single_layer.scale)
        assert layer.units[1:] == single_layer.units


def test_reader_stack_multiresolution(tmp_path):
    """Stacking multi-resolution tiffs stacks each resolution level."""
    data = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
    paths = []
    for i in range(2):
        path = tmp_path / f"pyramid_{i}.tif"
        with tifffile.TiffWriter(path) as tif:
            tif.write(data, subifds=1, tile=(16, 16))
            tif.write(data[::2, ::2], subfiletype=1, tile=(16, 16))
        paths.append(str(path))

    (levels, _, _), = reader_function(paths)
    assert [level.shape for level in levels] == [(2, 64, 64), (2, 32, 32)]
    np.testing.assert_array_equal(np.asarray(levels[0][1]), data)


def test_reader_stack_different_shapes_raises(tmp_path):
    """Files of different shapes cannot be stacked, and the error says which."""
    paths = write_tiffs(tmp_path, [np.zeros((4, 4), np.uint8), np.zeros((5, 5), np.uint8)])
    with pytest.raises(ValueError, match="same shape"):
        reader_function(paths)


def write_multifile_ome(tmp_path, positions) -> list[str]:
    """Write one OME-TIFF per position; each file's OME-XML lists all positions.

    This is the layout of multi-position acquisitions from pymmcore-plus /
    ome-writers: every image of the OME-XML points to its own file by UUID.
    """
    names = [f"pos_p{i:03}.ome.tif" for i in range(len(positions))]
    uuids = [f"urn:uuid:{uuid.uuid4()}" for _ in positions]
    size_t, size_y, size_x = positions[0].shape
    images = "".join(
        f'<Image ID="Image:{i}" Name="{i}">'
        f'<Pixels ID="Pixels:{i}" DimensionOrder="XYZCT" Type="uint8" '
        f'SizeX="{size_x}" SizeY="{size_y}" SizeZ="1" SizeC="1" SizeT="{size_t}">'
        f'<Channel ID="Channel:{i}:0" SamplesPerPixel="1"/>'
        f'<TiffData PlaneCount="{size_t}"><UUID FileName="{name}">{file_uuid}</UUID></TiffData>'
        "</Pixels></Image>"
        for i, (name, file_uuid) in enumerate(zip(names, uuids))
    )
    paths = []
    for name, file_uuid, data in zip(names, uuids, positions):
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" '
            f'UUID="{file_uuid}">{images}</OME>'
        )
        path = tmp_path / name
        with tifffile.TiffWriter(path) as tif:
            for i, plane in enumerate(data):  # one page per plane, as OME expects
                tif.write(plane, description=xml if i == 0 else None, metadata=None, contiguous=True)
        paths.append(str(path))
    return paths


@pytest.fixture
def multifile_ome(tmp_path):
    """Two positions of 2 timepoints; every pixel holds 10 * position + time."""
    positions = [np.full((2, 4, 4), 10 * p + np.arange(2)[:, None, None], np.uint8) for p in range(2)]
    return write_multifile_ome(tmp_path, positions), np.stack(positions)


@pytest.mark.parametrize("index", [0, 1])
def test_reader_multifile_ome_one_file(multifile_ome, index):
    """One file of a multi-file OME set opens only the position stored in it.

    The other files of the set are only read with "Open Files as Stack".
    """
    paths, expected = multifile_ome
    layer_data_list = reader_function(paths[index])

    assert len(layer_data_list) == 1
    data, kwargs, _ = layer_data_list[0]
    np.testing.assert_array_equal(data, expected[index])

    viewer = ViewerModel()
    add_images_to_viewer(viewer, layer_data_list)
    assert viewer.layers[0].ndim == 3  # time, y, x


def test_reader_stack_multifile_ome_set(multifile_ome):
    """Opening the files of a multi-file OME set as a stack does not repeat positions.

    Regression test: tifffile reads a series that lives in another file of the
    set from the opened file, so every position showed the opened file's data.
    """
    paths, expected = multifile_ome
    (data, _, _), = reader_function(paths)
    np.testing.assert_array_equal(data, expected)


def test_reader_stack_multifile_ome_subset(tmp_path):
    """Stacking some files of a multi-file OME set gives just those positions."""
    positions = [np.full((2, 4, 4), 10 * p, np.uint8) for p in range(3)]
    paths = write_multifile_ome(tmp_path, positions)

    (data, _, _), = reader_function([paths[0], paths[2]])
    np.testing.assert_array_equal(data, np.stack([positions[0], positions[2]]))


def test_reader_stack_two_multifile_ome_sets(tmp_path):
    """Stacking all files of two acquisitions stacks the acquisitions, not every file."""
    sets = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        positions = [np.full((2, 4, 4), 100 * len(sets) + 10 * p, np.uint8) for p in range(2)]
        sets.append((write_multifile_ome(tmp_path / name, positions), np.stack(positions)))

    (data, kwargs, _), = reader_function(sets[0][0] + sets[1][0])
    np.testing.assert_array_equal(data, np.stack([expected for _, expected in sets]))


@pytest.mark.parametrize("order", [1, -1])
def test_reader_stack_multifile_ome_renamed_file(tmp_path, order):
    """A renamed file of a multi-file OME set is still stacked as its position.

    The first file cannot find the renamed one by name, so it looks like a
    single-position file; the renamed file still describes the whole set.
    """
    positions = [np.full((2, 4, 4), 10 * p, np.uint8) for p in range(2)]
    paths = write_multifile_ome(tmp_path, positions)
    renamed = tmp_path / "pos_p003.ome.tif"
    os.rename(paths[1], renamed)
    paths = [paths[0], str(renamed)][::order]

    (data, _, _), = reader_function(paths)
    np.testing.assert_array_equal(data, np.stack(positions[::order]))
