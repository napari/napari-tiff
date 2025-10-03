import os
import zipfile

import numpy as np
import pytest
import tifffile


def example_data_filepath(tmp_path, original_data):
    example_data_filepath = str(tmp_path / "example_data_filepath.tif")
    tifffile.imwrite(example_data_filepath, original_data, imagej=False)
    return example_data_filepath


def example_data_zipped_filepath(tmp_path, original_data):
    example_tiff_filepath = str(tmp_path / "myfile.tif")
    tifffile.imwrite(example_tiff_filepath, original_data, imagej=False)
    example_zipped_filepath = str(tmp_path / "myfile.zip")
    with zipfile.ZipFile(example_zipped_filepath, "w") as myzip:
        myzip.write(example_tiff_filepath)
    os.remove(example_tiff_filepath)  # not needed now the zip file is saved
    return example_zipped_filepath


def example_data_tiff(tmp_path, original_data):
    example_data_filepath = str(tmp_path / "example_data_tiff.tif")
    tifffile.imwrite(example_data_filepath, original_data, imagej=False)
    return tifffile.TiffFile(example_data_filepath)


def example_data_imagej(tmp_path, original_data):
    example_data_filepath = str(tmp_path / "example_data_imagej.tif")
    tifffile.imwrite(example_data_filepath, original_data, imagej=True)
    return tifffile.TiffFile(example_data_filepath)


def example_data_ometiff(tmp_path, original_data):
    example_data_filepath = str(tmp_path / "example_data_ometiff.ome.tif")
    tifffile.imwrite(example_data_filepath, original_data, imagej=False)
    return tifffile.TiffFile(example_data_filepath)


@pytest.fixture(scope="session")
def imagej_hyperstack_image(tmp_path_factory):
    """ImageJ hyperstack tiff image.

    Write a 10 fps time series of volumes with xyz voxel size 2.6755x2.6755x3.9474
    micron^3 to an ImageJ hyperstack formatted TIFF file:
    """
    filename = tmp_path_factory.mktemp("data") / "imagej_hyperstack.tif"

    volume = np.random.randn(6, 57, 256, 256).astype("float32")
    image_labels = [f"{i}" for i in range(volume.shape[0] * volume.shape[1])]
    metadata = {
        "spacing": 3.947368,
        "unit": "um",
        "finterval": 1 / 10,
        "fps": 10.0,
        "axes": "TZYX",
        "Labels": image_labels,
    }
    tifffile.imwrite(
        filename,
        volume,
        imagej=True,
        resolution=(1.0 / 2.6755, 1.0 / 2.6755),
        metadata=metadata,
    )
    return (filename, metadata)


@pytest.fixture
def example_data_multiresolution(tmp_path):
    """Example multi-resolution tiff file.

    Write a multi-dimensional, multi-resolution (pyramidal), multi-series OME-TIFF
    file with metadata. Sub-resolution images are written to SubIFDs. Limit
    parallel encoding to 2 threads.

    This example code reproduced from tifffile.py, see:
    https://github.com/cgohlke/tifffile/blob/2b5a5208008594976d4627bcf01355fc08837592/tifffile/tifffile.py#L649-L688
    """
    example_data_filepath = str(tmp_path / "test-pyramid.ome.tif")
    data = np.random.randint(0, 255, (8, 2, 512, 512, 3), "uint8")
    subresolutions = 2  # so 3 resolution levels in total
    pixelsize = 0.29  # micrometer
    with tifffile.TiffWriter(example_data_filepath, bigtiff=True) as tif:
        metadata = {
            "axes": "TCYXS",
            "SignificantBits": 8,
            "TimeIncrement": 0.1,
            "TimeIncrementUnit": "s",
            "PhysicalSizeX": pixelsize,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixelsize,
            "PhysicalSizeYUnit": "µm",
            "Channel": {"Name": ["Channel 1", "Channel 2"]},
            "Plane": {"PositionX": [0.0] * 16, "PositionXUnit": ["µm"] * 16},
        }
        options = dict(
            photometric="rgb",
            tile=(128, 128),
            compression="jpeg",
            resolutionunit="CENTIMETER",
            maxworkers=2,
        )
        tif.write(
            data,
            subifds=subresolutions,
            resolution=(1e4 / pixelsize, 1e4 / pixelsize),
            metadata=metadata,
            **options,
        )
        # write pyramid levels to the two subifds
        # in production use resampling to generate sub-resolution images
        for level in range(subresolutions):
            mag = 2 ** (level + 1)
            tif.write(
                data[..., ::mag, ::mag, :],
                subfiletype=1,
                resolution=(1e4 / mag / pixelsize, 1e4 / mag / pixelsize),
                **options,
            )
        return tifffile.TiffFile(example_data_filepath)
