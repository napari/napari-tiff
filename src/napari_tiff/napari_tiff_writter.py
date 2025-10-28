import os.path
from typing import TYPE_CHECKING, Mapping

import numpy as np
import tifffile

if TYPE_CHECKING:
    from napari.types import FullLayerData



def _chek_if_images_could_be_concatenated(layer_data: list[FullLayerData]) -> None:
    """Check if images could be concatenated.

    Parameters
    ----------
    layer_data: list[FullLayerData]
        List of napari layer data to be checked.
    Raises
    ------
    ValueError
        If images could not be concatenated.
    """

    if len (set(x[0].shape for x in layer_data)) != 1:
        raise ValueError("All images must have the same shape to be concatenated and different channels of single image")
    if len (set(x[0].dtype for x in layer_data)) != 1:
        raise ValueError("All images must have the same dtype to be concatenated")
    if len (set(tuple(x[1]["scale"]) for x in layer_data)) != 1:
        raise ValueError("All images must have the same scale to be concatenated")


def prepate_metadata(layer_metadata: Mapping, file_name: str) -> dict:
    """Prepare metadata for writing based on first layer metadata."""
    spacing = layer_metadata["scale"]
    shift = layer_metadata["translate"]
    plane_li = [
        {
            "TheT": t,  # codespell:ignore thet
            "TheZ": z,
            "TheC": c,
            "PositionZ": shift[0] if len(shift) == 3 else 0,
            "PositionY": shift[-2],
            "PositionX": shift[-1],
            "PositionZUnit": "µm",
            "PositionYUnit": "µm",
            "PositionXUnit": "µm",
        }
        for t, z, c in product(range(image.times), range(image.layers), range(channels))
    ]

    metadata = {
        "Pixels": {
            "PhysicalSizeZ": spacing[0] if len(spacing) == 3 else 1,
            "PhysicalSizeY": spacing[-2],
            "PhysicalSizeX": spacing[-1],
            "TimeIncrement": image.time_increment,
            "PhysicalSizeZUnit": "µm",
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeXUnit": "µm",
            "TimeIncrementUnit": "s",
        },
        "Plane": plane_li,
        "Creator": "napari-tiff",
    }
    metadata["Name"] = file_name
    return metadata


def images_layer_writer(path: str, layer_data: list[FullLayerData]) -> list[str]:
    _chek_if_images_could_be_concatenated(layer_data)
    name = os.path.splitext(os.path.basename(path))[0]
    concatenated_data = np.stack([x[0] for x in layer_data])
    metadata = prepate_metadata(layer_data[0][1], name)

    tifffile.imwrite(path, concatenated_data, ome=True, metadata=metadata, software="napari-tiff", compression="ADOBE_DEFLATE")

    return [path]


