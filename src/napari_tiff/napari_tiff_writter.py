import os.path
from typing import TYPE_CHECKING, Mapping, Literal
from itertools import product

import numpy as np
import pint
import tifffile

if TYPE_CHECKING:
    from napari.types import FullLayerData
    from napari.utils.colormaps import Colormap


OME_UNITS_LENGTH: set[str] = {
    "Ym", "Zm", "Em", "Pm", "Tm", "Gm", "Mm",
    "km", "hm", "dam", "m", "dm", "cm", "mm",
    "µm", "nm", "pm", "fm", "am", "zm", "ym",
    "Å",
    "thou", "li", "in", "ft", "yd", "mi",
    "ua", "ly", "pc",
    "pt", "pixel", "reference frame",
}

UNITS_TIME: set[str] = {
    "Ys", "Zs", "Es", "Ps", "Ts", "Gs", "Ms",
    "ks", "hs", "das", "s", "ds", "cs", "ms",
    "µs", "ns", "ps", "fs", "as", "zs", "ys",
    "min", "h", "d",
}

def get_pint_length_unit_to_shortcut() -> dict[pint.Unit, str]:
    register = pint.get_application_registry()
    return {
        register[x].units: x for x in OME_UNITS_LENGTH
    }

def get_pint_time_unit_to_shortcut() -> dict[pint.Unit, str]:
    register = pint.get_application_registry()
    return {
        register[x].units: x for x in UNITS_TIME
    }


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


def _rgba_to_signed_int(rgba: tuple[np.int32, np.int32, np.int32, np.int32]) -> np.int32:
    """Convert an RGB tuple to a signed integer representation."""
    r, g, b, a = rgba
    return np.int32((r << 24) | (g << 16) | (b << 8) | a)


def colormap_to_int(colormap: "Colormap") -> np.int32:
    """Convert colormap to int32.

    Warnings
    --------
    This function properly converts black to color type of colormaps.
    Need to be fixed in future
    """

    last_color = tuple((colormap.colors[-1] * 255).astype(np.int32))
    return _rgba_to_signed_int(last_color)


def prepare_metadata(
        layer_metadata: Mapping,
        file_name: str,
        data_shape: tuple[int, ...],
        channel_names: list[str],
        colormaps: list["Colormap"],
) -> dict:
    """Prepare metadata for writing based on first layer metadata.

    Parameters
    ----------
    layer_metadata: dict
        Dictionary of napari layer metadata.
    file_name: str
        Name of the file that will be written to, without extension.
    data_shape: tuple[int, ...]
        Shape of the image data. First coordinate is channel
    channel_names: list[str]
        List of channel names.
    colormaps: list[Colormap]
        List of colormaps.

    Returns
    -------
    dict
        Dictionary of OME metadata.
    """
    axis_order = layer_metadata["axis_order"]
    axis_position = {y: x for x, y in enumerate(axis_order)}
    axis_size = {y: data_shape[x] for x, y in enumerate(axis_order, start=1)}
    axis_to_unit = {}
    spatial_units_from_pint = get_pint_length_unit_to_shortcut()
    units = layer_metadata["units"]

    for axis in axis_order:
        if axis == 't':
            time_units_from_pint = get_pint_time_unit_to_shortcut()
            axis_to_unit['t'] = time_units_from_pint[units[axis_position['t']]]
        else:
            axis_to_unit[axis] = spatial_units_from_pint[units[axis_position[axis]]]
    scale = layer_metadata["scale"]
    translate = layer_metadata["translate"]
    axis_to_translate = {
        x: translate[axis_position[x]] for x in axis_order
    }
    plane_li = [
        {
            "TheT": t,  # codespell:ignore thet
            "TheZ": z,
            "TheC": c,
            "PositionZ": axis_to_translate.get('z', 0),
            "PositionY": axis_to_translate.get('y', 0),
            "PositionX": axis_to_translate.get('x', 0),
            "ExposureTime": axis_to_translate.get('t', 0),
            "PositionZUnit": axis_to_unit.get('z', 'pixel'),
            "PositionYUnit": axis_to_unit.get('y', 'pixel'),
            "PositionXUnit": axis_to_unit.get('x', 'pixel'),
            "ExposureTimeUnit": axis_to_unit.get('t', 's'),
        }
        for t, z, c in product(range(axis_size.get('t', 1)), range(axis_size.get('z', 1)), range(data_shape[0]))
    ]
    pixels = {
        "PhysicalSizeX": scale[axis_position['x']],
        "PhysicalSizeY": scale[axis_position['y']],
        "PhysicalSizeXUnit": axis_to_unit['x'],
        "PhysicalSizeYUnit": axis_to_unit['y'],
    }
    if "z" in axis_position:
        pixels["PhysicalSizeZ"] = scale[axis_position['z']]
        pixels["PhysicalSizeZUnit"] = axis_to_unit['z']
    if "t" in axis_position:
        pixels["PhysicalSizeT"] = scale[axis_position['t']]
        pixels["PhysicalSizeTUnit"] = axis_to_unit['t']

    metadata = {
        "Pixels": pixels,
        "Plane": plane_li,
        "Creator": "napari-tiff",
        "Channel": {
            "Name": channel_names,
            "axes": list(axis_order),
            "Color": [colormap_to_int(c) for c in colormaps],
        }
    }
    metadata["Name"] = file_name
    return metadata


def determine_axis_order(layer_metadata: Mapping) -> list[Literal["c", "z", "t", "y", "x"]]:
    """Determine the axis order of an image based on metadata."""
    axis_labels = layer_metadata["axis_labels"].lower()
    units = layer_metadata["units"]
    if set(axis_labels).issubset("ctzyx"):
        return list(axis_labels)
    register = pint.get_application_registry()
    mm = register["mm"].units
    pixels = register["pixels"].units
    s = register["s"].units
    if all(mm.is_compatible_with(x) for x in units) or all(x == pixels for x in units):
        if len(units) == 2:
            return ["y", "x"]
        elif len(units) == 3:
            return ["z", "y", "x"]
    raise ValueError("Cannot determine axis order")




def images_layer_writer(path: str, layer_data: list[FullLayerData]) -> list[str]:
    _chek_if_images_could_be_concatenated(layer_data)
    name = os.path.splitext(os.path.basename(path))[0]
    concatenated_data = np.stack([x[0] for x in layer_data])
    channel_names = [x[1]["name"] for x in layer_data]
    colormaps = [x[1]["colormap"] for x in layer_data]
    metadata = prepare_metadata(layer_data[0][1], name, concatenated_data.shape, channel_names, colormaps)

    tifffile.imwrite(path, concatenated_data, ome=True, metadata=metadata, software="napari-tiff", compression="ADOBE_DEFLATE")

    return [path]


