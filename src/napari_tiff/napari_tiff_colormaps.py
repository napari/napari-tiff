from typing import Any

import numpy

CUSTOM_COLORMAPS = {}  # CUSTOM_COLORMAPS[colormap_hash] = colormap_name


def alpha_colormap(bitspersample=8, samples=4):
    """Return Alpha colormap."""
    n = 2**bitspersample
    ramp = numpy.linspace(0.0, 1.0, n).astype("float32")
    alpha_cmap = numpy.zeros((n, samples), dtype="float32")
    alpha_cmap[:, 3] = ramp[::-1]
    return {"name": "alpha",  "colors": alpha_cmap}


def int_to_rgba(intrgba: int) -> tuple:
    signed = intrgba < 0
    rgba = [x / 255 for x in intrgba.to_bytes(4, signed=signed, byteorder="big")]
    if rgba[-1] == 0:
        rgba[-1] = 1
    return tuple(rgba)


def qpi_color_to_rgba(color: Any) -> tuple[float, float, float, float] | None:
    """Convert a QPTIFF 'Color' element ("R,G,B", 0-255) to an RGBA tuple.

    Falls back to None, letting napari pick the colormaps.
    """
    if isinstance(color, str):
        color = color.split(",")
    if not isinstance(color, (list, tuple)) or len(color) != 3:
        return None
    try:
        # clamped because napari needs them between 0 and 1
        return (*(min(max(int(value), 0), 255) / 255 for value in color), 1.0)
    except (TypeError, ValueError):
        return None
