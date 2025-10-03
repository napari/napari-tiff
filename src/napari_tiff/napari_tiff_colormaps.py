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
