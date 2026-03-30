import numpy

CUSTOM_COLORMAPS = {}  # CUSTOM_COLORMAPS[colormap_hash] = colormap_name


def alpha_colormap(bitspersample=8, samples=4):
    """Return Alpha colormap."""
    n = 2**bitspersample
    ramp = numpy.linspace(0.0, 1.0, n).astype("float32")
    alpha_cmap = numpy.zeros((n, samples), dtype="float32")
    alpha_cmap[:, 3] = ramp[::-1]
    return {"name": "alpha",  "colors": alpha_cmap}


def _int_to_rgba_bytes(intrgba: int) -> list[int]:
    """Return RGBA bytes from a packed OME channel color integer."""
    signed = intrgba < 0
    rgba = list(intrgba.to_bytes(4, signed=signed, byteorder="big"))
    if rgba[-1] == 0:
        rgba[-1] = 255
    return rgba


def int_to_rgba(intrgba: int) -> tuple:
    """Return normalized RGBA values from a packed OME channel color integer."""
    return tuple(x / 255 for x in _int_to_rgba_bytes(intrgba))


def int_to_hex(intrgba: int) -> str:
    """Return a ``#rrggbbaa`` string from a packed OME channel color integer."""
    return f"#{bytes(_int_to_rgba_bytes(intrgba)).hex()}"
