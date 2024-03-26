import numpy
from vispy.color import Colormap


def alpha_colormap(bitspersample=8, samples=4):
    """Return Alpha colormap."""
    n = 2**bitspersample
    ramp = numpy.linspace(0.0, 1.0, n).astype("float32")
    a = numpy.zeros((n, samples), dtype="float32")
    a[:, 3] = ramp[::-1]
    return Colormap(a)


def rgb_colormaps(bitspersample=8, samples=3):
    """Return RGB colormaps."""
    n = 2**bitspersample
    ramp = numpy.linspace(0.0, 1.0, n).astype("float32")
    r = numpy.zeros((n, samples), dtype="float32")
    r[:, 0] = ramp
    g = numpy.zeros((n, samples), dtype="float32")
    g[:, 1] = ramp
    b = numpy.zeros((n, samples), dtype="float32")
    b[:, 2] = ramp
    if samples > 3:
        r[:, 3:] = 1.0
        g[:, 3:] = 1.0
        b[:, 3:] = 1.0
    return [Colormap(r), Colormap(g), Colormap(b)]


def cmyk_colormaps(bitspersample=8, samples=3):
    """Return CMYK colormaps."""
    n = 2**bitspersample
    ramp = numpy.linspace(1.0, 0.0, n).astype("float32")
    c = numpy.zeros((n, samples), dtype="float32")
    c[:, 1] = ramp
    c[:, 2] = ramp
    m = numpy.zeros((n, samples), dtype="float32")
    m[:, 0] = ramp
    m[:, 2] = ramp
    y = numpy.zeros((n, samples), dtype="float32")
    y[:, 0] = ramp
    y[:, 1] = ramp
    k = numpy.zeros((n, samples), dtype="float32")
    k[:, 0] = ramp
    k[:, 1] = ramp
    k[:, 2] = ramp
    if samples > 3:
        c[:, 3:] = 1.0
        m[:, 3:] = 1.0
        y[:, 3:] = 1.0
        k[:, 3:] = 1.0
    return [Colormap(c), Colormap(m), Colormap(y), Colormap(k)]


def int_to_rgba(intrgba: int) -> tuple:
    signed = intrgba < 0
    rgba = [x / 255 for x in intrgba.to_bytes(4, signed=signed, byteorder="big")]
    if rgba[-1] == 0:
        rgba[-1] = 1
    return tuple(rgba)
