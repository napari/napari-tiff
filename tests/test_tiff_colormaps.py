import numpy as np
import pytest

from napari_tiff.napari_tiff_colormaps import (
    alpha_colormap,
    int_to_rgba,
    qpi_color_to_rgba,
)


def test_alpha_colormap_default_shape_and_name():
    cmap = alpha_colormap()
    assert cmap["name"] == "alpha"
    assert cmap["colors"].shape == (256, 4)
    assert cmap["colors"].dtype == np.float32


def test_alpha_colormap_alpha_ramp_runs_from_opaque_to_transparent():
    colors = alpha_colormap()["colors"]
    np.testing.assert_allclose(colors[:, 3], np.linspace(1.0, 0.0, 256), rtol=1e-6)


def test_alpha_colormap_rgb_channels_are_black():
    colors = alpha_colormap()["colors"]
    np.testing.assert_array_equal(colors[:, :3], np.zeros((256, 3), dtype="float32"))


@pytest.mark.parametrize("bitspersample", [1, 4, 12])
def test_alpha_colormap_length_follows_bitspersample(bitspersample):
    colors = alpha_colormap(bitspersample=bitspersample)["colors"]
    assert colors.shape == (2**bitspersample, 4)
    assert colors[0, 3] == 1.0
    assert colors[-1, 3] == 0.0


def test_int_to_rgba_unpacks_packed_rgba_value():
    # Bytes are read most significant first as R, G, B, A.
    # 0xFF0000FF is red 255 with alpha 255.
    assert int_to_rgba(0xFF0000FF) == (1.0, 0.0, 0.0, 1.0)
    # 0x00FF00FF is green 255 with alpha 255.
    assert int_to_rgba(0x00FF00FF) == (0.0, 1.0, 0.0, 1.0)


def test_int_to_rgba_handles_negative_signed_value():
    # A signed 32-bit -1 is 0xFFFFFFFF, so every channel is 255.
    assert int_to_rgba(-1) == (1.0, 1.0, 1.0, 1.0)


def test_int_to_rgba_replaces_zero_alpha_with_opaque():
    # 0xFF804000 has alpha byte 0, which is turned into 1 so the color shows.
    assert int_to_rgba(0xFF804000) == (1.0, 128 / 255, 64 / 255, 1)


@pytest.mark.parametrize(
    # xml2dict parses a "R,G,B" element into a tuple of ints
    "color, expected",
    [
        ((0, 0, 255), (0.0, 0.0, 1.0, 1.0)),
        ((255, 128, 0), (1.0, 128 / 255, 0.0, 1.0)),
        ((255, 255, 255), (1.0, 1.0, 1.0, 1.0)),
        ((0, 0, 0), (0.0, 0.0, 0.0, 1.0)),
        ((17, 200, 3), (17 / 255, 200 / 255, 3 / 255, 1.0)),
    ],
)
def test_qpi_color_to_rgba(color, expected):
    assert qpi_color_to_rgba(color) == expected
