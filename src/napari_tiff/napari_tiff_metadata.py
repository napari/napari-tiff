from typing import Any

import numpy
from tifffile import PHOTOMETRIC, TiffFile, xml2dict

from napari_tiff.napari_tiff_colormaps import alpha_colormap, int_to_rgba, CUSTOM_COLORMAPS


def get_metadata(tif: TiffFile) -> dict[str, Any]:
    """Return metadata keyword argument dictionary for napari layer."""
    if tif.is_ome:
        metadata_kwargs = get_ome_tiff_metadata(tif)
    # TODO: combine interpretation of imagej and tags metadata?:
    elif tif.is_imagej:
        metadata_kwargs = get_imagej_metadata(tif)
    else:
        metadata_kwargs = get_tiff_metadata(tif)

    # napari does not use this this extra `metadata` layer attribute
    # but storing the information on the layer next to the data
    # will allow users to access it and use it themselves if they wish
    metadata_kwargs["metadata"] = get_extra_metadata(tif)

    return metadata_kwargs


def get_extra_metadata(tif: TiffFile) -> dict[str, Any]:
    """Return any non-empty tif metadata properties as a dictionary."""
    metadata_dict = {}
    empty_metadata_values = [None, "", (), []]
    for name in dir(tif.__class__):
        if "metadata" in name:
            metadata_value = getattr(tif.__class__, name).__get__(tif)
            if metadata_value not in empty_metadata_values:
                print(metadata_value)
                if isinstance(metadata_value, str):
                    try:
                        metadata_value = xml2dict(metadata_value)
                    except Exception:
                        pass
                metadata_dict[name] = metadata_value
    return metadata_dict


def get_tiff_metadata(tif: TiffFile) -> dict[str, Any]:
    """Return napari metadata from largest image series in TIFF file."""
    # TODO: fix (u)int32/64
    # TODO: handle complex
    series = tif.series[0]
    for s in tif.series:
        if s.size > series.size:
            series = s
    dtype = series.dtype
    axes = series.axes
    shape = series.shape
    page = next(p for p in series.pages if p is not None)
    extrasamples = page.extrasamples

    rgb = page.photometric in (PHOTOMETRIC.RGB, PHOTOMETRIC.YCBCR) and shape[-1] in (3, 4)
    name = None
    scale = None
    colormap = None
    contrast_limits = None
    blending = None
    channel_axis = None
    visible = True

    if page.photometric == PHOTOMETRIC.SEPARATED:
        # CMYK
        channel_axis = axes.find("S")
        if channel_axis >= 0 and shape[channel_axis] >= 4:
            colormap = ["cyan", "magenta", "yellow", "gray"]
            name = ["Cyan", "Magenta", "Yellow", "Black"]
            visible = [False, False, False, True]
            blending = ["additive", "additive", "additive", "additive"]
            # TODO: use subtractive blending
        else:
            channel_axis = None
    elif page.photometric in (PHOTOMETRIC.RGB, PHOTOMETRIC.YCBCR) and (
        page.planarconfig == 2
        or (page.bitspersample > 8 and dtype.kind in "iu")
        or (extrasamples and len(extrasamples) > 1)
    ):
        # RGB >8-bit or planar, or with multiple extrasamples
        channel_axis = axes.find("S")
        if channel_axis >= 0 and shape[channel_axis] > 2:
            rgb = False
            visible = [True, True, True]
            colormap = ["red", "green", "blue"]  # rgb_colormaps()
            name = ["Red", "Green", "Blue"]
            blending = ["additive", "additive", "additive"]
        else:
            channel_axis = None
    elif (
        page.photometric in (PHOTOMETRIC.MINISWHITE, PHOTOMETRIC.MINISBLACK)
        and extrasamples
        and any(sample > 0 for sample in extrasamples)
    ):
        # Grayscale with alpha channel
        channel_axis = axes.find("S")
        if channel_axis >= 0:
            visible = [True]
            colormap = ["gray"]
            name = ["Minisblack" if page.photometric == PHOTOMETRIC.MINISBLACK else "Miniswhite"]
            blending = ["additive"]
        else:
            channel_axis = None

    if channel_axis is not None and extrasamples:
        # add extrasamples
        for sample in extrasamples:
            if sample == 0:
                # UNSPECIFIED
                visible.append(False)  # hide by default
                colormap.append("gray")
                name.append("Extrasample")
                blending.append("additive")
            else:
                # alpha channel
                # TODO: handle ASSOCALPHA and UNASSALPHA
                visible.append(True)
                colormap.append(alpha_colormap())
                name.append("Alpha")
                blending.append("translucent")

    if channel_axis is None and page.photometric in (PHOTOMETRIC.MINISWHITE, PHOTOMETRIC.MINISBLACK):
        # separate up to 3 samples in grayscale images
        channel_axis = axes.find("S")
        if channel_axis >= 0 and 1 < shape[channel_axis] < 4:
            n = shape[channel_axis]
            colormap = ["red", "green", "blue", "gray", "cyan", "magenta", "yellow"][:n]
            name = [f"Sample {i}" for i in range(n)]
        else:
            channel_axis = None

    if channel_axis is None:
        # separate up to 3 channels
        channel_axis = axes.find("C")
        if channel_axis > 0 and 1 < shape[channel_axis] < 4:
            n = shape[channel_axis]
            colormap = ["red", "green", "blue", "gray", "cyan", "magenta", "yellow"][:n]
            name = [f"Channel {i}" for i in range(n)]
        else:
            channel_axis = None

        if page.photometric == PHOTOMETRIC.PALETTE and page.colormap is not None:
            # PALETTE
            colormap_values = page.colormap
            if numpy.max(colormap_values) > 255:
                colormap_values = colormap_values / 65535.0
            else:
                colormap_values = colormap_values / 255.0
            colormap_values = colormap_values.astype("float32").T
            # set up custom colormap
            colormap_hash = hash(tuple(tuple(x) for x in colormap_values))
            if colormap_hash in CUSTOM_COLORMAPS:
                colormap_name = CUSTOM_COLORMAPS[colormap_hash]
            else:
                colormap_name = "PALETTE: " + str(colormap_hash)
                CUSTOM_COLORMAPS[colormap_hash] = colormap_name
            colormap = {"name": colormap_name,  "colors": colormap_values}

    if colormap is None and page.photometric == PHOTOMETRIC.MINISWHITE:
        # MINISWHITE
        colormap = "gray_r"

    if (
        contrast_limits is None
        and dtype.kind == "u"
        and page.photometric != PHOTOMETRIC.PALETTE
        and page.bitspersample not in (8, 16, 32, 64)
    ):
        contrast_limits = (0, 2**page.bitspersample)
        if channel_axis is not None and shape[channel_axis] > 1:
            contrast_limits = [contrast_limits] * shape[channel_axis]

    kwargs = dict(
        rgb=rgb,
        channel_axis=channel_axis,
        name=name,
        scale=scale,
        colormap=colormap,
        contrast_limits=contrast_limits,
        blending=blending,
        visible=visible,
    )
    return kwargs


def get_imagej_metadata(tif: TiffFile) -> dict[str, Any]:
    """Return napari LayerData from ImageJ hyperstack."""
    # TODO: ROI overlays
    ijmeta = tif.imagej_metadata
    series = tif.series[0]

    dtype = series.dtype
    axes = series.axes
    shape = series.shape
    page = series.pages[0]
    rgb = page.photometric == PHOTOMETRIC.RGB and shape[-1] in (3, 4)
    mode = ijmeta.get("mode", None)
    channels = ijmeta.get("channels", 1)
    channel_axis = None

    name = None
    colormap = None
    contrast_limits = None
    blending = None
    visible = True

    if mode in ("composite", "color"):
        channel_axis = axes.find("C")
        if channel_axis < 0:
            channel_axis = None

    if channel_axis is not None:
        channels = shape[channel_axis]
        channel_only = channels == ijmeta.get("images", 0)

        if "LUTs" in ijmeta:
            colormap = [Colormap(c.T / 255.0) for c in ijmeta["LUTs"]]
        elif mode == "grayscale":
            colormap = "gray"
        elif channels < 8:
            colormap = ["red", "green", "blue", "gray", "cyan", "magenta", "yellow"][
                :channels
            ]

        if "Ranges" in ijmeta:
            contrast_limits = numpy.array(ijmeta["Ranges"]).reshape(-1, 2)
            contrast_limits = contrast_limits.tolist()

        if channel_only and "Labels" in ijmeta:
            name = ijmeta["Labels"]
        elif channels > 1:
            name = [f"Channel {i}" for i in range(channels)]

        if mode in ("color", "grayscale"):
            blending = "opaque"

    elif axes[-1] == "S" and dtype == "uint16":
        # RGB >8-bit
        channel_axis = axes.find("S")
        if channel_axis >= 0 and shape[channel_axis] in (3, 4):
            rgb = False
            n = shape[channel_axis]
            visible = [True, True, True]
            colormap = ["red", "green", "blue", alpha_colormap()]
            name = ["Red", "Green", "Blue", "Alpha"][:n]
            blending = ["additive", "additive", "additive", "translucent"][:n]
        else:
            channel_axis = None

    scale = {}
    res = page.tags.get("XResolution")
    if res is not None:
        scale["X"] = res.value[1] / max(res.value[0], 1)
    res = page.tags.get("YResolution")
    if res is not None:
        scale["Y"] = res.value[1] / max(res.value[0], 1)
    scale["Z"] = abs(ijmeta.get("spacing", 1.0))
    if channel_axis is None:
        scale = tuple(scale.get(x, 1.0) for x in axes if x != "S")
    else:
        scale = tuple(scale.get(x, 1.0) for x in axes if x not in "CS")

    kwargs = dict(
        rgb=rgb,
        channel_axis=channel_axis,
        name=name,
        scale=scale,
        colormap=colormap,
        contrast_limits=contrast_limits,
        blending=blending,
        visible=visible,
    )
    return kwargs


def get_ome_tiff_metadata(tif: TiffFile) -> dict[str, Any]:
    ome_metadata = xml2dict(tif.ome_metadata).get("OME")
    image_metadata = ensure_list(ome_metadata.get("Image", {}))[0]
    pixels = image_metadata.get("Pixels", {})

    pixel_size = []
    size = float(pixels.get("PhysicalSizeX", 0))
    if size > 0:
        pixel_size.append(
            get_value_units_micrometer(size, pixels.get("PhysicalSizeXUnit"))
        )
    size = float(pixels.get("PhysicalSizeY", 0))
    if size > 0:
        pixel_size.append(
            get_value_units_micrometer(size, pixels.get("PhysicalSizeYUnit"))
        )

    series = tif.series[0]
    shape = series.shape
    dtype = series.dtype
    axes = series.axes.lower().replace("s", "c")
    if "c" in axes:
        channel_axis = axes.index("c")
        nchannels = shape[channel_axis]
    else:
        channel_axis = None
        nchannels = 1

    channels = ensure_list(pixels.get("Channel", []))
    if len(channels) > nchannels:
        nchannels = len(channels)

    is_rgb = series.keyframe.photometric == PHOTOMETRIC.RGB and nchannels in (3, 4)

    if is_rgb:
        # channels_axis appears to be incompatible with RGB channels
        channel_axis = None

    names = []
    contrast_limits = []
    colormaps = []
    blendings = []
    visibles = []

    scale = None
    if pixel_size:
        scale = pixel_size

    for channeli, channel in enumerate(channels):
        name = channel.get("Name")
        color = channel.get("Color")
        colormap = None
        if color:
            colormap = int_to_rgba(int(color))
        elif is_rgb and len(channels) > 1:
            # separate channels provided for RGB (with missing color)
            colormap = ["red", "green", "blue", alpha_colormap()][channeli]
            if not name:
                name = colormap

        blending = "additive"
        visible = True

        if dtype.kind == "f":
            contrast_limit = None
        else:
            info = numpy.iinfo(dtype)
            contrast_limit = (info.min, info.max)

        if len(channels) > 1:
            names.append(name)
            blendings.append(blending)
            contrast_limits.append(contrast_limit)
            colormaps.append(colormap)
            visibles.append(visible)
        else:
            names = name
            blendings = blending
            contrast_limits = contrast_limit
            colormaps = colormap
            visibles = visible

    kwargs = dict(
        rgb=is_rgb,
        channel_axis=channel_axis,
        name=names,
        scale=scale,
        colormap=colormaps,
        contrast_limits=contrast_limits,
        blending=blendings,
        visible=visibles,
    )
    return kwargs


def get_value_units_micrometer(value: float, unit: str = None) -> float:
    unit_conversions = {
        "nm": 1e-3,
        "µm": 1,
        "\\u00B5m": 1,  # Unicode 'MICRO SIGN' (U+00B5)
        "um": 1,
        "micrometer": 1,
        "mm": 1e3,
        "cm": 1e4,
        "m": 1e6,
    }
    if unit:
        value_um = value * unit_conversions.get(unit, 1)
    else:
        value_um = value
    return value_um


def ensure_list(x):
    if not isinstance(x, (list, tuple)):
        x = [x]
    return x
