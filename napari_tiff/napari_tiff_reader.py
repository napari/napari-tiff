"""
This modeul is a napari reader for TIFF image files.

It implements the ``napari_get_reader`` hook specification, (to create
a reader plugin) but your plugin may choose to implement any of the hook
specifications offered by napari.
see: https://napari.org/docs/plugins/hook_specifications.html

Replace code below accordingly.  For complete documentation see:
https://napari.org/docs/plugins/for_plugin_developers.html
"""
from typing import List, Optional, Union, Any, Tuple, Dict, Callable

import numpy
from tifffile import TiffFile, TiffSequence, TIFF, xml2dict, PHOTOMETRIC
from vispy.color import Colormap

LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]
PathLike = Union[str, List[str]]
ReaderFunction = Callable[[PathLike], List[LayerData]]


def napari_get_reader(path: PathLike) -> Optional[ReaderFunction]:
    """Implements napari_get_reader hook specification.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]
    path = path.lower()
    if path.endswith('zip'):
        return zip_reader
    for ext in TIFF.FILE_EXTENSIONS:
        if path.endswith(ext):
            return reader_function
    return None


def reader_function(path: PathLike) -> List[LayerData]:
    """Return a list of LayerData tuples from path or list of paths."""
    # TODO: LSM
    with TiffFile(path) as tif:
        try:
            layerdata = tifffile_reader(tif)
        except Exception as exc:
            # fallback to imagecodecs
            log_warning(f'tifffile: {exc}')
            layerdata = imagecodecs_reader(path)
    return layerdata


def zip_reader(path: PathLike) -> List[LayerData]:
    """Return napari LayerData from sequence of TIFF in ZIP file."""
    with TiffSequence(container=path) as ims:
        data = ims.asarray()
    return [(data, {}, 'image')]


def tifffile_reader(tif):
    """Return napari LayerData from image series in TIFF file."""
    nlevels = len(tif.series[0])
    if nlevels > 1:
        import dask.array as da
        data = [da.from_zarr(tif.aszarr(level=level)) for level in range(nlevels)]
    else:
        data = tif.asarray()
    if tif.is_ome:
        kwargs = get_ome_tiff_metadata(tif)
    # TODO: combine interpretation of imagej and tags metadata?:
    elif tif.is_imagej:
        kwargs = get_imagej_metadata(tif)
    else:
        kwargs = get_tiff_metadata(tif)
    return [(data, kwargs, 'image')]


def get_tiff_metadata(tif):
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

    rgb = page.photometric in (2, 6) and shape[-1] in (3, 4)
    name = None
    scale = None
    colormap = None
    contrast_limits = None
    blending = None
    channel_axis = None
    visible = True

    if page.photometric == 5:
        # CMYK
        channel_axis = axes.find('S')
        if channel_axis >= 0 and shape[channel_axis] >= 4:
            colormap = cmyk_colormaps()
            name = ['Cyan', 'Magenta', 'Yellow', 'Black']
            visible = [False, False, False, True]
            blending = ['additive', 'additive', 'additive', 'additive']
            # TODO: use subtractive blending
        else:
            channel_axis = None
    elif (
        page.photometric in (2, 6) and (
            page.planarconfig == 2 or
            (page.bitspersample > 8 and dtype.kind in 'iu') or
            (extrasamples and len(extrasamples) > 1)
        )
    ):
        # RGB >8-bit or planar, or with multiple extrasamples
        channel_axis = axes.find('S')
        if channel_axis >= 0 and shape[channel_axis] > 2:
            rgb = False
            visible = [True, True, True]
            colormap = ['red', 'green', 'blue']  # rgb_colormaps()
            name = ['Red', 'Green', 'Blue']
            blending = ['additive', 'additive', 'additive']
        else:
            channel_axis = None
    elif (
        page.photometric in (0, 1) and
        extrasamples and
        any(sample > 0 for sample in extrasamples)
    ):
        # Grayscale with alpha channel
        channel_axis = axes.find('S')
        if channel_axis >= 0:
            visible = [True]
            colormap = ['gray']
            name = ['Minisblack' if page.photometric == 1 else 'Miniswhite']
            blending = ['additive']
        else:
            channel_axis = None

    if channel_axis is not None and extrasamples:
        # add extrasamples
        for sample in extrasamples:
            if sample == 0:
                # UNSPECIFIED
                visible.append(False)  # hide by default
                colormap.append('gray')
                name.append('Extrasample')
                blending.append('additive')
            else:
                # alpha channel
                # TODO: handle ASSOCALPHA and UNASSALPHA
                visible.append(True)
                colormap.append(alpha_colormap())
                name.append('Alpha')
                blending.append('translucent')

    if channel_axis is None and page.photometric in (0, 1):
        # separate up to 3 samples in grayscale images
        channel_axis = axes.find('S')
        if channel_axis >= 0 and 1 < shape[channel_axis] < 4:
            n = shape[channel_axis]
            colormap = ['red', 'green', 'blue', 'gray',
                        'cyan', 'magenta', 'yellow'][:n]
            name = [f'Sample {i}' for i in range(n)]
        else:
            channel_axis = None

    if channel_axis is None:
        # separate up to 3 channels
        channel_axis = axes.find('C')
        if channel_axis > 0 and 1 < shape[channel_axis] < 4:
            n = shape[channel_axis]
            colormap = ['red', 'green', 'blue', 'gray',
                        'cyan', 'magenta', 'yellow'][:n]
            name = [f'Channel {i}' for i in range(n)]
        else:
            channel_axis = None

        if page.photometric == 3 and page.colormap is not None:
            # PALETTE
            colormap = page.colormap
            if numpy.max(colormap) > 255:
                colormap = colormap / 65535.0
            else:
                colormap = colormap / 255.0
            colormap = Colormap(colormap.astype('float32').T)

    if colormap is None and page.photometric == 0:
        # MINISBLACK
        colormap = 'gray_r'

    if (
        contrast_limits is None and
        dtype.kind == 'u' and
        page.photometric != 3 and
        page.bitspersample not in (8, 16, 32, 64)
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


def get_imagej_metadata(tif):
    """Return napari LayerData from ImageJ hyperstack."""
    # TODO: ROI overlays
    ijmeta = tif.imagej_metadata
    series = tif.series[0]

    dtype = series.dtype
    axes = series.axes
    shape = series.shape
    page = series.pages[0]
    rgb = page.photometric == 2 and shape[-1] in (3, 4)
    mode = ijmeta.get('mode', None)
    channels = ijmeta.get('channels', 1)
    channel_axis = None

    name = None
    scale = None
    colormap = None
    contrast_limits = None
    blending = None
    visible = True

    if mode in ('composite', 'color'):
        channel_axis = axes.find('C')
        if channel_axis < 0:
            channel_axis = None

    if channel_axis is not None:
        channels = shape[channel_axis]
        channel_only = channels == ijmeta.get('images', 0)

        if 'LUTs' in ijmeta:
            colormap = [Colormap(c.T / 255.0) for c in ijmeta['LUTs']]
        elif mode == 'grayscale':
            colormap = 'gray'
        elif channels < 8:
            colormap = ['red', 'green', 'blue', 'gray',
                        'cyan', 'magenta', 'yellow'][:channels]

        if 'Ranges' in ijmeta:
            contrast_limits = numpy.array(ijmeta['Ranges']).reshape(-1, 2)
            contrast_limits = contrast_limits.tolist()

        if channel_only and 'Labels' in ijmeta:
            name = ijmeta['Labels']
        elif channels > 1:
            name = [f'Channel {i}' for i in range(channels)]

        if mode in ('color', 'grayscale'):
            blending = 'opaque'

    elif axes[-1] == 'S' and dtype == 'uint16':
        # RGB >8-bit
        channel_axis = axes.find('S')
        if channel_axis >= 0 and shape[channel_axis] in (3, 4):
            rgb = False
            n = shape[channel_axis]
            visible = [True, True, True]
            colormap = rgb_colormaps(samples=4)[:n]
            name = ['Red', 'Green', 'Blue', 'Alpha'][:n]
            blending = ['additive', 'additive', 'additive', 'translucent'][:n]
        else:
            channel_axis = None

    scale = {}
    res = page.tags.get('XResolution')
    if res is not None:
        scale['X'] = res.value[1] / max(res.value[0], 1)
    res = page.tags.get('YResolution')
    if res is not None:
        scale['Y'] = res.value[1] / max(res.value[0], 1)
    scale['Z'] = abs(ijmeta.get('spacing', 1.0))
    if channel_axis is None:
        scale = tuple(scale.get(x, 1.0) for x in axes if x != 'S')
    else:
        scale = tuple(scale.get(x, 1.0) for x in axes if x not in 'CS')

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


def get_ome_tiff_metadata(tif):
    metadata = xml2dict(tif.ome_metadata)
    if 'OME' in metadata:
        metadata = metadata['OME']

    series = tif.series[0]
    shape = series.shape
    dtype = series.dtype
    axes = series.axes.lower().replace('s', 'c')
    if 'c' in axes:
        channel_axis = axes.index('c')
        nchannels = shape[channel_axis]
    else:
        channel_axis = None
        nchannels = 1

    image = ensure_list(metadata.get('Image', {}))[0]
    pixels = image.get('Pixels', {})

    pixel_size = []
    size = float(pixels.get('PhysicalSizeX', 0))
    if size > 0:
        pixel_size.append(get_value_units_micrometer(size, pixels.get('PhysicalSizeXUnit')))
    size = float(pixels.get('PhysicalSizeY', 0))
    if size > 0:
        pixel_size.append(get_value_units_micrometer(size, pixels.get('PhysicalSizeYUnit')))

    channels = ensure_list(pixels.get('Channel', []))
    if len(channels) > nchannels:
        nchannels = len(channels)

    is_rgb = (series.keyframe.photometric == PHOTOMETRIC.RGB and nchannels in (3, 4))

    names = []
    contrast_limits = []
    colormaps = []
    blendings = []
    visibles = []

    scale = None
    if pixel_size:
        scale = pixel_size

    for channeli, channel in enumerate(channels):
        name = channel.get('Name')
        color = channel.get('Color')
        colormap = None
        if color:
            colormap = int_to_rgba(int(color))
        elif is_rgb and len(channels) > 1:
            # separate RGB channels
            colormap = ['red', 'green', 'blue', 'alpha'][channeli]
            if not name:
                name = colormap

        contrast_limit = None
        if dtype.kind != 'f':
            info = numpy.iinfo(dtype)
            contrast_limit = (info.min, info.max)

        blending = 'additive'
        visible = True

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

    meta = dict(
        rgb=is_rgb,
        channel_axis=channel_axis,
        name=names,
        scale=scale,
        colormap=colormaps,
        contrast_limits=contrast_limits,
        blending=blendings,
        visible=visibles,
    )
    return meta


def imagecodecs_reader(path):
    """Return napari LayerData from first page in TIFF file."""
    from imagecodecs import imread
    return [(imread(path), {}, 'image')]


def ensure_list(x):
    if not isinstance(x, (list, tuple)):
        x = [x]
    return x


def alpha_colormap(bitspersample=8, samples=4):
    """Return Alpha colormap."""
    n = 2**bitspersample
    ramp = numpy.linspace(0.0, 1.0, n).astype('float32')
    a = numpy.zeros((n, samples), dtype='float32')
    a[:, 3] = ramp[::-1]
    return Colormap(a)


def rgb_colormaps(bitspersample=8, samples=3):
    """Return RGB colormaps."""
    n = 2**bitspersample
    ramp = numpy.linspace(0.0, 1.0, n).astype('float32')
    r = numpy.zeros((n, samples), dtype='float32')
    r[:, 0] = ramp
    g = numpy.zeros((n, samples), dtype='float32')
    g[:, 1] = ramp
    b = numpy.zeros((n, samples), dtype='float32')
    b[:, 2] = ramp
    if samples > 3:
        r[:, 3:] = 1.0
        g[:, 3:] = 1.0
        b[:, 3:] = 1.0
    return [Colormap(r), Colormap(g), Colormap(b)]


def cmyk_colormaps(bitspersample=8, samples=3):
    """Return CMYK colormaps."""
    n = 2**bitspersample
    ramp = numpy.linspace(1.0, 0.0, n).astype('float32')
    c = numpy.zeros((n, samples), dtype='float32')
    c[:, 1] = ramp
    c[:, 2] = ramp
    m = numpy.zeros((n, samples), dtype='float32')
    m[:, 0] = ramp
    m[:, 2] = ramp
    y = numpy.zeros((n, samples), dtype='float32')
    y[:, 0] = ramp
    y[:, 1] = ramp
    k = numpy.zeros((n, samples), dtype='float32')
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
    signed = (intrgba < 0)
    rgba = [x / 255 for x in intrgba.to_bytes(4, signed=signed, byteorder="big")]
    if rgba[-1] == 0:
        rgba[-1] = 1
    return tuple(rgba)


def get_value_units_micrometer(value: float, unit: str = None) -> float:
    conversions = {'nm': 1e-3, 'µm': 1, 'um': 1, 'micrometer': 1, 'mm': 1e3, 'cm': 1e4, 'm': 1e6}
    if unit:
        value_um = value * conversions.get(unit, 1)
    else:
        value_um = value
    return value_um


def log_warning(msg, *args, **kwargs):
    """Log message with level WARNING."""
    import logging
    logging.getLogger(__name__).warning(msg, *args, **kwargs)
