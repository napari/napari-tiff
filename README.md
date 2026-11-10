# napari-tiff

[![License](https://img.shields.io/pypi/l/napari-tiff.svg?color=green)](https://github.com/napari/napari-tiff/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-tiff.svg?color=green)](https://pypi.org/project/napari-tiff)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-tiff.svg?color=green)](https://python.org)
[![tests](https://github.com/napari/napari-tiff/workflows/test-and-deploy/badge.svg)](https://github.com/napari/napari-tiff/actions)
[![codecov](https://codecov.io/gh/napari/napari-tiff/branch/main/graph/badge.svg)](https://codecov.io/gh/napari/napari-tiff)

A [napari] reader plugin for TIFF-like images that uses [tifffile](https://github.com/cgohlke/tifffile).

----------------------------------

## Supported files

The plugin supports reading TIFF-like images that `tifffile` recognizes, including:

- Standard TIFF and BigTIFF: `.tif`, `.tiff`
- OME-TIFF: `.ome.tif`, `.ome.tiff`
- Whole-slide multiscale/pyramidal formats: `.svs`, `.ndpi`, `.scn`
- Multiplexed imaging: `.qptiff`, `.qpi`

For the full list, see [`napari.yaml`](https://github.com/napari/napari-tiff/blob/main/src/napari_tiff/napari.yaml).

Please note that: 

- In most cases, only the first series is loaded.
- Support for complex structure or metadata might be incomplete -- please open issues!
- Multiscale/pyramidal data is loaded lazily, but single-scale data is read fully into memory.


## Metadata handling

Where the file provides the information, the plugin handles:

- **Channels**: multichannel images (other than standard RGB) are split into
  one layer per channel, with names and colormaps taken from the metadata when
  present.
- **Scale and units**: physical pixel sizes and their units are applied to the
  layer's scale and units.
- **RGB, CMYK, palette and alpha**: RGB images are usually shown as a single
  RGB layer; CMYK, palette and alpha are mapped to suitable napari colormaps
  and blending modes.

The plugin currently supports the following metadata:

| Format | What is used |
| --- | --- |
| OME-TIFF | Channel names and colors, physical sizes, time increment, axis labels |
| ImageJ hyperstacks | Channel labels, LUTs, display ranges, pixel size, Z spacing, frame interval |
| QPTIFF | Biomarker or channel names and colors; per-channel instrument metadata |
| SVS | Microns per pixel (MPP); the full SVS description |
| Plain TIFF | Resolution tags (XResolution, YResolution, ResolutionUnit) and photometric interpretation |

Note: all metadata that `tifffile` parses from the file (for example 
`ome_metadata` or `imagej_metadata`) is also stored in each
layer's `metadata` dictionary.

## Installation

You can install `napari-tiff` via [pip]:

    pip install napari-tiff

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-tiff" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://napari.org
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[file an issue]: https://github.com/napari/napari-tiff/issues
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
