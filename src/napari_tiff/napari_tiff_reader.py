"""
This modeul is a napari reader for TIFF image files.

It implements the ``napari_get_reader`` hook specification, (to create
a reader plugin) but your plugin may choose to implement any of the hook
specifications offered by napari.
see: https://napari.org/docs/plugins/hook_specifications.html

Replace code below accordingly.  For complete documentation see:
https://napari.org/docs/plugins/for_plugin_developers.html
"""
from pathlib import Path

import dask.array as da
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from tifffile import TIFF, TiffFile, TiffSequence

from napari_tiff.napari_tiff_metadata import get_metadata

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
    if path.endswith("zip"):
        return zip_reader
    for ext in TIFF.FILE_EXTENSIONS:
        if path.endswith(ext):
            return reader_function
    return None


def reader_function(path: PathLike) -> List[LayerData]:
    """Return a list of LayerData tuples from path or list of paths."""
    if isinstance(path, list):
        # napari hands over a list of paths for "Open Files as Stack"
        return stack_reader(path)
    # TODO: LSM
    with TiffFile(path) as tif:
        try:
            layerdata = tifffile_reader(tif)
        except Exception as exc:
            # fallback to imagecodecs
            log_warning(f"tifffile: {exc}")
            layerdata = imagecodecs_reader(path)
    return layerdata


def stack_reader(paths: List[str]) -> List[LayerData]:
    """Return napari LayerData stacking the images of TIFF files on a new first axis.

    Every file is read like a single file, so each must hold one image of the
    same shape. The layer metadata (channels, colormaps, scale, units) is taken
    from the first file, with a leading entry added for the new stack axis.

    Files of one multi-file OME-TIFF set (e.g. one file per position) are read
    together as a single image with a position axis holding the selected files,
    since each file of the set describes the whole set. If all paths belong to
    one set, no stack axis is added.
    """
    groups = _group_multifile_ome(paths)
    layers = []
    for path, series_indices in groups:
        if series_indices is None:
            layer_data_list = reader_function(path)
        else:
            layer_data_list = _ome_set_reader(path, series_indices)
        if len(layer_data_list) != 1:  # pragma: no cover
            raise ValueError(f"Cannot stack {path!r}: it holds {len(layer_data_list)} layers.")
        layers.append(layer_data_list[0])
    if len(layers) == 1:
        return layers
    paths = [path for path, _ in groups]

    datas = [data for data, _, _ in layers]
    first = datas[0]
    if isinstance(first, list):  # multiscale: stack each resolution level
        level_shapes = [[level.shape for level in data] for data in datas]
        if any(shapes != level_shapes[0] for shapes in level_shapes):
            raise _shape_mismatch_error(paths, level_shapes)
        data = [da.stack(levels) for levels in zip(*datas)]
    else:
        shapes = [d.shape for d in datas]
        if any(shape != shapes[0] for shape in shapes):
            raise _shape_mismatch_error(paths, shapes)
        data = da.stack(datas)

    return [(data, _prepend_axis(layers[0][1], "stack"), "image")]


def _multifile_ome_series(path: str) -> Optional[Dict[Path, int]]:
    """Return {file: series index} if path is one file of a multi-file OME set.

    Only sets whose series are positions (see ``_ome_positions``) and each live
    in one file count; for anything else return None.
    """
    try:
        with TiffFile(path) as tif:
            if not _ome_positions(tif):
                return None
            series_files = {}
            for index, series in enumerate(tif.series):
                files = {Path(page.parent.filehandle.path).resolve() for page in series.pages if page is not None}
                if len(files) != 1:
                    return None
                series_files[files.pop()] = index
    except Exception as exc:
        log_warning(f"Cannot read the OME series of {path}: {exc}")
        return None
    if len(series_files) != len(tif.series):  # several series in one file
        return None
    return series_files


def _group_multifile_ome(paths: List[str]) -> List[Tuple[str, Optional[List[int]]]]:
    """Group paths into (path, series indices) per multi-file OME set.

    Paths that are not part of a multi-file OME set get ``None`` as indices.
    The files of a set keep the order of paths, so the selected positions do too.

    A file may not reveal its set on its own, e.g. when another file of the set
    was renamed: tifffile cannot find it by name and drops it, while the renamed
    file still finds the others (it matches the UUID its set refers to). So a
    file is also grouped into a set that another selected file describes.
    """
    resolved = {path: Path(path).resolve() for path in paths}
    cache: Dict[str, Optional[Dict[Path, int]]] = {}

    def series_files_of(path: str) -> Optional[Dict[Path, int]]:
        if path not in cache:
            cache[path] = _multifile_ome_series(path)
        return cache[path]

    groups: List[Tuple[str, Optional[List[int]]]] = []
    claimed = set()  # files already grouped into a multi-file OME set
    for path in paths:
        if resolved[path] in claimed:
            continue
        # the file's own set first; other files are only opened if needed
        owner = next(
            (
                candidate
                for candidate in (path, *paths)
                if resolved[candidate] not in claimed
                and resolved[path] in (series_files_of(candidate) or {})
            ),
            None,
        )
        if owner is None:
            groups.append((path, None))
            continue
        series_files = series_files_of(owner)
        members = [resolved[p] for p in paths if resolved[p] in series_files and resolved[p] not in claimed]
        claimed.update(members)
        groups.append((owner, [series_files[member] for member in members]))
    return groups


def _ome_set_reader(path: str, series_indices: List[int]) -> List[LayerData]:
    """Return napari LayerData of the selected positions of a multi-file OME set."""
    with TiffFile(path) as tif:
        data = da.stack([_read_series(tif, tif.series[index]) for index in series_indices])
        metadata_kwargs = get_metadata(tif)
    return [(data, _prepend_axis(metadata_kwargs, "p"), "image")]


def _prepend_axis(metadata: Dict[str, Any], label: str) -> Dict[str, Any]:
    """Return layer metadata for data with a new leading axis."""
    metadata = dict(metadata)
    if metadata.get("channel_axis") is not None:
        metadata["channel_axis"] += 1
    # per-axis values exclude the channel axis, so only the new axis is added
    if metadata.get("scale") is not None:
        metadata["scale"] = (1.0, *metadata["scale"])
    if metadata.get("units") is not None:
        metadata["units"] = ("pixel", *metadata["units"])
    if metadata.get("axis_labels") is not None:
        labels = tuple(metadata["axis_labels"])
        metadata["axis_labels"] = (_unique_label(label, labels), *labels)
    return metadata


def _shape_mismatch_error(paths: List[str], shapes: list) -> ValueError:
    listing = "\n".join(f"  {shape}: {path}" for path, shape in zip(paths, shapes))
    return ValueError(f"Cannot stack TIFF files: they must all have the same shape.\n{listing}")


def _unique_label(label: str, labels: Tuple[str, ...]) -> str:
    while label in labels:
        label = f"{label}_"
    return label


def zip_reader(path: PathLike) -> List[LayerData]:
    """Return napari LayerData from sequence of TIFF in ZIP file."""
    with TiffSequence(container=path) as ims:
        data = ims.asarray()
    return [(data, {}, "image")]


def tifffile_reader(tif: TiffFile) -> List[LayerData]:
    """Return napari LayerData from image series in TIFF file."""
    nlevels = len(tif.series[0].levels)
    positions = []
    if nlevels > 1:
        import zarr
        store = tif.aszarr(multiscales=True)
        group = zarr.open_group(store=store, mode='r')
        try:
            datasets = group.attrs['ome']['multiscales'][0]['datasets']
        except KeyError:
            datasets = group.attrs['multiscales'][0]['datasets']

        # using group.attrs to get multiscales is recommended by cgohlke
        # default dask chunk is 128MiB, so use 1 MiB, which is more reasonable for visualization
        data = [da.from_zarr(group[path_dict['path']], chunks='1 MiB') for path_dict in datasets]
        # assert array shapes are in descending order for napari multiscale image
        shapes = [arr.shape for arr in data]
        assert shapes == list(reversed(sorted(shapes)))
    else:
        # Only positions stored in this file. In a multi-file OME set (one file
        # per position) the other files are read only with "Open Files as Stack".
        positions = [series for series in _ome_positions(tif) if _is_in_file(tif, series)]
        if len(positions) > 1:
            data = da.stack([_read_series(tif, series) for series in positions])
        elif positions:
            data = _read_series(tif, positions[0])
        else:
            # explicitly use series[0] to get the data
            data = _read_series(tif, tif.series[0])

    metadata_kwargs = get_metadata(tif)
    if len(positions) > 1:
        metadata_kwargs = _prepend_axis(metadata_kwargs, "p")

    return [(data, metadata_kwargs, "image")]


def _ome_positions(tif: TiffFile) -> list:
    """Return the series of an OME-TIFF that are positions of one acquisition.

    Multi-position acquisitions are stored as several OME images of identical
    shape, often one file per position. Return them so that they can be stacked
    on a position axis, or an empty list if the file is not like that.
    """
    if not tif.is_ome or len(tif.series) < 2:
        return []
    first = tif.series[0]
    for series in tif.series:
        if (
            series.shape != first.shape
            or series.dtype != first.dtype
            or series.axes != first.axes
            or len(series.levels) > 1
        ):
            return []
    return list(tif.series)


def _is_in_file(tif: TiffFile, series) -> bool:
    """Return True if the data of series is stored in the file of tif."""
    pages = [page for page in series.pages if page is not None]
    return bool(pages) and all(page.parent is tif for page in pages)


def _read_series(tif: TiffFile, series) -> np.ndarray:
    """Return the data of a series as a numpy array."""
    if any(page is not None and page.parent is not tif for page in series.pages):
        # The series lives in another file of a multi-file OME-TIFF. In that
        # case series.asarray() reads contiguous data from the opened file
        # instead (tifffile <= 2026.11.8), so read it page by page via zarr.
        import zarr

        return np.asarray(zarr.open(series.aszarr(), mode="r"))
    return series.asarray()


def imagecodecs_reader(path: PathLike):
    """Return napari LayerData from first page in TIFF file."""
    from imagecodecs import imread

    return [(imread(path), {}, "image")]


def log_warning(msg, *args, **kwargs):
    """Log message with level WARNING."""
    import logging

    logging.getLogger(__name__).warning(msg, *args, **kwargs)
