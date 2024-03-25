import numpy as np
from tifffile import TiffWriter


data = np.random.randint(0, 255, (8, 2, 512, 512, 3), 'uint8')
subresolutions = 2
pixelsize = 0.29  # micrometer
with TiffWriter('temp-pyramid.ome.tif', bigtiff=True) as tif:
    metadata={
        'axes': 'TCYXS',
        'SignificantBits': 8,
        'TimeIncrement': 0.1,
        'TimeIncrementUnit': 's',
        'PhysicalSizeX': pixelsize,
        'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': pixelsize,
        'PhysicalSizeYUnit': 'µm',
        'Channel': {'Name': ['Channel 1', 'Channel 2']},
        'Plane': {'PositionX': [0.0] * 16, 'PositionXUnit': ['µm'] * 16}
    }
    options = dict(
        photometric='rgb',
        tile=(128, 128),
        compression='jpeg',
        resolutionunit='CENTIMETER',
        maxworkers=2
    )
    tif.write(
        data,
        subifds=subresolutions,
        resolution=(1e4 / pixelsize, 1e4 / pixelsize),
        metadata=metadata,
        **options
    )
    # write pyramid levels to the two subifds
    # in production use resampling to generate sub-resolution images
    for level in range(subresolutions):
        mag = 2**(level + 1)
        tif.write(
            data[..., ::mag, ::mag, :],
            subfiletype=1,
            resolution=(1e4 / mag / pixelsize, 1e4 / mag / pixelsize),
            **options
        )
    # add a thumbnail image as a separate series
    # it is recognized by QuPath as an associated image
    thumbnail = (data[0, 0, ::8, ::8] >> 2).astype('uint8')
    tif.write(thumbnail, metadata={'Name': 'thumbnail'})

