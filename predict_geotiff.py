import os
from itertools import product
from glob import glob
import click
import numpy as np
import tensorflow as tf
from tensorflow import keras
import rasterio as rio
from rasterio import windows
from rasterio import features
from rasterio.merge import merge

from skimage.color import rgba2rgb

@click.command()
@click.option('--tifsource',help='Path to the input tiff-file.')
@click.option('--model', help='Path to the model which should be used. (should take the same input dimensions as the tilesize specified)')
@click.option('--prediction_threshold', default=0.5, help='The probability threshold applied when predicting the binary mask. Use False to output probabilities.')
@click.option('--targetdir', help='Folder where the output should be stored.')
@click.option('--preprocess', default="DeepLabV3", help='Type of preprocessing to perform.')
@click.option('--tilesize', default=512, help='Tilesize used to tile the input tifsource.')
@click.option('--overlap', default=0, required=False, help='Overlap percentage used to tile the input (0=no overlap).')
@click.option('--drop_alpha_threshold', required=False, default=0.0, help='Use to drop tiles with a set ratio of alpha values (default 0 exclusively drops tiles that are fully transparant).')
@click.option('--rowlimit', default=-1, required=False, help='Number of rows of size <tilesize> to read from the input tif.')
@click.option('--merge', default=True, required=False, help='Merge the final predictions into one rasterfile.')


def predict_controler(tifsource, model, prediction_threshold, targetdir, preprocess, tilesize, rowlimit, overlap, drop_alpha_threshold, merge):
    tf.get_logger().setLevel('ERROR')

    click.echo(f"Source {tifsource}")
    click.echo(f"Model {model}")
    click.echo(f"Prediction threshold {prediction_threshold}")
    click.echo(f"Target dir {targetdir}")
    click.echo(f"Preprocess {preprocess}")
    click.echo(f"Tilesize {tilesize}")
    click.echo(f"Overlap {overlap}")
    click.echo(f"Drop alpha threshold {drop_alpha_threshold}")
    click.echo(f"Rowlimit {rowlimit}")

    rowlimit = rowlimit * tilesize

    # Assert files exist
    for file in [tifsource, model]:
        assert os.path.exists(file), f"{file} doesn't exist"

    # Load the model
    model = keras.models.load_model(model, compile=False)

    # Read tiles and metadata
    for i, (tile, meta) in enumerate(tile_tif(tifsource, tilesize, overlap, rowlimit, drop_alpha_threshold)):
        image_tensor = preprocessor(tile, preprocess, tilesize)
        predicted = infer(model, image_tensor)

        # print(predicted)
        if prediction_threshold:
            predicted = predicted > prediction_threshold

        meta.update(count=1)

        # Write to file with georeference
        with rio.open(f"{targetdir}/{i}_mask.tif", 'w', **meta) as out:
            out.write(predicted, indexes=1)

    if merge:
        merge_masks(targetdir)



def merge_masks(targetdir):
    path = f'{targetdir}/*_mask.tif'
    raster_files = glob(path)

    raster_to_mosiac = []
    for p in raster_files:
        raster = rio.open(p)
        raster_to_mosiac.append(raster)

    mosaic, output = merge(raster_to_mosiac)

    output_meta = raster.meta.copy()
    output_meta.update(
        {"driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": output,
        }
    )

    with rio.open(f"{targetdir}/final.tif", "w", **output_meta) as m:
        m.write(mosaic)


def preprocessor(image, preprocess, size):
    if preprocess == 'DeepLabV3':
        image = image.transpose((1, 2, 0))[:,:,:3]
        image = tf.convert_to_tensor(image)
        # image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[size, size])
        image = image / 127.5 - 1
        return image
    else:
        raise NotImplementedError

def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
#     predictions = np.argmax(predictions, axis=2)
    return predictions


def tile_tif(tifsource, tilesize, overlap, rowlimit, drop_alpha_threshold):
    '''
    Open tif file and initialize tiles
    '''

    # Read the raster source 
    with rio.open(tifsource) as src:

        # Save the metadata
        meta = src.meta.copy()
        ncols, nrows = src.meta['width'], src.meta['height']

        # Calculate stepsize and offsets
        step = int(round(tilesize-tilesize*overlap))
        offsets = list(product(range(0, nrows, step), range(0, ncols, step)))

        # Enveloping window if the source tiff
        big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)
        size = None

        # Get the tile bounding boxes 
        for i, data in enumerate(offsets):
            row_off, col_off = data
            window = windows.Window(col_off=col_off, row_off=row_off, width=tilesize, height=tilesize).intersection(big_window)
            transform = windows.transform(window, src.transform)

            # Hard break of rowlimit is specified
            if rowlimit > 0 and window.row_off > rowlimit:
                break

            # Copy transformation params to metadata dict
            meta['transform'] = transform 
            meta['width'], meta['height'] = window.width, window.height
            meta.update(compress='lzw')

            # # Calculate the number of alpha values
            tile = src.read(window=window)

            # Skip record if alpha threshold is met
            if drop_alpha_threshold != 0:
                nonzero = np.count_nonzero(tile)

                # Calculate size once
                if not size:
                    size = np.size(tile)
                prc = (size-nonzero)/size

                # If more than x% is background, drop the tile.
                if prc >= drop_alpha_threshold:
                    continue
                
            yield tile, meta

if __name__ == '__main__':
    predict_controler()
