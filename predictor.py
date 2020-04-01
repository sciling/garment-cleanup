import os
import pickle
import tempfile
import numpy as np
import tensorflow as tf
import requests
import cv2 as cv
import tools
import json
from base64 import b64encode

class MicoletPredictor(object):
  def __init__(self, model):
    self._model = model

  def predict(self, instances, **kwargs):

    if "url" not in instances.keys():
        return {"error": "The url to the image file to process is necessary."}
    else:
        url = instances["url"]

    file_size = instances["file_size"] if "file_size" \
            in instances.keys() else None
    jpg_quality = instances["jpg_quality"] if "jpg_quality" \
            in instances.keys() else 100
    file_resolution = instances["file_resolution"] if "file_resolution" \
            in instances.keys() else (900, 1170)
    margin = instances["margin"] if "margin" \
            in instances.keys() else (114, 114, 114, 114)
    background_color = instances["background_color"] if "background_color" \
            in instances.keys() else [241, 241, 241]
    max_degree = instances["max_degree"] if "max_degree" \
            in instances.keys() else 5
    size_for_thread_detection = instances["size_for_thread_detection"] if "size_for_thread_detection" \
            in instances.keys() else (400, 400)
    unet_input_resolution = instances["unet_input_resolution"] if "unet_input_resolution" \
            in instances.keys() else (256, 256)
    unet_margin = instances["unet_margin"] if "unet_margin" \
            in instances.keys() else (100, 100, 100, 100)
    unet_mask_threshold = instances["unet_mask_threshold"] if "unet_mask_threshold" \
            in instances.keys() else (0, 1)

    try:
        data = requests.get(url)
    except requests.exceptions.RequestException as e:
        return {"error": "Could not download the image from the given url. {}".format(e)}

    with tempfile.NamedTemporaryFile(delete=False) as temp:
        img_filename = temp.name
        temp.write(data.content)
        temp.flush()

        try:
            source = cv.imread(img_filename, 1)
            output_filename = os.path.join("/tmp/output/", os.path.basename(img_filename))

            # * Initial detection and removal of background.
            cleaned = tools.background_removal_v2(source, background_color)

            # * Re-orientation of the garment.
            source_reoriented, cleaned_reoriented = tools.garment_reorientation_v2(
                source,
                cleaned,
                size_for_thread_detection,
                max_degree,
                background_color
            )

            # * Initial centering and zooming of the garment.
            _, image = tools.crop_garment(cleaned_reoriented, source_reoriented, unet_margin)

            # * Detection and removal of background by unet.
            image = tools.unet_background_removal(image, self._model, unet_input_resolution)

            # * Fixing color of the image before removing the background.
            image = tools.rescale_intensity(image, 25, 240)

            # * Apply mask to the image
            image = tools.apply_mask_background_removal(image, background_color, unet_mask_threshold)

            # * Final centering and zooming of the garment.
            image = tools.crop_garment(image)

            # * Resizing the final image.
            image = tools.image_resize(image, margin, file_resolution, background_color)

            if file_size is not None:
                # * Writting the image to a JPG file with a maximum size of %s bytes.' % file_size
                tools.image_write(output_filename, image, file_size=file_size)
            else:
                # * Writting the image to a JPG file with %s %% compression quality.' % jpg_quality
                tools.image_write(output_filename, image, jpg_quality=jpg_quality)

            # Load final image to be send in the response
            image = cv.imread(output_filename, 1)
            
            bts = cv.imencode('.jpg', image)[1]
            return np.array(bts).tostring().decode("latin1")

        except RuntimeError as e:
            return {"error": '* The image "{}" could not be processed. {}'.format(url, e)}

    return results

  @classmethod
  def from_path(cls, model_dir):
    model_path = os.path.join(model_dir, 'model.h5')
    model = tf.keras.models.load_model(model_path)

    return cls(model)
