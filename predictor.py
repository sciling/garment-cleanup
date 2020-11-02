import os
import numpy as np
import tensorflow as tf
import urllib3
import cv2 as cv
import tools
from base64 import b64encode
import gc
import defaults

class MicoletPredictor(object):
    def __init__(self, model):
        self._model = model

    def _process_url(self, url):
        try:
            data = urllib3.PoolManager().request('GET', url)
        except requests.exceptions.RequestException as e:
            return {"error": "Could not download the image from the given url. {}".format(e)}

        try:
            source = cv.imdecode(np.asarray(bytearray(data.data), dtype='uint8'),flags=1)

            output_filename = os.path.join("/tmp/output/" + str(np.random.randint(100000000)) + ".jpg")

            # * Initial detection and removal of background.
            cleaned = tools.background_removal_v2(source, self.background_color)

            # * Re-orientation of the garment.
            source_reoriented, cleaned_reoriented = tools.garment_reorientation_v2(
                source,
                cleaned,
                self.size_for_thread_detection,
                self.max_degree,
                self.background_color
            )
            del source; del cleaned; gc.collect()

            # * Initial centering and zooming of the garment.
            _, image = tools.crop_garment(cleaned_reoriented, source_reoriented, self.unet_margin)
            del cleaned_reoriented; del source_reoriented; gc.collect()

            # * Detection and removal of background by unet.
            image = tools.unet_background_removal(image, self._model, self.unet_input_resolution)

            # * Fixing color of the image before removing the background.
            image = tools.rescale_intensity(image, 25, 240)

            # * Apply mask to the image
            image = tools.apply_mask_background_removal(image, self.background_color, self.unet_mask_threshold)

            # * Final centering and zooming of the garment.
            image = tools.crop_garment(image)

            # * Resizing the final image.
            image = tools.image_resize(image, self.margin, self.file_resolution, self.background_color)

            if self.file_size is not None:
                # * Writting the image to a JPG file with a maximum size of %s bytes.' % file_size
                tools.image_write(output_filename, image, file_size=self.file_size)
            else:
                # * Writting the image to a JPG file with %s %% compression quality.' % jpg_quality
                tools.image_write(output_filename, image, jpg_quality=self.jpg_quality)

            with open(output_filename, 'rb') as f:
                data = f.read()
                b64data = b64encode(data)
                return b64data.decode('utf8')

        except RuntimeError as e:
            return {"error": '* The image "{}" could not be processed. {}'.format(url, e)}

    def _process_urls(self, url_list):
        result_list = []
        for url in url_list:
            result_list.append(self._process_url(url))
        return result_list

    def predict(self, instances, **kwargs):

        self.file_size = instances["file_size"] if "file_size" \
            in instances.keys() else defaults.file_size
        self.jpg_quality = instances["jpg_quality"] if "jpg_quality" \
            in instances.keys() else defaults.jpg_quality
        self.file_resolution = instances["file_resolution"] if "file_resolution" \
            in instances.keys() else defaults.file_resolution
        self.margin = instances["margin"] if "margin" \
            in instances.keys() else defaults.margin
        self.background_color = instances["background_color"] if "background_color" \
            in instances.keys() else defaults.background_color
        self.max_degree = instances["max_degree"] if "max_degree" \
            in instances.keys() else defaults.max_degree
        self.size_for_thread_detection = instances["size_for_thread_detection"] if "size_for_thread_detection" \
            in instances.keys() else deafults.size_for_thread_detection
        self.unet_input_resolution = instances["unet_input_resolution"] if "unet_input_resolution" \
            in instances.keys() else defaults.unet_input_resolution
        self.unet_margin = instances["unet_margin"] if "unet_margin" \
            in instances.keys() else defaults.unet_margin
        self.unet_mask_threshold = instances["unet_mask_threshold"] if "unet_mask_threshold" \
            in instances.keys() else defaults.unet_mask_threshold

        try:
            if "url" in instances.keys():
                url = instances["url"]
                return self._process_url(url)
            elif "urls" in instances.keys():
                urls = instances["urls"]
                return self._process_urls(urls)
            else:
                return {"error": "The url to the image file to process is necessary."}
        except Exception as e:
            return {"error": "Found error {}".format(e)}

    @classmethod
    def from_path(cls, model_dir):
        model_path = os.path.join(model_dir, 'model.h5')
        model = tf.keras.models.load_model(model_path)

        return cls(model)
