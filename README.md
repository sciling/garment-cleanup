# Image processing for garment photographs of Micolet

The image processing performs the following tasks in order to prepare the final garment images:

* Re-orientation of the garment, in case it is rotated in the original image.
* Detection and removal of background.
* Centering and zooming of the garment.
* White normalization.
* The resulting image will have a maximum size of 100KB.

## Intructions to use the image correction script

The scritp accepts a set of input arguments that allow you to adjust the main parameters that influence the different stages of image processing.
At least the input and output should be indicated. The input can be an image or a directory (which can contain both images and subdirectories) 
and the output should be a directory where the images will be saved according to their original names, taking into account the structure of subdirectories if necessary.

The parameters that can be modified are:
- -i: The input, image or directory.
- -o: The output directory.
- -fs: The final JPG file size in bytes, 100000 by default.
- -fr: The final JPG image resolution in pixels, (900, 1170) by default.
- -m: The margin around the garment in the final JPG image in pixels, (top, bottom, left, right), (114, 114, 114, 114) by default.
- -bc: The background color in 24-bits RGB codification, [241, 241, 241] by default.
- -ptd: The portion image size (in pixels) for thread detection,(400, 400) by default.
- -md: The maximum degree for orientation correction, 5 degrees by default.
- -b: The range of block sizes used in the adaptative threshold filter for background removal, (3, 91) by default.
- -c: The range of constant values used in the adaptive threshold filter for background removal, (6,11) by default.
- -n: The path to the unet model, None by default.
- -nr: The size of the unet input in pixels, (256, 256) by default.
- -nm: The margin added to the unet input image in pixels, (100, 100, 100, 100) by default
- -nmt: The thresholds to apply the predicted unet mask, the fist value indicates the maximum value in the range from zero that are assigned to the background, 
the second value indicates the minimum value in the range until one that are assigned to the garment, (0, 1) by default.
- -show: Show the input and correted output images, False by default.

See the next example photograph:
![Picture](img/picture.jpg = 250x)

This photograph can be processed without using a U-Net deep learning model as follows:
```console
python3 image_correction.py -i img/picture.jpg -o output/
```
See the resulting image after the correction process:
![Result without Deep Learning model](img/corrected_1 = 250x)


If we have a U-Net deep learning model, we can use it for the background removal as follows:
```console
python3 image_correction.py -i img/picture.jpg -o output/ -n models/unet_micolet.hdf5
```
Then, the resulting image after the correction process is:
![Result with Deep Learning model](img/corrected_2 = 250x)

