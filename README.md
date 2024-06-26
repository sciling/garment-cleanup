# Image processing for garment photographs of GarmentCleanup

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
- -ic: Allows to deactivate the illumination correction, True by default.
- -show: Show the input and correted output images, False by default.

See the next example photograph:


<img src="img/picture.jpg"  width="384" height="576">


This photograph can be processed without using a U-Net deep learning model as follows:
```console
python3 image_correction.py -i img/picture.jpg -o output/
```
See the resulting image after the correction process:


<img src="img/corrected_1.jpg"  width="384" height="576">

If we have a U-Net deep learning model, we can use it for the background removal as follows:
```console
python3 image_correction.py -i img/picture.jpg -o output/ -n models/unet_garment-cleanup.hdf5
```
Then, the resulting image after the correction process is:


<img src="img/corrected_2.jpg"  width="384" height="576">


Now, suppose a directory *img* that contains the following structure:
- img/image_1.jpg
- img/image_2.jpg
- img/class_A/image_1.jpg
- img/class_A/image_2.jpg
- img/class_B/image_1.jpg
- img/class_B/image_2.jpg

The script can be used to process all the images contained in the directory *img* as follows:
```console
python3 image_correction.py -i img/ -o output/
```
Then, the processed images are written in the *output* directory maintaining the original directory structure:
- output/img/image_1.jpg
- output/img/image_2.jpg
- output/img/class_A/image_1.jpg
- output/img/class_A/image_2.jpg
- output/img/class_B/image_1.jpg
- output/img/class_B/image_2.jpg


## Instruccions to Deploy in Google AI Platform

### Export the application credentials and define some variables:

```console
export GOOGLE_APPLICATION_CREDENTIALS="/credentials.json"

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=scilings_random_test_bucket
REGION="us-central1"
```

### Create te package with the code:
```console
python setup.py sdist --formats=gztar
```

### Upload the code:
```console
gsutil cp ./dist/garment-cleanup-0.1.tar.gz gs://$BUCKET_NAME/garment-cleanup/garment-cleanup-0.1.tar.gz
```

### Upload the model:
```console
gsutil cp model.h5 gs://$BUCKET_NAME/garment-cleanup/model/
```

### Create a model:
```console
MODEL_NAME='GarmentCleanupPredictor'
gcloud ai-platform models create $MODEL_NAME \
  --regions $REGION
```

### Create a version:
```console
VERSION_NAME='v1'
gcloud beta ai-platform versions create $VERSION_NAME \
  --model $MODEL_NAME \
  --runtime-version 1.15 \
  --python-version 3.7 \
  --origin gs://$BUCKET_NAME/garment-cleanup/model/ \
  --package-uris gs://$BUCKET_NAME/garment-cleanup/garment-cleanup-0.1.tar.gz \
  --prediction-class predictor.GarmentCleanupPredictor \
  --machine-type=mls1-c4-m4 
```

### Test locally:
```console
gcloud ai-platform local predict --model-dir "./model/" \
      --json-instances sample.json 
```

### Test:
```console
gcloud ai-platform predict --model $MODEL_NAME --version $VERSION_NAME --json-instances sample.json
```

where sample.json may contain, for instance:
```
{"instances": {"url":"https://sciling.com/img/GarmentCleanup/picture2.jpg"}}
```

### Test through the python API:
```console
python3 test.py
```

### Test via the public API:
```console
curl -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json" -d @sample.json https://ml.googleapis.com/v1/projects/reverberant-joy-184509/models/GarmentCleanupPredictorG/versions/v1_31:predict | jq -r .predictions | base64 -d > output.jpg
```

### Delete version resource:
```console
gcloud ai-platform versions delete $VERSION_NAME --quiet --model $MODEL_NAME
```

### Delete model resource:
```console
gcloud ai-platform models delete $MODEL_NAME --quiet
```

### Delete Cloud Storage objects that were created:
```console
gsutil -m rm -r gs://$BUCKET_NAME/garment-cleanup
```
