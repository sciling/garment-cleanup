import googleapiclient.discovery
import cv2 as cv
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

PROJECT_ID="reverberant-joy-184509"
BUCKET_NAME='scilings_random_test_bucket'
MODEL_NAME='MicoletPredictor'
VERSION_NAME='v1'

arguments = {"url":'https://sciling.com/img/Micolet/picture2.jpg',
        "background_color": [241,241,241],
        "file_resolution": (900, 1170),
        "margin": (114, 114, 114, 114),
        "size_for_thread_detection": (400, 400),
        "max_degree_correction": 5,
        "unet_resolution": (256, 256),
        "unet_margin": (100, 100, 100, 100),
        "unet_mask_threshold": (0, 1),
        "jpg_quality": 100}

service = googleapiclient.discovery.build('ml', 'v1')
name = 'projects/{}/models/{}/versions/{}'.format(PROJECT_ID, MODEL_NAME, VERSION_NAME)

response = service.projects().predict(
    name=name,
    body={'instances': arguments}
).execute()

if 'error' in response:
    raise RuntimeError(response['error'])
else:
    nparr = np.asarray(bytearray(response['predictions'].encode("latin1")), dtype=np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    cv.imwrite("predictions.jpg", img)
