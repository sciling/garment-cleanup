from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from image_correction import main
import json
import logging
import argparse
import os

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _response(message, error: bool = False):
    return jsonify({'error': error, 'message': str(message)})


def init_server(args):
    # Init flask stuff
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    app.logger.addHandler(logging.StreamHandler())  # pylint: disable=E1101
    app.logger.setLevel(logging.DEBUG)  # pylint: disable=E1101

    logger = app.logger
    logging.basicConfig(format='%(name)s:%(levelname)s:%(asctime)s %(message)s',
                        datefmt='%H:%M:%S')

    @app.route('/', methods=['GET'])
    def hello():
        """
        Simple hello world
        """
        return _response('Welcome to sciling image prediction API.')

    @app.route('/predict', methods=['POST'])
    def predict():
        """
        POST /predict
        """
        if request.method == 'POST':
            if 'file' not in request.files:
                return _response('Not found file', True)
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                args.input = filepath 
                processed = main(args)
                return send_file(processed)
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images.')
    parser.add_argument('-o', '--output', default='/processed')
    parser.add_argument('-fs', '--file_size', default=100000)
    parser.add_argument('-fr', '--file_resolution', default='(900, 1170)')
    parser.add_argument('-m', '--margin', default='(114, 114, 114, 114)')
    parser.add_argument('-bc', '--background_color', default='[241, 241, 241]')
    parser.add_argument(
        '-ptd', '--size_for_thread_detection', default='(400, 400)')
    parser.add_argument('-md', '--max_degree_correction', default=5)
    parser.add_argument('-show', '--show_images', action='store_true')
    parser.add_argument('-b', '--filter_block_sizes', default='(3, 91)')
    parser.add_argument('-c', '--filter_constant', default='(6, 11)')
    parser.add_argument('-n', '--unet_model_path', default=None)
    parser.add_argument('-nr', '--unet_resolution', default='(256, 256)')
    parser.add_argument('-nm', '--unet_margin', default='(100, 100, 100, 100)')
    parser.add_argument('-nmt', '--unet_mask_threshold', default='(0, 1)')
    parser.add_argument('-ic', '--illumination_correction',
                        action='store_true')

    args = parser.parse_args()
    app = init_server(args)
    app.run(debug=True, host='0.0.0.0', port=5000)
