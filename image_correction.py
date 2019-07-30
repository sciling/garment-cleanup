#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script performs the following tasks in order to process the input images:

* Re-orientation of the garment, in case it is rotated in the original image
* Detection and removal of background
* Centering and zooming of the garment
* White normalization
* The resulting image will have a max. size of 100KB.

Created on Fri Mar 15 10:24:15 2019

Author: Emilio Granell <egranell@sciling.com>

(c) 2019 Sciling, SL
"""

from tools import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images.')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-fs', '--file_size', default=100000)
    parser.add_argument('-fr', '--file_resolution', default='(900, 1170)')
    parser.add_argument('-m', '--margin', default='(114, 114, 114, 114)')
    parser.add_argument('-bc', '--background_color', default='[241, 241, 241]')
    parser.add_argument('-ptd', '--size_for_thread_detection', default='(400, 400)')
    parser.add_argument('-md', '--max_degree_correction', default=5)
    parser.add_argument('-show', '--show_images', action='store_true')
    parser.add_argument('-b', '--filter_block_sizes', default='(3, 91)')
    parser.add_argument('-c', '--filter_constant', default='(6, 11)')
    parser.add_argument('-n', '--unet_model_path', default=None)
    parser.add_argument('-nr', '--unet_resolution', default='(256, 256)')
    parser.add_argument('-nm', '--unet_margin', default='(100, 100, 100, 100)')
    parser.add_argument('-nmt', '--unet_mask_threshold', default='(0, 1)')
    parser.add_argument('-ic', '--illumination_correction', action='store_true')

    args = parser.parse_args()

    input = args.input
    output = args.output

    file_size = int(args.file_size)
    file_resolution = ast.literal_eval(args.file_resolution)
    margin = ast.literal_eval(args.margin)
    background_color = ast.literal_eval(args.background_color)
    max_degree = int(args.max_degree_correction)
    size_for_thread_detection = ast.literal_eval(args.size_for_thread_detection)
    show = args.show_images
    correct_illu = args.illumination_correction
 
    B_values = ast.literal_eval(args.filter_block_sizes)
    C_values = ast.literal_eval(args.filter_constant)

    model_path = args.unet_model_path
    unet_input_resolution = ast.literal_eval(args.unet_resolution)
    unet_margin = ast.literal_eval(args.unet_margin)
    unet_mask_threshold = ast.literal_eval(args.unet_mask_threshold)

    if os.path.exists(input):
        img_lst = []

        if os.path.isfile(input):
            img_lst.append(input)
            input_path = os.path.split(input)[0]
            # Remove first / if the files are in an absolute path
            input_path = input_path[1:] if input_path.startswith('/') else input_path
            output_path = os.path.join(output, input_path.replace('../',''))

            if not os.path.exists(output_path):
                os.makedirs(output_path)
        else:
            for path, subdirs, files in os.walk(input):
                # Remove first / if the files are in an absolute path
                path = path[1:] if path.startswith('/') else path
                output_path = os.path.join(output, path.replace('../',''))

                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                for name in files:
                    img_lst.append(os.path.join(path, name))
    else:
        print("The input does not exist:", input)
        sys.exit(0)

    if model_path is not None:
        from keras.models import load_model

        try:
            model = load_model(model_path)
        except OSError as e:
            print("Error when loading the model:", e)
            sys.exit(0)
    else:
        model = None

    for img_filename in img_lst:
        print("\nProcessing the image:", img_filename)

        try:
            source = cv.imread(img_filename, 1)
            # Remove first / if the files are in an absolute path
            img_filename = img_filename[1:] if img_filename.startswith('/') else img_filename
            output_filename = os.path.join(output, img_filename.replace('../',''))

            if model is None:
                print("* Re-orientation of the garment.")
                reoriented = garment_reorientation(source, 
                        size_for_thread_detection, max_degree, background_color)

                print("* Detection and removal of background.")
                cleaned = background_removal(reoriented, background_color, B_values, C_values, correct_illu)

                print("* Centering and zooming of the garment.")
                image = crop_garment(cleaned)
            else:
                print("* Detection and removal of background by unet.")
                image = unet_background_removal(source, model, unet_input_resolution)
                image = apply_mask_background_removal(image, background_color, unet_mask_threshold, correct_illu)

                print("* Re-orientation of the garment.")
                _, image = garment_reorientation_v2(source, image, size_for_thread_detection, max_degree, background_color)

                print("* Final centering and zooming of the garment.")
                image = crop_garment(image)

            print("* Resizing the final image.")
            image = image_resize(image, margin, file_resolution, background_color)

            print("* Writting the image to a JPG file with a maximum size of %s bytes." % file_size)
            image_write(output_filename, image, file_size)

            print("* The processed image has been saved as:", output_filename)

            if show:
                cv.imshow("Input", cv.resize(source, (int(0.25 * source.shape[1]),
                    int(0.25 * source.shape[0])), interpolation=cv.INTER_CUBIC))

                cv.imshow("Output", cv.resize(image, (int(0.75 * image.shape[1]),
                    int(0.75 * image.shape[0])), interpolation=cv.INTER_CUBIC))

                key = cv.waitKey(30)
                if key == 27:
                    break
        except cv.error as e:
            print("* The image {} could not be processed.".format(img_filename))
    if show:
        cv.destroyAllWindows()
