#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This script performs the following tasks in order to process the input images:

* Re-orientation of the garment, in case it is rotated in the original image
* Detection and removal of background
* Centering and zooming of the garment
* White normalization
* The resulting image will have a max. size of 100KB.

Created on Fri Mar 15 10:24:15 2019

Author: Emilio Granell <egranell@sciling.com>

(c) 2019 Sciling, SL
'''

import os
import sys
import ast
import argparse
import cv2 as cv
import tools


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process images.')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-fs', '--file_size')
    parser.add_argument('-fr', '--file_resolution', default='(900, 1170)')
    parser.add_argument('-m', '--margin', default='(114, 114, 114, 114)')
    parser.add_argument('-bc', '--background_color', default='[241, 241, 241]')
    parser.add_argument('-ptd', '--size_for_thread_detection', default='(400, 400)')
    parser.add_argument('-md', '--max_degree_correction', default=5)
    parser.add_argument('-show', '--show_images', default=False)
    parser.add_argument('-b', '--filter_block_sizes', default='(3, 91)')
    parser.add_argument('-c', '--filter_constant', default='(6, 11)')
    parser.add_argument('-n', '--unet_model_path', default=None)
    parser.add_argument('-nr', '--unet_resolution', default='(256, 256)')
    parser.add_argument('-nm', '--unet_margin', default='(100, 100, 100, 100)')
    parser.add_argument('-nmt', '--unet_mask_threshold', default='(0, 1)')
    parser.add_argument('-jq', '--jpg_quality', default=100)

    return parser.parse_args()

def main():
    args = parse_arguments()

    input = args.input
    output = args.output

    file_size = int(args.file_size) if args.file_size else None
    jpg_quality = int(args.jpg_quality)
    file_resolution = ast.literal_eval(args.file_resolution)
    margin = ast.literal_eval(args.margin)
    background_color = ast.literal_eval(args.background_color)
    max_degree = int(args.max_degree_correction)
    size_for_thread_detection = ast.literal_eval(args.size_for_thread_detection)
    show = args.show_images

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
            output_path = os.path.join(output, input_path.replace('../', ''))

            if not os.path.exists(output_path):
                os.makedirs(output_path)
        else:
            for path, subdirs, files in os.walk(input):
                output_path = os.path.join(output, path.replace('../', ''))

                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                for name in files:
                    img_lst.append(os.path.join(path, name))
    else:
        print('The input does not exist:', input)
        sys.exit(0)

    if model_path is not None:
        from keras.models import load_model

        try:
            model = load_model(model_path)
        except OSError as e:
            print('Error when loading the model:', e)
            sys.exit(0)
    else:
        model = None

    for img_filename in img_lst:
        print('\nProcessing the image:', img_filename)

        try:
            source = cv.imread(img_filename, 1)
            output_filename = os.path.join(output, img_filename.replace('../', ''))

            if model is None:
                print('* Re-orientation of the garment.')
                reoriented = tools.garment_reorientation(
                    source,
                    size_for_thread_detection,
                    max_degree,
                    background_color
                )

                print('* Detection and removal of background.')
                cleaned = tools.background_removal(reoriented, background_color, B_values, C_values)

                print('* Centering and zooming of the garment.')
                image = tools.crop_garment(cleaned)
            else:
                print('* Initial detection and removal of background.')
                cleaned = tools.background_removal_v2(source, background_color)

                print('* Re-orientation of the garment.')
                source_reoriented, cleaned_reoriented = tools.garment_reorientation_v2(
                    source,
                    cleaned,
                    size_for_thread_detection,
                    max_degree,
                    background_color
                )

                print('* Initia centering and zooming of the garment.')
                _, reoriented_cropped = tools.crop_garment(cleaned_reoriented, source_reoriented, unet_margin)

                print('* Detection and removal of background by unet.')
                image = tools.unet_background_removal(reoriented_cropped, model, unet_input_resolution)

                print('* Fixing color of the image before removing the background.')
                image = tools.rescale_intensity(image, 25, 240)

                image = tools.apply_mask_background_removal(image, background_color, unet_mask_threshold)

                print('* Final centering and zooming of the garment.')
                image = tools.crop_garment(image)

            print('* Resizing the final image.')
            image = tools.image_resize(image, margin, file_resolution, background_color)

            if file_size:
                print('* Writting the image to a JPG file with a maximum size of %s bytes.' % file_size)
                tools.image_write(output_filename, image, file_size=file_size)
            else:
                print('* Writting the image to a JPG file with %s %% compression quality.' % jpg_quality)
                tools.image_write(output_filename, image, jpg_quality=jpg_quality)

            print('* The processed image has been saved as:', output_filename)

            if show:
                cv.imshow('Input', cv.resize(
                    source,
                    (int(0.25 * source.shape[1]), int(0.25 * source.shape[0])),
                    interpolation=cv.INTER_CUBIC)
                )

                cv.imshow('Output', cv.resize(
                    image,
                    (int(0.75 * image.shape[1]), int(0.75 * image.shape[0])),
                    interpolation=cv.INTER_CUBIC)
                )

                key = cv.waitKey(30)
                if key == 27:
                    break
        except cv.error as e:
            print('* The image {} could not be processed.'.format(img_filename))
    if show:
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
