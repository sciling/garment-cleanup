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

import numpy as np
import cv2 as cv
import time
import os
import math
import sys
import argparse
from pathlib import Path
import ast
import scipy

def garment_reorientation(img, clean_img, size_for_thread_detection, max_degree=5, background_color=[0, 0, 0]):
    '''
    Reorientation of the garment by detecting the angle of the thread from which it hangs.
    '''
    
    # Get the size of the image portion for the thread detection
    (w, h) = size_for_thread_detection
    
    # Extract the portion of image for thread detection
    portion = img[0:h, int(img.shape[1]/2 - w/2):int(img.shape[1]/2 + w/2)]
    
    # Get the gray levels
    gray = cv.cvtColor(portion, cv.COLOR_BGR2GRAY)
    
    sobelx = cv.Sobel(gray,cv.CV_64F,1,0,ksize=5)
    sobely = cv.Sobel(gray,cv.CV_64F,0,1,ksize=5)
    abs_sobel = np.absolute(np.power((np.power(sobelx, 2) + np.power(sobely, 2)), 1/2))
    sobel = np.uint8(abs_sobel)
    
    _, threshed = cv.threshold(sobel,127,255,cv.THRESH_BINARY_INV)

    # Find the thread contour
    cnts, hierarchy = cv.findContours(threshed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # If the we found a contour in the search area
    if len(cnts) > 0:
        cnt = sorted(cnts, key=cv.contourArea)[-1]

        # Fit the contour to a line
        [vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)

        # Get the angle of the line
        angle = math.degrees(math.atan2(vy, vx))
    
    # Otherwise, we search the rotated rectangle in the whole image
    else:
        # Get the gray levels
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        sobelx = cv.Sobel(gray,cv.CV_64F,1,0,ksize=5)
        sobely = cv.Sobel(gray,cv.CV_64F,0,1,ksize=5)
        abs_sobel = np.absolute(np.power((np.power(sobelx, 2) + np.power(sobely, 2)), 1/2))
        sobel = np.uint8(abs_sobel)
        
        _, threshed = cv.threshold(sobel,127,255,cv.THRESH_BINARY_INV)

        # Find the garment contour
        cnts, hierarchy = cv.findContours(threshed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if len(cnts) > 0:
            cnt = sorted(cnts, key=cv.contourArea)[-1]
            # Get the rotated Rectangle
            rect = cv.minAreaRect(cnt)
            center, size, angle = rect
        # Else return the input image
        else:
            return clean_img
    
    # Rotate the image if the detected angle is less than the maximum
    if 90 - abs(angle) < max_degree:
        angle = -(90 - angle) if angle > 0 else 90 + angle
        rows, cols = img.shape[:2]
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv.warpAffine(clean_img, M, (cols, rows), borderValue=background_color)
    else:
        return clean_img

def crop_garment(img, margin=0):
    """
    Detects the garment in the image and crop it.
    """
    
    # Correct the illumination to increase the contrast
    new_img = illumination_correction(img)
    
    # Get the gray thresholds
    gray = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)
    threshed = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 19, 8)

    # Find the garment contour and get the bounding rectangle
    cnts, _ = cv.findContours(threshed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if len(cnts) > 0:
        x, y, w, h = cv.boundingRect(np.concatenate(cnts))
        
        # Add the margin pixels
        (img_h, img_w) = img.shape[:2]

        if x - margin >= 0:
            x = x - margin
        if y - margin >= 0:
            y = y - margin
        if x + w + margin <= img_w:
            w = w + margin
        if y + h + margin <= img_h:
            h = h + margin

        # Crop the garment
        cropped = img[y:y + h, x:x + w]

        return cropped
    else:
        return img

def background_removal(img, background_color=[0,0,0], B_values=(77,91), C_values=(6,11)):
    """
    Detects the garment in the image and sets the color of the background.
    """

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    sobelx = cv.Sobel(gray,cv.CV_64F,1,0,ksize=5)
    sobely = cv.Sobel(gray,cv.CV_64F,0,1,ksize=5)
    abs_sobel = np.absolute(np.power((np.power(sobelx, 2) + np.power(sobely, 2)), 1/2))
    sobel = np.uint8(abs_sobel)
    
    # Apply a closing morphological transformation, i.e. dilation followed by erosion.
    # It is useful in closing small holes inside the foreground objects, or small black points on the object.
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    closed = cv.morphologyEx(sobel, cv.MORPH_CLOSE, kernel)
    
    ret,thresh2 = cv.threshold(closed,127,255,cv.THRESH_BINARY)

    ## Find the garment contours
    cnts,_ = cv.findContours(thresh2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # From the found contours select those with more than the 50% of total area
    area = 0
    for cnt in cnts:
        area = area + cv.contourArea(cnt)

    cntsb = []
    for cnt in sorted(cnts, key=cv.contourArea, reverse=True):
        
        if cv.contourArea(cnt)/area > 0.2:
            cntsb.append(cnt) 

    # Create a closed mask with the selected contours
    mask = np.zeros(img.shape, img.dtype)
    cv.fillPoly(mask, cntsb, (255,)*img.shape[2], )
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask2 = cv.erode(mask,kernel,iterations = 2)
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (35, 35))
    mask2 = cv.morphologyEx(mask2, cv.MORPH_CLOSE, kernel)
    
    blur = cv.GaussianBlur(mask2,(5,5),0)
    mask2 = cv.addWeighted(blur,1.5,mask2,-0.5,0)
    
    # Apply the mask to extract the garment
    fg_masked = cv.bitwise_and(img, mask2)
    
    # Apply the mask to colour the background
    bg = np.full(img.shape, background_color, dtype=np.uint8)
    mask = cv.bitwise_not(mask2)
    bg_masked = cv.bitwise_and(bg, mask)

    # Combine the extracted garment with the colored background
    masked = cv.bitwise_or(fg_masked, bg_masked)
    
    return masked

def image_resize(img, file_resolution=None, background_color=[0, 0, 0], inter=cv.INTER_AREA):
    """
    Resize the image to a desired resolution.
    """
    
    # Get the width and the height of the final image
    (width, height) = file_resolution if file_resolution is not None else (None, None)
    
    # Initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = img.shape[:2]

    # If both the width and height are None, then return the original image
    if width is None and height is None:
        return img

    # Check if the width is None
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # Check if the height is None
    elif height is None:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))
        
    # Otherwise, both (width and heigth) are not None
    else:
        # Calculate the ratio of the original image
        r = float(h) / float(w)
        
        # With the original ratio, get the resized width and height 
        nw = int(height / r)
        nh = int(width * r)

        # The final dimensions are determined by the width and height limits
        if nw > width:
            dim = (width, nh)
        else:
            dim = (nw, height)

    # Resize the image
    resized = cv.resize(img, dim, interpolation=inter)

    # Wrap the border
    t = int((height - dim[1]) / 2)
    b = int((height - dim[1]) / 2)
    l = int((width - dim[0]) / 2)
    r = int((width - dim[0]) / 2)

    # Wrap the resized image to the defined canvas size and fill with the background color
    wrapped = cv.copyMakeBorder(resized, t, b, l, r, cv.BORDER_CONSTANT, value=background_color)

    # Return the wrapped image
    return wrapped

def image_write(file_name, img, file_size):
    """
    Writes the image to a JPG file with a size smaller than the determined by 'file_size'.
    Size adjustment is achieved by adjusting the quality of the JPG compression.
    """
    
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    quality = 100
    while True:
        cv.imwrite(file_name, img, [cv.IMWRITE_JPEG_QUALITY, quality])
        statinfo = os.stat(file_name)
        if (statinfo.st_size < file_size):
            break
        quality = quality - 1

def illumination_correction(img):
    """
    Perform an illumniation correction based on CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """
    
    # Convert the RGB image to Lab color-space
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    
    # Apply adaptive histogram equalization to the L channel
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Convert the resulting Lab back to RGB
    limg = cv.merge((cl, a, b))
    new_img = cv.cvtColor(limg, cv.COLOR_LAB2BGR)

    return new_img

def Graywold_white_balance(img, saturation=0.99):
    """
    Perform a white balance based on the the gray world algorithm which is based on an assumption that the average reflectance in the scene with rich color changes is achromatic.
    """
    
    wb = cv.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(saturation)
    new_img = wb.balanceWhite(img)
    
    return new_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images.')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-fs', '--file_size', default=100000)
    parser.add_argument('-fr', '--file_resolution', default='(900, 1170)')
    parser.add_argument('-m', '--margin', default=0)
    parser.add_argument('-bc', '--background_color', default='[241, 241, 241]')
    parser.add_argument('-ptd', '--size_for_thread_detection', default='(400, 400)')
    parser.add_argument('-md', '--max_degree_correction', default=5)
    parser.add_argument('-show', '--show_images', default=False)
    parser.add_argument('-b', '--filter_block_sizes', default='(3, 91)')
    parser.add_argument('-c', '--filter_constant', default='(6, 11)')

    args = parser.parse_args()

    input = args.input
    output = args.output

    file_size = int(args.file_size)
    file_resolution = ast.literal_eval(args.file_resolution)
    margin = int(args.margin)
    background_color = ast.literal_eval(args.background_color)
    max_degree = int(args.max_degree_correction)
    size_for_thread_detection = ast.literal_eval(args.size_for_thread_detection)
    show = args.show_images
    B_values = ast.literal_eval(args.filter_block_sizes)
    C_values = ast.literal_eval(args.filter_constant)

    if Path(input).exists():
        img_lst = [os.path.basename(input)] if Path(input).is_file() else sorted(os.listdir(input))
        input_dir = os.path.dirname(input)
    else:
        print("The input does not exist:", input)
        sys.exit(0)

    for img in img_lst:        
        print("\nProcessing the image:", img)
        
        source = cv.imread(os.path.join(input_dir, img), 1)

        print("* Detection and removal of background.")
        cleaned = background_removal(source, background_color, B_values, C_values)
        
        print("* Re-orientation of the garment.")
        reoriented = garment_reorientation(source, cleaned, size_for_thread_detection, max_degree, background_color)
        
        print("* Centering and zooming of the garment.")
        cropped = crop_garment(reoriented, margin)

        print("* Resizing the final image.")
        resized = image_resize(cropped, file_resolution, background_color)

        print("* Writting the image to a JPG file with a maximum size of %s bytes." % file_size)
        image_write(os.path.join(output, str(os.path.splitext(img)[0] + ".jpg")), resized, file_size)

        if show:
            height, width = source.shape[:2]
            source = cv.resize(source, (int(0.25 * width), int(0.25 * height)), interpolation=cv.INTER_CUBIC)

            height, width = resized.shape[:2]
            resized = cv.resize(resized, (int(0.75 * width), int(0.75 * height)), interpolation=cv.INTER_CUBIC)

            cv.imshow("Input", source)
            cv.imshow("Output", resized)
            
            key = cv.waitKey(30)
            if key == 27:
                break

    if show:
        cv.destroyAllWindows()
