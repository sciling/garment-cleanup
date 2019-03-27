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

if Path(input).exists():
    img_lst = [os.path.basename(input)] if Path(input).is_file() else sorted(os.listdir(input))
    input_dir = os.path.dirname(input)
else:
    print("The input does not exist:", input)
    sys.exit(0)

def garment_reorientation(img, size_for_thread_detection, max_degree=5, background_color=[0, 0, 0]):
    '''
    Reorientation of the garment by detecting the angle of the thread from which it hangs.
    '''
    
    # Correct the illumination to increase the contrast
    new_img = illumination_correction(img)

    # Get the size of the image portion for the thread detection
    (w, h) = size_for_thread_detection
    
    # Extract the portion of image for thread detection
    portion = new_img[0:h, int(img.shape[1]/2 - w/2):int(img.shape[1]/2 + w/2)]
    
    # Get the gray thresholds
    gray = cv.cvtColor(portion, cv.COLOR_BGR2GRAY)

    # Remove noise in the gray image
    gray = cv.fastNlMeansDenoising(gray, None, 10, 11, 21)
    threshed = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 233, 3)

    # Apply a closing morphological transformation, i.e. dilation followed by erosion.
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

    # Find the thread contour
    cnts, hierarchy = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # If the we found a contour in the search area
    if len(cnts) > 0:
        cnt = sorted(cnts, key=cv.contourArea)[-1]

        # Fit the contour to a line
        [vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)

        # Get the angle of the line
        angle = math.degrees(math.atan2(vy, vx))
    
    # Otherwise, we search the rotated rectangle in the whole image
    else:
        # Get the gray thresholds
        gray = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)
        threshed = cv.adaptiveThreshold(gray, 245, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 233, 3)

        # Apply a closing morphological transformation, i.e. dilation followed by erosion.
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
        morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

        # Find the thread contour
        cnts, hierarchy = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if len(cnts) > 0:
            cnt = sorted(cnts, key=cv.contourArea)[-1]
            # Get the rotated Rectangle
            rect = cv.minAreaRect(cnt)
            center, size, angle = rect
        # Else return the input image
        else:
            return img

    # Rotate the image if the detected angle is less than the maximum
    if 90 - abs(angle) < max_degree:
        angle = -(90 - angle) if angle > 0 else 90 + angle
        rows, cols = img.shape[:2]
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img = cv.warpAffine(img, M, (cols, rows), borderValue=background_color)

    # To prevent the rotated upper edge from introducing distortion in the following steps, we cut the upper portion indicated for the detection of the thread
    return img[h:img.shape[0], 0:img.shape[1]]

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

def background_removal(img, background_color=[0,0,0], B_values=(3,91), C_values=(6,11)):
    """
    Detects the garment in the image and sets the color of the background.
    """

    new_img = cv.fastNlMeansDenoisingColored(img,None,4,4,7,21)   
    # Correct the illumination to increase contrast
    new_img = illumination_correction(new_img)

    gray = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)
    
    # Apply noise filter to the gray image
    gray = cv.fastNlMeansDenoising(gray, None, 6, 7, 21)
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    gray = cv.bilateralFilter(gray, 4, 7, 21)

    # Adjust the adaptiveThreshold to the mean value of the gray image
    #mean = np.uint8(cv.mean(gray))[0]
    
    #if mean <= 210:
        #B_values = (3,91)
        #C_values = (6,11)
    #elif 210 < mean <= 220:
        #B_values = (11,51)
        #C_values = (9,11)
    #elif 221 <= mean < 230:
        #B_values = (3,91)
        #C_values = (6,11)
    #elif 230 <= mean < 240:
        #B_values = (79,150)
        #C_values = (5,11)
    #else:
        #B_values = (91,150)
        #C_values = (4,8)
    
    #print(mean)
    #print(B_values)
    #print(C_values)

    th =  np.zeros(gray.shape, gray.dtype)
    for B in range(B_values[0], B_values[1]): 
        if B % 2 != 0: 
            for C in range(C_values[0], C_values[1]):
                thb = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, B, C)
                th = cv.addWeighted(th, 1, thb, 1/(len(B_values)*len(C_values)), 0)
            
    # To remove shadows
    _,th = cv.threshold(th,60,255,cv.THRESH_BINARY)
    
    # floodFill from the four corners for differenciate in very clear garment
    h, w = th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(th, mask, (0, 0), 0)
    cv.floodFill(th, mask, (0, h-1), 0)
    cv.floodFill(th, mask, (w-1, 0), 0)
    cv.floodFill(th, mask, (w-1, h-1), 0)
    
    # Apply noise filter to the threshold image
    blur = cv.GaussianBlur(th,(5, 5), 0)
    th = cv.addWeighted(blur, 50, th, 10, 0)
    th = cv.fastNlMeansDenoising(th, None, 4, 7, 21)
    
    # Apply a opening morphological transformation, i.e. erosion followed by dilation.
    # It is useful in removing noise
    #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    #opened = cv.morphologyEx(th, cv.MORPH_OPEN, kernel)
    
    # Apply a closing morphological transformation, i.e. dilation followed by erosion.
    # It is useful in closing small holes inside the foreground objects, or small black points on the object.
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (35, 35))
    closed = cv.morphologyEx(th, cv.MORPH_CLOSE, kernel)

    ## Find the garment contours
    cnts,_ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # From the found contours select those with more than the 50% of total area
    area = 0
    for cnt in cnts:
        area = area + cv.contourArea(cnt)

    cntsb = []
    for cnt in sorted(cnts, key=cv.contourArea, reverse=True):
        if cv.contourArea(cnt)/area > 0.5:
            cntsb.append(cnt) 

    # Create a closed mask with the selected contours
    mask = np.zeros(img.shape, img.dtype)
    cv.fillPoly(mask, cntsb, (255,)*img.shape[2], )

    height, width = mask.shape[:2]
    mask1 = cv.resize(mask, (int(0.25*width), int(0.25*height)), interpolation = cv.INTER_CUBIC)
    cv.imshow("mask", mask1)
    
    mask1 = cv.resize(th, (int(0.25*width), int(0.25*height)), interpolation = cv.INTER_CUBIC)
    cv.imshow("th", mask1)
    
    #print("White balance.")
    #img = Graywold_white_balance(img)

    # Apply the mask to extract the garment
    fg_masked = cv.bitwise_and(img, mask)
    
    # Apply the mask to colour the background
    bg = np.full(img.shape, background_color, dtype=np.uint8)
    mask = cv.bitwise_not(mask)
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

for img in img_lst:
    img='IMG_0849.JPG'
    print("\nProcessing the image:", img)
    
    source = cv.imread(os.path.join(input_dir, img), 1)

    print("* Re-orientation of the garment.")
    reoriented = garment_reorientation(source, size_for_thread_detection, max_degree, background_color)
    
    print("* Detection and removal of background.")
    cleaned = background_removal(reoriented, background_color)
    
    print("* Centering and zooming of the garment.")
    cropped = crop_garment(cleaned, margin)

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
