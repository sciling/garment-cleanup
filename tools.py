#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This file contains all the methods required to perform the image processing.

Created on Fri Jun 7 13:12:13 2019

Author: Emilio Granell <egranell@sciling.com>

(c) 2019 Sciling, SL
'''

import numpy as np
import cv2 as cv
import os
import math

def garment_reorientation(img, size_for_thread_detection=(400, 400), max_degree=5, background_color=[0, 0, 0]):
    '''
    Reorientation of the garment by detecting the angle of the thread from which it hangs.
    '''

    # Correct the illumination to increase the contrast
#    new_img = illumination_correction(img)
    new_img = img

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

    return img

def garment_reorientation_v2(img, img2=None, size_for_thread_detection=(400, 400), max_degree=5, background_color=[0, 0, 0]):
    '''
    Reorientation of the garment by detecting the angle of the thread from which it hangs.
    This method is based on Sobel edge detection algorithm. The detection is preformed on the first input image
    but the transformation is applied to the two images.
    This method allows to obtain a orientation corrected image after the background removal.
    '''

    # Get the size of the image portion for the thread detection
    (w, h) = size_for_thread_detection

    # Extract the portion of image for thread detection
    portion = img[0:h, int(img.shape[1]/2 - w/2):int(img.shape[1]/2 + w/2)].copy()

    # Get the gray levels
    gray = cv.cvtColor(portion, cv.COLOR_BGR2GRAY)

    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
    abs_sobel = np.absolute(np.sqrt((np.power(sobelx, 2) + np.power(sobely, 2))))
    sobel = np.uint8(abs_sobel)

    _, threshed = cv.threshold(sobel, 127, 255, cv.THRESH_BINARY_INV)

    # Find the thread contour
    cnts, hierarchy = cv.findContours(threshed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    angle = 0
    # If the we found a contour in the search area
    if len(cnts) > 0:
        cnt = sorted(cnts, key=cv.contourArea)[-1]

        # Fit the contour to a line
        [vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)

        # Get the angle of the line
        angle = math.degrees(math.atan2(vy, vx))

    # If the we don't get a valid angle, we search the rotated rectangle in the whole image
    if angle == 0:
        # Get the gray levels
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
        sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
        abs_sobel = np.absolute(np.sqrt((np.power(sobelx, 2) + np.power(sobely, 2))))
        sobel = np.uint8(abs_sobel)

        _, threshed = cv.threshold(sobel, 127, 255, cv.THRESH_BINARY)

        # Find the garment contour
        cnts, hierarchy = cv.findContours(threshed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if len(cnts) > 0:
            cnt = sorted(cnts, key=cv.contourArea)[-1]
            # Get the rotated Rectangle
            rect = cv.minAreaRect(cnt)
            center, size, angle = rect

    # Rotate the image if the detected angle is less than the maximum
    if 90 - abs(angle) < max_degree:
        angle = -(90 - angle) if angle > 0 else 90 + angle
        rows, cols = img.shape[:2]
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

        if img2 is None:
            return cv.warpAffine(img, M, (cols, rows), borderValue=background_color)
        else:
            return cv.warpAffine(img, M, (cols, rows), borderValue=background_color), cv.warpAffine(img2, M, (cols, rows), borderValue=background_color)
    else:
        if img2 is None:
            return img
        else:
            return img, img2

def crop_garment(img, img2=None, margin=(0, 0, 0, 0)):
    '''
    Detects the garment in the image and crop it.
    '''

    new_img = img.copy()
    new_img2 = img2.copy() if img2 is not None else None

    # Get the gray thresholds
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    threshed = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 19, 8)

    # Find the garment contour and get the bounding rectangle
    cnts, _ = cv.findContours(threshed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:
        x, y, w, h = cv.boundingRect(np.concatenate(cnts))

        # Give some margin for the unet to better detect the background
        i_h, i_w = new_img.shape[:2]

        x = x - margin[0] if x > margin[0] else 0
        y = y - margin[1] if y > margin[1] else 0
        w = w + (margin[0] + margin[2]) if (x + w) + (margin[0] + margin[2]) < i_w else i_w
        h = h + (margin[1] + margin[3]) if (y + h) + (margin[1] + margin[3]) < i_h else i_h

        # Crop the garment
        new_img = new_img[y:y + h, x:x + w]

        if new_img2 is not None:
            new_img2 = new_img2[y:y + h, x:x + w]

    if new_img2 is not None:
        return new_img, new_img2
    else:
        return new_img


def background_removal(img, background_color=[0, 0, 0], B_values=(3, 91), C_values=(6, 11)):
    '''
    Detects the garment in the image and sets the color of the background.
    '''

    new_img = img.copy()
    new_img = cv.fastNlMeansDenoisingColored(new_img, None, 4, 4, 7, 21)
    # Correct the illumination to increase contrast
#    new_img = illumination_correction(new_img)

    gray = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)

    # Apply noise filter to the gray image
    gray = cv.fastNlMeansDenoising(gray, None, 4, 7, 21)
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    gray = cv.bilateralFilter(gray, 4, 7, 21)

    th = np.zeros(gray.shape, gray.dtype)
    for B in range(B_values[0], B_values[1]):
        if B % 2 != 0:
            for C in range(C_values[0], C_values[1]):
                thb = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, B, C)
                th = cv.addWeighted(th, 1, thb, 1/(len(B_values)*len(C_values)), 0)

    # To remove shadows
    _, th = cv.threshold(th, 60, 255, cv.THRESH_BINARY)

    # floodFill from the four corners for differenciate in very clear garment
    h, w = th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(th, mask, (0, 0), 0)
    cv.floodFill(th, mask, (0, h-1), 0)
    cv.floodFill(th, mask, (w-1, 0), 0)
    cv.floodFill(th, mask, (w-1, h-1), 0)

    # Apply noise filter to the threshold image
    blur = cv.GaussianBlur(th, (5, 5), 0)
    th = cv.addWeighted(blur, 50, th, 10, 0)
    th = cv.fastNlMeansDenoising(th, None, 4, 7, 21)

    # Apply a closing morphological transformation, i.e. dilation followed by erosion.
    # It is useful in closing small holes inside the foreground objects, or small black points on the object.
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (35, 35))
    closed = cv.morphologyEx(th, cv.MORPH_CLOSE, kernel)

    # Find the garment contours
    cnts, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

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

    # Apply a ilumination correction
#    img = illumination_correction(img)

    # Apply the mask to extract the garment
    fg_masked = cv.bitwise_and(img, mask)

    # Apply the mask to colour the background
    bg = np.full(img.shape, background_color, dtype=np.uint8)
    mask = cv.bitwise_not(mask)
    bg_masked = cv.bitwise_and(bg, mask)

    # Combine the extracted garment with the colored background
    masked = cv.bitwise_or(fg_masked, bg_masked)

    return masked

def background_removal_v2(img, background_color=[0, 0, 0]):
    '''
    Detects the garment in the image and sets the color of the background based on Sobel edge detection algorithm.
    '''

    new_img = img.copy()
    gray = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)

    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
    abs_sobel = np.absolute(np.sqrt((np.power(sobelx, 2) + np.power(sobely, 2))))
    sobel = np.uint8(abs_sobel)

    # Apply a closing morphological transformation, i.e. dilation followed by erosion.
    # It is useful in closing small holes inside the foreground objects, or small black points on the object.
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    closed = cv.morphologyEx(sobel, cv.MORPH_CLOSE, kernel)

    ret, thresh2 = cv.threshold(closed, 127, 255, cv.THRESH_BINARY)

    # Find the garment contours
    cnts, _ = cv.findContours(thresh2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # From the found contours select those with more than the 50% of total area
    area = 0
    for cnt in cnts:
        area = area + cv.contourArea(cnt)

    cntsb = []
    for cnt in sorted(cnts, key=cv.contourArea, reverse=True):
        if cv.contourArea(cnt)/area > 0.2:
            cntsb.append(cnt)

    # Create a closed mask with the selected contours
    mask = np.zeros(new_img.shape, new_img.dtype)
    cv.fillPoly(mask, cntsb, (255,)*new_img.shape[2], )

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask2 = cv.erode(mask, kernel, iterations=2)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (35, 35))
    mask2 = cv.morphologyEx(mask2, cv.MORPH_CLOSE, kernel)

    blur = cv.GaussianBlur(mask2, (5, 5), 0)
    mask2 = cv.addWeighted(blur, 1.5, mask2, -0.5, 0)

    # Apply a ilumination correction
#    new_img = illumination_correction(new_img)

    # Apply the mask to extract the garment
    fg_masked = cv.bitwise_and(new_img, mask2)

    # Apply the mask to colour the background
    bg = np.full(new_img.shape, background_color, dtype=np.uint8)
    mask = cv.bitwise_not(mask2)
    bg_masked = cv.bitwise_and(bg, mask)

    # Combine the extracted garment with the colored background
    masked = cv.bitwise_or(fg_masked, bg_masked)

    return masked

def image_resize(img, margin=None, file_resolution=None, background_color=[0, 0, 0], inter=cv.INTER_AREA):
    '''
    Resize the image to a desired resolution.
    '''

    # Get the four margins
    (top, bottom, left, right) = margin if margin is not None else (0, 0, 0, 0)

    # Get the width and the height of the final image
    (width, height) = file_resolution if file_resolution is not None else (None, None)

    # Set the width and the height without margins
    width = width - left - right
    height = height - top - bottom

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

    # Add border
    img_border = cv.copyMakeBorder(
        wrapped,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv.BORDER_CONSTANT,
        value=background_color
    )

    # Return the wrapped image
    return img_border

def image_write(file_name, img, file_size=None, jpg_quality=None):
    '''
    Writes the image to a JPG file with a size smaller than the determined by 'file_size'.
    Size adjustment is achieved by adjusting the quality of the JPG compression.
    '''

    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    quality = jpg_quality if jpg_quality is not None else 100
    while True:
        cv.imwrite(file_name, img, [cv.IMWRITE_JPEG_QUALITY, quality])
        statinfo = os.stat(file_name)
        if not file_size or (statinfo.st_size < file_size) or jpg_quality:
            break
        quality = quality - 1

def illumination_correction(img):
    '''
    Perform an illumniation correction based on CLAHE (Contrast Limited Adaptive Histogram Equalization).
    '''

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
    '''
    Perform a white balance based on the the gray world algorithm which is based on
    an assumption that the average reflectance in the scene with rich color changes is achromatic.
    '''

    wb = cv.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(saturation)
    new_img = wb.balanceWhite(img)

    return new_img

def unet_background_removal(img, model, unet_input_resolution):
    '''
    Removes the background of an RGB input image by using a unet model.
    Returns the same input image with the predicted alpha channel.
    '''

    #import skimage.transform as trans
    from scipy.ndimage import gaussian_filter

    source_size = img.shape[:2]
    new_img = img.copy()
    new_img = new_img[:, :, ::-1]
    new_img = new_img / 255

    # INTER_NEAREST, INTER_LINEAR, INER_AREA, INTER_CUBIC, INTER_LANCZOS4
    if unet_input_resolution is not (None, None):
        #new_img = trans.resize(new_img, unet_input_resolution)
        factors = (np.asarray(new_img.shape , dtype=float) /
                   np.asarray(tuple(unet_input_resolution) + (new_img.shape[-1], ), dtype=float))
        sigma = np.maximum(0, (factors-1) / 2)
        #new_img = cv.GaussianBlur(new_img, ksize=(0,0), sigmaX=sigma[0], sigmaY=sigma[1])
        new_img = gaussian_filter(new_img, sigma, mode='mirror', cval=0)
        new_img = cv.resize(new_img, unet_input_resolution, interpolation=cv.INTER_CUBIC)

    new_img = np.reshape(new_img, (1,)+new_img.shape)

    alpha = model.predict(new_img)[0]
    if unet_input_resolution is not (None, None):
        #alpha = trans.resize(alpha, source_size)
        alpha = cv.resize(alpha, (img.shape[1], img.shape[0]), interpolation=cv.INTER_LINEAR)
        alpha = np.reshape(alpha, alpha.shape+(1,))

    alpha = (255 * alpha)
    new_img = np.concatenate((img, alpha), axis=2)

    return new_img.astype(np.uint8)

def apply_mask_background_removal(img, background_color=[0, 0, 0], threshold=(0, 1)):
    '''
    Apply a background color to the RGBA input image according to the alpha channel and the defined thresholds.
    Returns a RGB image.
    '''
    image = img.copy()

    background_color_t = background_color.copy()
    background_color_t.append(255)

    background = np.full(image.shape, background_color_t)
    background = background / 255

    mask = np.stack([image[:, :, 3] / 255 for _ in range(3)], axis=2)
    mask[mask < threshold[0]] = 0
    mask[mask > threshold[1]] = 1

    inv_mask = 1. - mask

    # Apply a ilumination correction
    # image = illumination_correction(image)
    image = image / 255

    result = background[:, :, :3] * inv_mask + image[:, :, :3] * mask

    return (result * 255).astype(np.uint8)

def rescale_intensity(img, r1, r2):
    '''
    Return image after stretching or shrinking its intensity levels.
    '''

    # Clip the image to the specified range, renormalize ranges, and rescale.
    # Code adapted from Skimage to reduce overhead
    img_rescaled = np.clip(img[:, :, :3], r1, r2)
    img_rescaled = (img_rescaled - r1) / float(r2 - r1)
    img_rescaled = np.array(img_rescaled * 255, dtype=img.dtype.type)
    # Add one dimension (will be the alpha channel).
    img_rescaled = np.dstack((img_rescaled, np.zeros(img_rescaled.shape[:2]))).astype(np.uint8)
    # Restore alpha from original image.
    img_rescaled[:, :, 3] = img[:, :, 3]

    return img_rescaled
