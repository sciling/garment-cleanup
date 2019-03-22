import numpy as np
import cv2 as cv
import time
import os
import math

source_dir = "../data/38342NA3PCE/"
dest_dir = "../results/38342NA3PCE/"

file_size = 100000
final_width = 900
final_height = 1170
margin=0
background_color = (241, 241, 241)
max_degre = 5
pixels_for_thread_detection = 400
img_list = os.listdir(source_dir)

subtractor = cv.createBackgroundSubtractorMOG2(history=10, varThreshold=4, detectShadows=True)


def reorientation(img, pixels_for_thread_detection, max_degree=5,  background_color=[0,0,0]):
  '''
  Reorientation of the image by detecting the angle of the thread
  '''
  
  # Correct the illumination to increase contrast
  new_img = illumination_correction(img)
  
  # Extract the portion of image for thread detection
  new_img = img[0:pixels_for_thread_detection, 0:img.shape[1]]
  gray = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)
  threshed = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 91, 8)
  
  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
  threshed = cv.dilate(threshed, kernel)    
  morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

  # Find the contours
  cnts, hierarchy = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  cnt = sorted(cnts, key=cv.contourArea)[-1]
  
  # Fit the contour to a line
  [vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
  
  '''
  rows,cols = new_img.shape[:2]
  lefty = int((-x*vy/vx) + y)
  righty = int(((cols-x)*vy/vx)+y)
  cv.line(new_img,(cols-1,righty),(0,lefty),(0,255,0),2)
  
  
  height, width = new_img.shape[:2]
  new_img = cv.resize(new_img, (int(0.5*width), int(0.5*height)), interpolation = cv.INTER_CUBIC)
  cv.imshow("new_img", new_img)
  '''
  
  # Get the angle of the line
  angle = math.degrees(math.atan2(vy,vx))
  
  if 90 - abs(angle) < max_degree:
    angle = -(90 - angle) if angle > 0 else 90 + angle
    rows,cols = src.shape[:2]
    M = cv.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img = cv.warpAffine(img, M, (cols,rows), borderValue=background_color)
  
  return img[pixels_for_thread_detection:img.shape[0], 0:img.shape[1]] 

def cut(img, margin=0):
  # Correct the illumination to increase contrast
  new_img = illumination_correction(img)
  # crop image
  gray = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)
  #th, threshed = cv.threshold(gray, 230, 245, cv.THRESH_BINARY_INV)
  #threshed = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 91, 8)
  threshed = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 19, 10)

  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
  threshed = cv.dilate(threshed, kernel)
  morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

  cnts,_ = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  cnt = sorted(cnts, key=cv.contourArea)[-1]
  
  # Rotated Rectangle
#  rect = cv.minAreaRect(cnt)
#  box = cv.boxPoints(rect)
#  box = np.int0(box)
#  cv.drawContours(img,[box],0,(0,0,255),2)
  
  
  (img_h, img_w) = img.shape[:2]
  x,y,w,h = cv.boundingRect(cnt)
  
  if x - margin >= 0:
      x = x - margin
  if y - margin >= 0:
      y = y - margin
  if x + w + margin <= img_w:
      w = w + margin
  if y + h + margin <= img_h:
      h = h + margin
      
  new_img = img[y:y+h, x:x+w]

  return new_img

def background_removal(img, background_color=[0,0,0]):
  
  # Correct the illumination to increase contrast
  new_img = illumination_correction(img)
  
  gray = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)
  th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 37, 11)
  
  # Remove noise
  th = cv.fastNlMeansDenoising(th, None, 200, 53, 3)
  
  blur = cv.GaussianBlur(th,(25,25), 0)
  threshed = cv.addWeighted(blur, 60, th, 10, 0)
    
  # Open
  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
  opened = cv.morphologyEx(threshed, cv.MORPH_OPEN, kernel)
  
  # Close
  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
  closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel)
    
  cnts,_ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
 
  #cnt = sorted(cnts, key=cv.contourArea)[-1]
  #cv.drawContours(img, cnt, -1, (127, 255, 0), 3)
  
  area = 0
  for cnt in cnts:
      area = area + cv.contourArea(cnt)

  cntsb = []
  for cnt in sorted(cnts, key=cv.contourArea, reverse=True):
      if cv.contourArea(cnt)/area > 0.5:
          cntsb.append(cnt) 

  mask = np.zeros(img.shape, img.dtype)

  cv.fillPoly(mask, cntsb, (255,)*img.shape[2], )
  
  #height, width = mask.shape[:2]
  #mask1 = cv.resize(threshed, (int(0.25*width), int(0.25*height)), interpolation = cv.INTER_CUBIC)
  #cv.imshow("threshed", mask1)
  #mask1 = cv.resize(th, (int(0.25*width), int(0.25*height)), interpolation = cv.INTER_CUBIC)
  #cv.imshow("th", mask1)
  #mask1 = cv.resize(blur, (int(0.25*width), int(0.25*height)), interpolation = cv.INTER_CUBIC)
  #cv.imshow("blur", mask1)
  #mask1 = cv.resize(opened, (int(0.25*width), int(0.25*height)), interpolation = cv.INTER_CUBIC)
  #cv.imshow("opened", mask1)
  #mask1 = cv.resize(closed, (int(0.25*width), int(0.25*height)), interpolation = cv.INTER_CUBIC)
  #cv.imshow("closed", mask1)
  
  #mask12 = cv.resize(th2, (int(0.25*width), int(0.25*height)), interpolation = cv.INTER_CUBIC)
  #cv.imshow("video", mask12)
  #mask1 = cv.resize(threshed, (int(0.25*width), int(0.25*height)), interpolation = cv.INTER_CUBIC)
  #cv.imshow("combination", mask1)
  
  #mask2 = cv.resize(mask, (int(0.25*width), int(0.25*height)), interpolation = cv.INTER_CUBIC)
  #cv.imshow("mask", mask2)
  
  
  #print("White balance.")
  #img = balance_white(img)
    
  masked_image = cv.bitwise_and(img, mask)
  masked_image[mask == 0] = [241]
 
  #masked_image = cv.GaussianBlur(masked_image,(5,5),0)
  
  return masked_image

def image_resize(image, width = None, height = None, background_color=[0,0,0], inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    elif height is None:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        r = float(h) / float(w)
        nw = int(height / r)
        nh = int(width * r)
        
        if nw > width:
            dim = (width, nh)
        else:
            dim = (nw, height)

    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)
    
    # Wrap the border
    t = int((height - dim[1])/2)
    b = int((height - dim[1])/2)
    l = int((width - dim[0])/2)
    r = int((width - dim[0])/2)
    
    # Wrap the resized image to the defined canvas size
    wrap = cv.copyMakeBorder(resized,t,b,l,r,cv.BORDER_CONSTANT, value=background_color)

    # return the wrap image
    return wrap

def image_write(file_name, img, file_size):
    quality = 100
    while True:
        cv.imwrite(file_name, img, [cv.IMWRITE_JPEG_QUALITY, quality])
        statinfo = os.stat(file_name)
        if(statinfo.st_size < file_size):
            break
        quality = quality - 1

def illumination_correction(img):
    lab= cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv.merge((cl,a,b))
    new_img = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    
    return new_img

def balance_white(img):
    
    wb = cv.xphoto.createSimpleWB()
    wb.setInputMin(0)
    wb.setInputMax(255)
    wb.setOutputMin(0)
    wb.setOutputMax(255)
    #wb.setSaturationThreshold(0.99)
    new_img = wb.balanceWhite(img)
    
    return new_img

'''
# For background removal
for img in img_list:
    src = cv.imread(os.path.join(source_dir, img))
    dst = illumination_correction(src)
    mask = subtractor.apply(dst)
'''    

for img in img_list:
    #img="IMG_0899.JPG"
    #img="IMG_0902.JPG"
    src = cv.imread(os.path.join(source_dir, img), 1)
    
    print("\nCorrecting the image:", img)
    
    #print("Illumination correction.")
    #dst = illumination_correction(src)
        
    print("Re-orientation of the garment, in case it is rotated in the original image")
    dst = reorientation(src, pixels_for_thread_detection, max_degre, background_color)
    
    print("Detection and removal of background.")
    dst = background_removal(dst, background_color)
    
    print("Centering and zooming of the garment.")
    dst = cut(dst, margin)    
    
    print("Resizing the final image.")
    dst = image_resize(dst, final_width, final_height, background_color)
    
    #print("White balance.")
    #dst = balance_white(dst)
    
    
    print("The resulting image will have a maximum size of %s bytes"% file_size)
    image_write(os.path.join(dest_dir, str(img + ".jpg")), dst, file_size)
    
    height, width = src.shape[:2]
    src = cv.resize(src, (int(0.25*width), int(0.25*height)), interpolation = cv.INTER_CUBIC)

    height, width = dst.shape[:2]
    dst = cv.resize(dst, (int(0.75*width), int(0.75*height)), interpolation = cv.INTER_CUBIC)
    
    cv.imshow("src", src)
    cv.imshow("dst", dst)
 
    #input("Press Enter to continue...")

    key = cv.waitKey(30)
    if key == 27:
        break
 
cv.destroyAllWindows()

