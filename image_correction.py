import numpy as np
import cv2 as cv
import time
import os

source_dir = "../data/38342NA3PCE/"
dest_dir = "../results/38342NA3PCE/"

file_size = 100000
final_width = 900
final_height = 1170
margin=150
background_color = (241, 241, 241)
max_degre = 10
img_list = os.listdir(source_dir)

def reorientation(img, max_degre):
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  threshed = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 111, 12)

  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
  morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)  
  
  threshed = cv.dilate(threshed, kernel)    
  morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

  cnts, hierarchy = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  cnt = sorted(cnts, key=cv.contourArea)[-1]
  
  '''
  # Rotated Rectangle
  rect = cv.minAreaRect(cnt)
  box = cv.boxPoints(rect)
  box = np.int0(box)
  cv.drawContours(img,[box],0,(0,0,255),2)
  (x,y),(width,height),angle = rect
  #cv.imshow("Gaussian", img)
  '''
  
  (x, y), (MA, ma), angle = cv.fitEllipse(cnt)
  ellipse = cv.fitEllipse(cnt)
  #cv.ellipse(src,ellipse,(0,55,55),2)

  angle = -min(angle%max_degre, max_degre - angle%max_degre) if angle > 0 else min(angle%max_degre, max_degre - angle%max_degre)
  #print(angle)
  rows,cols = src.shape[:2]

  #cv.ellipse(img,ellipse,(0,55,55),2)
  M = cv.getRotationMatrix2D((cols/2,rows/2),angle,1)
  new_img = cv.warpAffine(img, M, (cols,rows))
  
  return new_img 

def cut(img, margin=0):
  # crop image
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  #th, threshed = cv.threshold(gray, 230, 245, cv.THRESH_BINARY_INV)
  #threshed = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 91, 8)
  threshed = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 111, 12)

  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
  threshed = cv.dilate(threshed, kernel)
  morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

  cnts,_ = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  cnt = sorted(cnts, key=cv.contourArea)[-1]
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

def transBg(img):   
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  th, threshed = cv.threshold(gray, 240, 255, cv.THRESH_BINARY_INV)

  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11))
  morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

  roi, _ = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

  mask = np.zeros(img.shape, img.dtype)

  cv.fillPoly(mask, roi, (255,)*img.shape[2], )

  masked_image = cv.bitwise_and(img, mask)

  return masked_image

def colorBg(img, background_color=[0,0,0]):   
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  #th, threshed = cv.threshold(gray, 240, 255, cv.THRESH_BINARY_INV)
  threshed = cv.adaptiveThreshold(gray, 245, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 91, 12)

  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (30, 30))
  threshed = cv.dilate(threshed, kernel)
  morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

  cnts,_ = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  cnt = sorted(cnts, key=cv.contourArea)[-1]
  
  
  area = 0
  for cnt in cnts:
      area = area + cv.contourArea(cnt)

  cntsb = []
  for cnt in sorted(cnts, key=cv.contourArea, reverse=True):
      if cv.contourArea(cnt)/area > 0.5:
          cntsb.append(cnt) 

  mask = np.zeros(img.shape, img.dtype)

  cv.fillPoly(mask, cntsb, (255,)*img.shape[2], )

  masked_image = cv.bitwise_and(img, mask)
  masked_image[mask == 0] = [241]
 
  masked_image = cv.GaussianBlur(masked_image,(5,5),0)
  
  return masked_image

def fourChannels(img):
  height, width, channels = img.shape
  if channels < 4:
    new_img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
    return new_img

  return img
    
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

def balance_white(img):
    wb = cv.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(0.99)
    new_img = wb.balanceWhite(img)
    
    return new_img

def illumination_correction(img):
    lab= cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv.merge((cl,a,b))
    new_img = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    
    return new_img
    
for img in img_list:    
    src = cv.imread(os.path.join(source_dir, img), 1)
    
    print("\nCorrecting the image:", img)

    print("Automatic re-orientation of the garment, in case it is rotated in the original image")
    dst = reorientation(src, max_degre)
    
    print("Illumination correction.")
    dst = illumination_correction(dst)
    
    print("Centering and zooming of the garment.")
    dst = cut(dst, margin)
    
    print("Automatic detection and removal of background.")
    dst = colorBg(dst, background_color)
    
    print("Resizing the final image.")
    dst = image_resize(dst, final_width, final_height, background_color)
    
    print("The resulting image will have a maximum size of %s bytes"% file_size)
    image_write(os.path.join(dest_dir, str(img + ".jpg")), dst, file_size)
    
    #src = illumination_correction(src)
    #src = reorientation(src)
    
    height, width = src.shape[:2]
    src = cv.resize(src, (int(0.25*width), int(0.25*height)), interpolation = cv.INTER_CUBIC)

    height, width = dst.shape[:2]
    dst = cv.resize(dst, (int(0.75*width), int(0.75*height)), interpolation = cv.INTER_CUBIC)
    
    cv.imshow("src", src)
    #cv.imshow("src2", cv.cvtColor(src, cv.COLOR_BGR2GRAY))
    cv.imshow("dst", dst)
 
    #input("Press Enter to continue...")

    key = cv.waitKey(30)
    if key == 27:
        break
 
cv.destroyAllWindows()

