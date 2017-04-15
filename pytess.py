import numpy
import pytesseract
from matplotlib import pyplot as plt

import cv2

try:
	import Image
except ImportError:
	import PIL.Image


# reading the image
img = cv2.imread('test samples/1.jpg')

#converting image to greyscale 
grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#applying otsu's thresholding method after Gaussian blur
blur_image = cv2.GaussianBlur(grey_image, (11, 11), 0)

#preparation and application of sobel edge
ddepth = cv2.CV_16S
kw = dict(ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

# Gradient-X.
grad_x = cv2.Sobel(blur_image, ddepth, 1, 0, **kw)

# Gradient-Y.
grad_y = cv2.Sobel(blur_image, ddepth, 0, 1, **kw)

# Converting back to uint8.
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
sobel_no_blend = cv2.add(abs_grad_x, abs_grad_y)


#finding image graidents for edge detection
#edge_image = cv2.Canny(blur_image, 250, 100)

#using otsu's agorithm to perform binarization
retVal,thresh_image = cv2.threshold(sobel_no_blend, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#connected component analysis on the thresh_image
thresh_image_copy = thresh_image.copy()
contours, hierarchy = cv2.findContours(thresh_image_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#looping through contours to find poosible plates
long_plates = []
short_plates = []
full_set_plates = []
for contour in contours:
    [x, y, width, height] = cv2.boundingRect(contour)
    
    #filtering contours for possible plates
    if (height >  30 and width > 50) and height<120 and width < 300:
        #filtering for short and long plates with aspect_ration
        aspect_ratio = width/height
        if aspect_ratio >= 1.5 and aspect_ratio <= 3:
            possible_candidate = blur_image[y:y+height, x:x+width]
            short_plates.append(possible_candidate)
            cv2.rectangle(img, (x,y), (x+width, y+height), (0, 255, 0), 2) #drawing rectange around possible_candidate
        elif aspect_ratio >= 3.5 and aspect_ratio <=4.5:
            possible_candidate = blur_image[y:y+height, x:x+width]
            long_plates.append(possible_candidate)
            cv2.rectangle(img, (x,y), (x+width, y+height), (0, 255, 0), 2) #drawing rectange around possible_candidate

full_set_plates += long_plates
full_set_plates += short_plates

'''
for i in range(0, len(full_set_plates)):
    cv2.imshow(str(i) + '.png', full_set_plates[i])
'''

# Candidate analysis on the full_set_plates
strong_plates = []
fuzzy_plates = []
for candidate in full_set_plates: 
    candidate_edge = cv2.Canny(candidate, 250, 100)
    cand_h, cand_w = candidate.shape
    plate_candidae_copy = candidate_edge.copy()

    ## perform connected component analysis on plate_candidate
    contours2, hierrachy2 = cv2.findContours(plate_candidae_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
    chars_count = 0
    for contour in contours2:
        [w, y, w, h] = cv2.boundingRect(contour)

        ## the apsect ration analysis to check for possible characters
        character_aspect_ratio = w/h
        high_index = 1.2
        low_index = 3.5
        if (h>(0.4*cand_h) and h < cand_h): #  h>(cand_h/low_index) and h<(cand_h/high_index) and width < (cand_w/3):
            #character_aspect_ratio > 0.4 and character_aspect_ratio < 1.5:
            chars_count += 1
                
    if chars_count >= 5:
        strong_plates.append(candidate)
    elif chars_count >= 2 and chars_count <= 4:
        fuzzy_plates.append(candidate)

print("Strong: {}".format(len(strong_plates)))
print("Fuzzy: {}".format(len(fuzzy_plates)))

# Strong and Fuzzy plate analysis to get the best candidate
for i in range(0, len(strong_plates)):
    cv2.imshow(str(i) + '.png', strong_plates[i])
#assuming the best best candidate is the only plate in the strog plates, process this way


#displaying various forms of images
cv2.imshow('blur Image', blur_image)
#cv2.imshow('canny Image', edge_image)
cv2.imshow('grey Image', grey_image)
cv2.imshow('sobel', sobel_no_blend)
cv2.imshow('thresh_image', thresh_image)
cv2.imshow('original', img)

cv2.waitKey(0)
cv2.destroyAllWindows();