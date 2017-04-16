import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import cv2
import scipy.fftpack

try:
	import Image
except ImportError:
	import PIL.Image

'''
    IMAGE PREPROCESSING
'''
# reading the image
First_Image = cv2.imread('test samples/1.jpg')

#converting image to greyscale 
grey_image = cv2.cvtColor(First_Image, cv2.COLOR_BGR2GRAY)

#applying otsu's thresholding method after Gaussian blur
blur_image = cv2.GaussianBlur(grey_image, (5,5), 0)

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

'''
    PLATE LOCALIZATION
'''

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
            possible_candidate = grey_image[y:y+height, x:x+width]
            short_plates.append(possible_candidate)
            cv2.rectangle(First_Image, (x,y), (x+width, y+height), (0, 255, 0), 2) #drawing rectange around possible_candidate
        elif aspect_ratio >= 3.5 and aspect_ratio <=4.5:
            possible_candidate = grey_image[y:y+height, x:x+width]
            long_plates.append(possible_candidate)
            cv2.rectangle(First_Image, (x,y), (x+width, y+height), (0, 255, 0), 2) #drawing rectange around possible_candidate

full_set_plates += long_plates
full_set_plates += short_plates
                    
'''
    CANDIDATE ANALYSIS AND PLATE EXTRACTION
'''

# Candidate analysis on the full_set_plates
strong_plates = []
fuzzy_plates = []
for candidate in full_set_plates:
    blurr = cv2.GaussianBlur(candidate, (5,5),0) 
    candidate_edge = cv2.Canny(blurr, 250, 100)
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
            #if character_aspect_ratio > 0.4 and character_aspect_ratio < 1.5:
            chars_count += 1
            
    if chars_count >= 5:
        strong_plates.append(candidate)
    elif chars_count >= 2 and chars_count <= 4:
        fuzzy_plates.append(candidate)

print("Strong_plates: {}".format(len(strong_plates)))
print("Fuzzy_plates: {}".format(len(fuzzy_plates)))


# Strong and Fuzzy plate analysis to get the best candidate
for plate in strong_plates:
    #plate_threshold_image = cv2.adaptiveThreshold(plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
    p_h, p_w = plate.shape
    
    #resiszing the plate if the height and width are below a certain size
    if p_h < 74 or p_w < 285: 
        plate_to_save = cv2.resize(plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) 
    else:
        plate_to_save = plate
    
    cv2.imwrite('extracted_plate.jpg', plate_to_save)

'''
    PLATE SEGMENTATION
'''

#### imclearborder definition

def imclearborder(imgBW, radius):

    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)

    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]    

    contourList = [] # ID list of contours that touch the border

    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]

        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border
            # this contour goes bye bye!
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy

#### bwareaopen definition
def bwareaopen(imgBW, areaPixels):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy
    
#### Main program

# Read in image
img = cv2.imread('extracted_plate.jpg', 0)

# Number of rows and columns
rows = img.shape[0]
cols = img.shape[1]

# Remove some columns from the beginning and end
#img = img[:, 59:cols-20]

# Number of rows and columns
rows = img.shape[0]
cols = img.shape[1]

# Convert image to 0 to 1, then do log(1 + I)
imgLog = np.log1p(np.array(img, dtype="float") / 255)

# Create Gaussian mask of sigma = 10
M = 2*rows + 1
N = 2*cols + 1
sigma = 10
(X,Y) = np.meshgrid(np.linspace(0,N-1,N), np.linspace(0,M-1,M))
centerX = np.ceil(N/2)
centerY = np.ceil(M/2)
gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2

# Low pass and high pass filters
Hlow = np.exp(-gaussianNumerator / (2*sigma*sigma))
Hhigh = 1 - Hlow

# Move origin of filters so that it's at the top left corner to
# match with the input image
HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

# Filter the image and crop
If = scipy.fftpack.fft2(imgLog.copy(), (M,N))
Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M,N)))
Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M,N)))

# Set scaling factors and add
gamma1 = 0.3
gamma2 = 1.5
Iout = gamma1*Ioutlow[0:rows,0:cols] + gamma2*Iouthigh[0:rows,0:cols]

# Anti-log then rescale to [0,1]
Ihmf = np.expm1(Iout)
Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
Ihmf2 = np.array(255*Ihmf, dtype="uint8")

# Threshold the image - Anything below intensity 65 gets set to white
Ithresh = Ihmf2 < 65
Ithresh = 255*Ithresh.astype("uint8")

# Clear off the border.  Choose a border radius of 5 pixels
Iclear = imclearborder(Ithresh, 5)

# Eliminate regions that have areas below 120 pixels
Iopen = bwareaopen(Iclear, 120)

'''
    CHARACTER RECOGNITION
    using the tesseract ocr to detect the plate text out of the prepared image
'''



#displaying various forms of images
cv2.imshow('grey Image', grey_image)
cv2.imshow('blur Image', blur_image)
cv2.imshow('sobel', sobel_no_blend)
cv2.imshow('thresh_image', thresh_image)
cv2.imshow('original', First_Image)


# Show all plate candidate series
cv2.imshow('Original Image', img)
cv2.imshow('Homomorphic Filtered Result', Ihmf2)
cv2.imshow('Thresholded Result', Ithresh)
cv2.imshow('Opened Result', Iopen)


cv2.waitKey(0)
cv2.destroyAllWindows();