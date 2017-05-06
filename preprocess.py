import cv2
import numpy as np

'''
    IMAGE PREPROCESSING
'''
def image_preprocessing(img):
    First_Image = img
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
    edge_image = cv2.Canny(blur_image, 250, 100)

    #using otsu's agorithm to perform binarization
    retVal,thresh_image = cv2.threshold(edge_image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return grey_image, thresh_image
