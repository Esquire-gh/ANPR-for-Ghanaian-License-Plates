import cv2
import numpy as np

'''
    PLATE LOCALIZATION
'''

def plate_localization(thresh_image, grey_image, First_Image):
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
    
    return long_plates, short_plates, full_set_plates

