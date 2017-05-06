import cv2
import numpy as np

'''
    CANDIDATE ANALYSIS AND PLATE EXTRACTION
'''

def candidate_analysis(full_set_plates):
    
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
    
    return strong_plates, fuzzy_plates


def plate_extraction(strong_plates):
    # Strong and Fuzzy plate analysis to get the best candidate
    for i in range(0, len(strong_plates)):
        plate = strong_plates[i]
        #plate_threshold_image = cv2.adaptiveThreshold(plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
        p_h, p_w = plate.shape
        
        #resiszing the plate if the height and width are below a certain size
        if p_h < 74 or p_w < 285: 
            plate_to_save = cv2.resize(plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) 
        else:
            plate_to_save = plate
        
        cv2.imwrite('saves/extractedplate' + str(i) + '.jpg', plate_to_save)
    return 

