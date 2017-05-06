import cv2
import numpy
import preprocess
import localization
import plate_analysis_and_extraction
import segmentation
import ocr
'''
    this is the main application for the anpr using edge detection and machine learning
    Written by: Richard Ackon
'''

# reading the image
Image = cv2.imread('test samples/4.jpg')

#preprocessing the image
grey_image, thresh_image = preprocess.image_preprocessing(Image)

#performing localization on the preprocessed image
long_plates, short_plates, full_set_plates = localization.plate_localization(thresh_image, grey_image, Image)

#performing plate analysis and extraction
strong_plates, fuzzy_plates = plate_analysis_and_extraction.candidate_analysis(full_set_plates)

plate_analysis_and_extraction.plate_extraction(strong_plates)

# Read in image for segmentation
img = cv2.imread('saves/extracted_plate.jpg', 0)

#performing plate segmentation
segmented_image = segmentation.plate_segmentation(img)

#saving segemented image for ocr
cv2.imwrite('saves/chars.jpeg', segmented_image)

#perform OCR on the segmented plate
ocr.ocr_by_tesseract()
ocr.cloud_ml_ocr()


'''
for i in range(0, len(strong_plates)):
    cv2.imshow(str(i), strong_plates[i])
'''



cv2.imshow('bla', thresh_image)

cv2.waitKey(0)
cv2.destroyAllWindows