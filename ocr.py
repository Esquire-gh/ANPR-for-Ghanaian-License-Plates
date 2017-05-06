import cv2
import numpy as np
import io
import os

import pytesseract
from google.cloud import vision

try:
	import Image
except ImportError:
	import PIL.Image


'''
    CHARACTER RECOGNITION
'''
# using the tesseract OCR
def ocr_by_tesseract():
    img_with_chars = PIL.Image.open('saves/chars.jpeg')
    text = pytesseract.image_to_string(img_with_chars)
    print('Number Plate: {}'.format(text))

    return

# using the Google Cloud Machine Learning Engine for OCR
def cloud_ml_ocr():
    vision_client = vision.Client('anpr-166523')

    with io.open('saves/chars.jpeg', 'rb') as image_file:
        content = image_file.read()

    image = vision_client.image(content=content)

    texts = image.detect_text()
    print("USING THE ML ENGINE")
    print('Plate:')

    for text in texts:
        print('\n"{}"'.format(text.description))
    
    return
