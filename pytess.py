import cv2
import numpy
from matplotlib import pyplot as plt
try:
	import Image
except ImportError:
	import PIL.Image
import pytesseract


# reading the image in grayscale form
img1 = cv2.imread('test samples/car1.jpg')
img = cv2.imread('test samples/car1.jpg', cv2.IMREAD_GRAYSCALE)

#applying otsu's thresholding method after Gaussian blur
blur_image = cv2.GaussianBlur(img, (5,5), 0)
ret,thresh_image = cv2.threshold(blur_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
dilated = cv2.dilate(thresh_image, kernel, iterations=13)

# applying sobel algorithm of edges on the image
sobel_horizontal = cv2.Sobel(thresh_image, cv2.CV_64F, 1, 0, ksize=5)
sobel_vertical = cv2.Sobel(thresh_image, cv2.CV_64F, 0, 1, ksize=5)

#finding contour in the thresh/binary images
thresh_image_copy = thresh_image.copy()
contours, hierarchy = cv2.findContours(thresh_image_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#cv2.drawContours(img, contours2, -1, (0, 255, 0), 3)

height, width = thresh_image.shape
thresh_image_array = numpy.zeros((height, width, 3), numpy.uint8)

#looping through the contours of the threshimage and removing nonplate contours
contour_images = []
for contour in contours:
	[x, y, w, h] = cv2.boundingRect(contour)
	
	if h>160 and w>300:
		continue

	if h<50 or w<80:
		continue
	
	cv2.rectangle(img1, (x,y), (x+w, y+h), (0, 255, 0), 2)
	
	#taking the sample plates out of the main image to they can be worked on seperately
	sample_plate = thresh_image[y:y+h, x:x+w]
	contour_images.append(sample_plate)

#looping through contour images to display them and process further
for i in range(0, len(contour_images)):
	cv2.imshow(str(i) + '.png', contour_images[i])
	cv2.imwrite(str(i) + '.png', contour_images[i])
	
	# applying canny edge detection on the image(possible plate from contours)
	plate_to_edge = cv2.imread(str(i) + '.png')
	edge_image = cv2.Canny(plate_to_edge,100,200)
	cv2.imwrite(str(i) + '.png', edge_image)
	
	#apply tesseract on the edge images to find the ones that return a possible plate
	im = PIL.Image.open(str(i) + '.png')
	plate_text = pytesseract.image_to_string(im)
	print('plate'+ str(i) + ': == ' + plate_text)

'''
# showing the image in its different forms
cv2.imshow('possible plates', img1)
cv2.imshow('grayscale', img)
cv2.imshow('thresh_image',thresh_image)
cv2.imshow('canny', edge_image)
'''

# the waitKey function it needed by opencv to allow 
# the usage of keyboard keys to close image windows
cv2.waitKey(0)