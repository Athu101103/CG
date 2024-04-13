from __future__ import print_function
import cv2 as cv
import argparse
parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
parser.add_argument('--input', help='Path to input image.', default='lena.jpg')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))
if src is None:
 print('Could not open or find the image:', args.input)
 exit(0)
src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
dst = cv.equalizeHist(src)
cv.imshow('Source image', src)
cv.imshow('Equalized Image', dst)
cv.waitKey()


"matching"
import cv2
import numpy as np

def hist_match(source, template):
    # Convert images to grayscale
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Calculate histograms
    source_hist = cv2.calcHist([source_gray], [0], None, [256], [0,256])
    template_hist = cv2.calcHist([template_gray], [0], None, [256], [0,256])
    
    # Normalize histograms
    source_hist_norm = source_hist / source_gray.size
    template_hist_norm = template_hist / template_gray.size
    
    # Cumulative distribution functions
    source_cdf = source_hist_norm.cumsum()
    template_cdf = template_hist_norm.cumsum()
    
    # Map the pixel values
    lut = np.interp(source_cdf, template_cdf, range(256))
    matched = cv2.LUT(source_gray, lut.astype('uint8'))
    
    return matched

# Load the source and reference images
source_image = cv2.imread('source_image.jpg')
reference_image = cv2.imread('reference_image.jpg')

# Perform histogram matching
matched_image = hist_match(source_image, reference_image)

# Display the images
cv2.imshow('Source Image', source_image)
cv2.imshow('Reference Image', reference_image)
cv2.imshow('Matched Image', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
