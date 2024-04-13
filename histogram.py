import cv2
import numpy as np

import matplotlib.pyplot as plt

def histogram_equalization(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the histogram of the grayscale image
    hist, bins = np.histogram(gray.flatten(), 256, [0,256])
    
    # Calculate the cumulative distribution function (CDF) of the histogram
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    
    # Perform histogram equalization
    equalized = np.interp(gray.flatten(), bins[:-1], cdf_normalized)
    equalized = equalized.reshape(gray.shape)
    
    # Convert the equalized image back to BGR
    equalized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    
    return equalized

# Load the input image
image = cv2.imread('input.jpg')

# Perform histogram equalization
equalized_image = histogram_equalization(image)

# Display the original and equalized images
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(122), plt.imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB)), plt.title('Equalized')
plt.show()