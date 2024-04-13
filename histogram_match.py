import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the histogram of the grayscale image
    hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
    
    # Calculate the cumulative distribution function (CDF) of the histogram
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]  # Normalize the CDF to [0, 255]
    
    # Perform histogram equalization
    equalized = np.interp(gray.flatten(), bins[:-1], cdf_normalized)
    equalized = equalized.reshape(gray.shape).astype(np.uint8)  # Convert back to uint8
    
    return equalized

def histogram_matching(image, reference):
    # Convert the images to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    
    # Calculate histograms
    hist_image, bins_image = np.histogram(gray_image.flatten(), 256, [0, 256])
    hist_reference, bins_reference = np.histogram(gray_reference.flatten(), 256, [0, 256])
    
    # Calculate cumulative distribution functions (CDFs)
    cdf_image = hist_image.cumsum()
    cdf_reference = hist_reference.cumsum()
    
    # Normalize CDFs
    cdf_image = (cdf_image - cdf_image.min()) * 255 / (cdf_image.max() - cdf_image.min())
    cdf_reference = (cdf_reference - cdf_reference.min()) * 255 / (cdf_reference.max() - cdf_reference.min())
    
    # Interpolation
    matched = np.interp(gray_image.flatten(), bins_image[:-1], cdf_reference)
    matched = matched.reshape(gray_image.shape).astype(np.uint8)
    
    return matched

# Load the input and reference images
image = cv2.imread('image_ex.jpeg')
reference = cv2.imread('main_but_5.jpeg')

# Perform histogram matching
matched_image = histogram_matching(image, reference)

# Display the original, reference, and matched images
plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Input')
plt.subplot(132), plt.imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)), plt.title('Reference')
plt.subplot(133), plt.imshow(matched_image, cmap='gray'), plt.title('Matched')
plt.show()
