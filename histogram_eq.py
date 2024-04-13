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

# Load the input image
image = cv2.imread('image_ex.jpeg')

# Perform histogram equalization
equalized_image = histogram_equalization(image)

# Convert the original image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the original, grayscale, and equalized images
plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(132), plt.imshow(gray_image, cmap='gray'), plt.title('Grayscale')
plt.subplot(133), plt.imshow(equalized_image, cmap='gray'), plt.title('Equalized')
plt.show()



# def histogram_equalization(image):
#     # Convert the image to 8-bit depth if it's not already
#     if image.dtype != 'uint8':
#         image = cv2.convertScaleAbs(image)

#     # Convert the image to grayscale if it's not already
#     if len(image.shape) > 2:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = image

#     # Perform histogram equalization
#     equalized = cv2.equalizeHist(gray)

#     # Convert the equalized image back to the original depth and format
#     if len(image.shape) > 2:
#         equalized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

#     return equalized

