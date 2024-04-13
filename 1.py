import cv2
import numpy as np 
# Load the image
image = cv2.imread('path_to_image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert the image to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Convert the image to LAB
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Display the original image and the converted images
cv2.imshow('Original Image', image)
cv2.imshow('Grayscale', gray)
cv2.imshow('HSV', hsv)
cv2.imshow('LAB', lab)

# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply different geometric transformations

# Rotate the image
rows, cols = image.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

# Scale the image
scaled_image = cv2.resize(image, None, fx=0.5, fy=0.5)

# Translate the image
translation_matrix = np.float32([[1, 0, 50], [0, 1, 50]])
translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))

# Display the transformed images
cv2.imshow('Rotated Image', rotated_image)
cv2.imshow('Scaled Image', scaled_image)
cv2.imshow('Translated Image', translated_image)

# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply image thresholding
_, thresholded_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, thresholded_image_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
_, thresholded_image_trunc = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
_, thresholded_image_tozero = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
_, thresholded_image_tozero_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)

# Display the thresholded images
cv2.imshow('Thresholded Image (Binary)', thresholded_image)
cv2.imshow('Thresholded Image (Binary Inverted)', thresholded_image_inv)
cv2.imshow('Thresholded Image (Truncated)', thresholded_image_trunc)
cv2.imshow('Thresholded Image (To Zero)', thresholded_image_tozero)
cv2.imshow('Thresholded Image (To Zero Inverted)', thresholded_image_tozero_inv)

# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply image smoothing
# Gaussian blur
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
# Median blur
median_blur = cv2.medianBlur(image, 5)
# Averaging blur
average_blur = cv2.blur(image, (5, 5))

# Display the smoothed images
cv2.imshow('Gaussian Blur', gaussian_blur)
cv2.imshow('Median Blur', median_blur)
cv2.imshow('Average Blur', average_blur)

# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
# Bilateral blur
bilateral_blur = cv2.bilateralFilter(image, 9, 75, 75)

# Display the smoothed images
cv2.imshow('Gaussian Blur', gaussian_blur)
cv2.imshow('Median Blur', median_blur)
cv2.imshow('Average Blur', average_blur)
cv2.imshow('Bilateral Blur', bilateral_blur)

# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply morphological operations
# Erosion
kernel = np.ones((5, 5), np.uint8)
eroded_image = cv2.erode(image, kernel, iterations=1)
# Dilation
dilated_image = cv2.dilate(image, kernel, iterations=1)
# Opening
opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
# Closing
closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
# Morphological Gradient
gradient_image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
# Top Hat
tophat_image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
# Black Hat
blackhat_image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
# Display the morphological operation results
cv2.imshow('Eroded Image', eroded_image)
cv2.imshow('Dilated Image', dilated_image)
cv2.imshow('Opened Image', opened_image)
cv2.imshow('Closed Image', closed_image)
cv2.imshow('Morphological Gradient', gradient_image)
cv2.imshow('Top Hat', tophat_image)
cv2.imshow('Black Hat', blackhat_image)
# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculate the image gradient
gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
# Calculate the magnitude and direction of the gradient
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
gradient_direction = np.arctan2(gradient_y, gradient_x)
# Display the gradient images
cv2.imshow('Gradient X', gradient_x)
cv2.imshow('Gradient Y', gradient_y)
cv2.imshow('Gradient Magnitude', gradient_magnitude)
cv2.imshow('Gradient Direction', gradient_direction)
# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
# Apply Laplacian operator
laplacian = cv2.Laplacian(image, cv2.CV_64F)
# Display the Laplacian image
cv2.imshow('Laplacian', laplacian)
# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply Scharr operator
scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
# Calculate the magnitude and direction of the Scharr gradient
scharr_magnitude = np.sqrt(scharr_x**2 + scharr_y**2)
scharr_direction = np.arctan2(scharr_y, scharr_x)
# Display the Scharr gradient images
cv2.imshow('Scharr X', scharr_x)
cv2.imshow('Scharr Y', scharr_y)
cv2.imshow('Scharr Magnitude', scharr_magnitude)
cv2.imshow('Scharr Direction', scharr_direction)
# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply Canny edge detection
canny_edges = cv2.Canny(image, 100, 200)
# Display the Canny edges
cv2.imshow('Canny Edges', canny_edges)
# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculate and display the histograms
hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])

# Plot the histograms
plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.plot(hist_gray, color='gray')
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
plt.plot(hist_b, color='b')
plt.title('Blue Channel Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.plot(hist_g, color='g')
plt.title('Green Channel Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(2, 2, 4)
plt.plot(hist_r, color='r')
plt.title('Red Channel Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
# Apply a mask to the image
masked_image = cv2.bitwise_and(image, image, mask=thresholded_image)
# Display the masked image
cv2.imshow('Masked Image', masked_image)
# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
# Apply a mask to the grayscale image
masked_gray = cv2.bitwise_and(gray, gray, mask=thresholded_image)
# Calculate and display the histogram of the masked grayscale image
hist_masked_gray = cv2.calcHist([masked_gray], [0], None, [256], [0, 256])
plt.figure()
plt.plot(hist_masked_gray, color='gray')
plt.title('Masked Grayscale Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
# Apply a mask to the BGR image
masked_bgr = cv2.bitwise_and(image, image, mask=thresholded_image)
# Calculate and display the histograms of the masked BGR image
hist_masked_b = cv2.calcHist([masked_bgr], [0], None, [256], [0, 256])
hist_masked_g = cv2.calcHist([masked_bgr], [1], None, [256], [0, 256])
hist_masked_r = cv2.calcHist([masked_bgr], [2], None, [256], [0, 256])
plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.plot(hist_masked_gray, color='gray')
plt.title('Masked Grayscale Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.subplot(2, 2, 2)
plt.plot(hist_masked_b, color='b')
plt.title('Masked Blue Channel Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.subplot(2, 2, 3)
plt.plot(hist_masked_g, color='g')
plt.title('Masked Green Channel Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.subplot(2, 2, 4)
plt.plot(hist_masked_r, color='r')
plt.title('Masked Red Channel Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Apply Fourier Transform to the grayscale image
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Display the magnitude spectrum
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.show()

# Apply Inverse Fourier Transform to reconstruct the image
f_ishift = np.fft.ifftshift(fshift)
reconstructed_image = np.fft.ifft2(f_ishift)
reconstructed_image = np.abs(reconstructed_image)

# Display the reconstructed image
cv2.imshow('Reconstructed Image', reconstructed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Load the template image
template = cv2.imread('path_to_template.jpg', 0)

# Perform template matching using different methods
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

for method in methods:
    # Apply template matching
    result = cv2.matchTemplate(gray, template, method)
    # Find the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # Draw a rectangle around the best match
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# Display the image with the template matches
cv2.imshow('Template Matches', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply Hough Line Transform
edges = cv2.Canny(gray, 50, 150)
lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

# Draw the detected lines on the image
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the image with the detected lines
cv2.imshow('Hough Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply Hough Circle Transform
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
# Convert the circles to integers
circles = np.uint16(np.around(circles))
# Draw the detected circles on the image
if circles is not None:
    for circle in circles[0, :]:
        center = (circle[0], circle[1])
        radius = circle[2]
        cv2.circle(image, center, radius, (0, 255, 0), 2)
# Display the image with the detected circles
cv2.imshow('Hough Circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply image segmentation using watershed algorithm
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply adaptive thresholding to obtain a binary image
_, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# Apply morphological operations to remove noise and fill gaps
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
# Calculate the distance transform
distance_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
# Apply thresholding to obtain markers for watershed algorithm
_, markers = cv2.threshold(distance_transform, 0.7 * distance_transform.max(), 255, cv2.THRESH_BINARY)
# Apply connected component labeling to assign labels to markers
markers = cv2.connectedComponents(markers)[1]
# Apply watershed algorithm to segment the image
segmented_image = cv2.watershed(image, markers)
# Apply color mapping to visualize the segmented regions
segmented_image = np.uint8(segmented_image)
segmented_image = cv2.applyColorMap(segmented_image, cv2.COLORMAP_JET)
# Display the segmented image
cv2.imshow('Segmented Image', segmented_image)
# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()