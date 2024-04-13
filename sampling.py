import cv2

def convert_to_grayscale(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform quantization by reducing the number of colors to 256
    quantized_image = cv2.convertScaleAbs(grayscale_image, alpha=(256.0/256.0))

    # Display the original and quantized images
    cv2.imshow("Original Image", image)
    cv2.imshow("Quantized Image", quantized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the input image
image_path = "path/to/your/image.jpg"

# Call the function to convert the image to grayscale
convert_to_grayscale(image_path)