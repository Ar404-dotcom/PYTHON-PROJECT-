import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_quantization(img, k):
    data = np.float32(img).reshape((-1, 3))

    # Define criteria and apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers to uint8 and map back
    centers = np.uint8(centers)
    quantized_img = centers[labels.flatten()]
    quantized_img = quantized_img.reshape(img.shape)

    return quantized_img

def edge_mask(img, line_size, blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, 
                                  cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 
                                  line_size, blur_value)
    return edges

def smooth_image(img):
    return cv2.bilateralFilter(img, d=7, sigmaColor=200, sigmaSpace=200)

def cartoonify(img, line_size=7, blur_value=7, k=9):
    edges = edge_mask(img, line_size, blur_value)
    quantized = color_quantization(img, k)
    smoothed = smooth_image(quantized)
    cartoon_img = cv2.bitwise_and(smoothed, smoothed, mask=edges)
    return cartoon_img

def process_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cartoon_img = cartoonify(img)
    plt.imshow(cartoon_img)
    plt.axis("off")
    plt.show()

# Get user input
image_path = input("Enter the path of the image: ")
process_image(image_path)



lur_value=7, k=9):
    edges = edge_mask(img, line_size, blur_value)
    quantized = color_quantization(img, k)
    smoothed = smooth_image(quantized)
    
    cartoon_img = cv2.bitwise_and(smoothed, smoothed, mask=edges)
    
    return cartoon_img

# Read and preprocess the image
img = cv2.imread("german_shepherd.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply cartoonification
cartoon_img = cartoonify(img)

# Display the cartoonified image
plt.imshow(cartoon_img)
plt.axis("off")
plt.show()
