from flask import Flask, render_template, request, url_for  # Fix here
import cv2
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
RESULT_FOLDER = os.path.join("static", "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def color_quantization(img, k):
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    return centers[labels.flatten()].reshape(img.shape)

def edge_mask(img, line_size, blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    return cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)

def smooth_image(img):
    return cv2.bilateralFilter(img, d=7, sigmaColor=200, sigmaSpace=200)

def cartoonify(img, line_size=7, blur_value=7, k=9):
    edges = edge_mask(img, line_size, blur_value)
    quantized = color_quantization(img, k)
    smoothed = smooth_image(quantized)
    return cv2.bitwise_and(smoothed, smoothed, mask=edges)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected.")
        if not allowed_file(file.filename):
            return render_template("index.html", error="Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP.")

        try:
            filename = file.filename
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)

            img = cv2.imread(upload_path)
            if img is None:
                raise ValueError("Failed to read the uploaded image.")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cartoon_img = cartoonify(img_rgb)
            
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            result_filename = f"cartoon_{timestamp}_{filename}"
            result_path = os.path.join(RESULT_FOLDER, result_filename)
            
            cv2.imwrite(result_path, cv2.cvtColor(cartoon_img, cv2.COLOR_RGB2BGR))
            
            if not os.path.exists(result_path):
                raise FileNotFoundError("Failed to save the processed image.")

            return render_template(
                "index.html",
                uploaded=True,
                original_image=url_for("static", filename=f"uploads/{filename}"),
                result_image=url_for("static", filename=f"results/{result_filename}"),
                timestamp=timestamp
            )

        except Exception as e:
            return render_template("index.html", error=f"Error: {str(e)}")

    return render_template("index.html", uploaded=False)

if __name__ == "__main__":
    app.run(debug=True)
