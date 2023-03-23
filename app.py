# to test the api, run the following command in the terminal
# go to postman and send a post request to http://address:105/analyze
# go to body and select form-data
# add two files with the keys file1 and file2
# select the files and send the request
# for crop send 1 image 
from flask import Flask, jsonify, request,send_file, render_template, redirect, url_for
from skimage.metrics import structural_similarity
import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
app = Flask(__name__)


app.config['DEBUG'] = True

# Set the upload folder and allowed file types
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'png', 'png', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file type is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    # If the user does not select a file, the browser may submit an empty file without a filename
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('display_image', filename=filename))

@app.route('/display/<filename>')
def display_image(filename):
    # Get the absolute path of the uploaded image file
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # Serve the image file to the browser
    return send_file(image_path, mimetype='image/png')

@app.route('/ssim_score', methods=['POST'])
def ssim_scores_api():
    # Get the uploaded files from the request
    file1 = request.files['file1']
    file2 = request.files['file2']

    # Save the files locally
    file1.save('file1.png')
    file2.save('file2.png')

    # Read the images
    first = cv2.imread('file1.png')
    second_original = cv2.imread('file2.png')
    second= cv2.bitwise_not(second_original)  ## Inverts Colors

    # Convert images to grayscale
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    score, diff = structural_similarity(first_gray, second_gray, full=True)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type so we must convert the array
    # to 8-bit unsigned integers in the range [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")
    return {
        'score': score*100,
        }

def readImg(filename):
    # Load image from file
    img = cv2.imread(filename)
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

@app.route('/canny', methods=['POST'])
def canny_api():
    # Get the uploaded file from the request
    file = request.files['file']

    # Save the file locally
    file.save('input.png')

    # Read the image
    img = readImg('input.png')

    # Apply Canny edge detection
    edges = cv2.Canny(img, 100, 200, 3, L2gradient=True)

    # Convert the edges image to png format
    ret, png = cv2.imencode('.png', edges)

    # Create a file-like object from the encoded png data
    file_bytes = io.BytesIO(png.tobytes())

    # Send the png file as a response
    return send_file(file_bytes, mimetype='image/png')

@app.route('/analyze', methods=['POST'])
def analyze_api():
    # Get the uploaded files from the request
    #reference image
    file1 = request.files['file1']
    #image to be compared
    file2 = request.files['file2']

    # Save the files locally
    file1.save('file1.png')
    file2.save('file2.png')

   # Read the image
    img = readImg('file1.png')

    # Apply Canny edge detection
    edges = cv2.Canny(img, 100, 200, 3, L2gradient=True)
    cv2.imwrite('file1.png', edges)
    # Convert the edges image to png format
    # ret, png = cv2.imencode('file1.png', edges)

    # Create a file-like object from the encoded png data
    first =  cv2.imread('file1.png')

    # Read the images
    # first = cv2.imread('file1.png')
    second_original = cv2.imread('file2.png')
    second= cv2.bitwise_not(second_original)  ## Inverts Colors

    # Convert images to grayscale
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    score, diff = structural_similarity(first_gray, second_gray, full=True)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type so we must convert the array
    # to 8-bit unsigned integers in the range [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image, followed by finding contours to
    # obtain the regions that differ between the two images
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Highlight differences
    mask = np.zeros(first.shape, dtype='uint8')
    ssim_output = second.copy()

    #For iterable results
    duplicate_input = second.copy()

    err = []

    for c in contours:
        area = cv2.contourArea(c)
        if area > 100:
            temp = duplicate_input.copy()

            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(first, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.rectangle(second, (x, y), (x + w, y + h), (36, 255, 12), 2)

            cv2.rectangle(temp, (x, y), (x + w, y + h), (36, 255, 12), 2)  #Bounding box for iterable results stored in temp image

            cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)

            cv2.drawContours(ssim_output, [c], 0, (0, 255, 0), -1)   #--All errors saved on final result image ssim_output

            cv2.drawContours(temp, [c], 0, (0, 255, 0), -1)        #Iterable errors stored in temp image

            err.append(temp)
    # save ssim_output image as png
    
    cv2.imwrite('analye_output.png', ssim_output)
    image_output = Image.open('analye_output.png')
    os.remove('file1.png')
    os.remove('file2.png')
    return send_file(image_output, mimetype='image/pmg')
    # return {"ssim_output": image_output, "score": score*100}

@app.route('/image')
def serve_image():
    # Assuming the image file is named "image.png" and located in the same directory as your Flask app
    return send_file('image.png', mimetype='image/png')

@app.route('/hello/', methods=['GET', 'POST'])
def welcome():
    return "Hello World!"

@app.route('/crop_image', methods=['POST'])
def crop_image():
    # Read the input image from the request
    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to obtain a binary image
    thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours in the binary image
    contours = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Loop over the contours and find the bounding box around each contour
    for i in contours:
        x,y,w,h = cv2.boundingRect(i)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 4)
        
        # Crop the image using the bounding box coordinates
        crop_img = img[y:y+h, x:x+w]
    
    # Convert the cropped image to bytes for returning in the response
    _, buffer = cv2.imencode('.png', crop_img)
    cropped_image = buffer.tobytes()
    with open("output.png", "wb") as f:
        f.write(cropped_image)
    # Return the cropped image with the bounding box drawn around it as the response
    return send_file(cropped_image, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)