from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
import tensorflow as tf
import cv2
from waitress import serve

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model
model = tf.keras.models.load_model('model.h5')

# Emotion labels
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    try:
        # Convert to image
        image_data = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
        if len(faces_detected) == 0:
            return jsonify({'error': 'No face detected'}), 400

        # Process the first detected face
        x, y, w, h = faces_detected[0]
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = roi_gray.reshape(1, 48, 48, 1) / 255.0

        # Predict
        predictions = model.predict(img_pixels)
        predicted_emotion = emotions[np.argmax(predictions[0])]

        return jsonify({'prediction': predicted_emotion})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=5000)
    serve(app, host='0.0.0.0', port=5000)