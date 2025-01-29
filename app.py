from flask import Flask, request, jsonify
import cv2
import face_recognition
import numpy as np
import os
import pickle
import base64
from flask_cors import CORS
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS

KNOWN_FACES_DIR = "known_faces"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def decode_image(image_data):
    image_data = image_data.split(",")[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    return np.array(image)

@app.route('/register_face', methods=['POST'])
def register_face():
    try:
        name = request.args.get('name')
        if not name:
            return jsonify({"error": "Name is required"}), 400

        image_data = request.json.get('image')
        if not image_data:
            return jsonify({"error": "Image data is required"}), 400

        image = decode_image(image_data)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_image)

        if not face_encodings:
            return jsonify({"error": "No face detected in image"}), 400

        encoding = face_encodings[0]
        file_path = os.path.join(KNOWN_FACES_DIR, f"{name}.pkl")
        
        with open(file_path, "wb") as f:
            pickle.dump(encoding, f)
        
        return jsonify({"message": f"Face registered for {name}"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    try:
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({"error": "Image data is required"}), 400

        image = decode_image(image_data)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        known_encodings = []
        known_names = []

        # Load known faces
        for file_name in os.listdir(KNOWN_FACES_DIR):
            if file_name.endswith(".pkl"):
                name = os.path.splitext(file_name)[0]
                file_path = os.path.join(KNOWN_FACES_DIR, file_name)
                if os.path.getsize(file_path) > 0:
                    with open(file_path, "rb") as f:
                        encoding = pickle.load(f)
                        known_encodings.append(encoding)
                        known_names.append(name)

        # Detect faces in the image
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        # Find matches
        names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]
            names.append(name)

        return jsonify({"names": names}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
