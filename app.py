from flask import Flask, request, jsonify
import face_recognition
import cv2
import numpy as np
import base64
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

KNOWN_FACES_DIR = "known_faces"

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)


@app.route('/enroll', methods=['POST'])
def enroll():
    try:
        data = request.json
        if 'image' not in data or 'name' not in data:
            return jsonify({'status': 'error', 'message': 'Missing image or name'}), 400

        # Decode the Base64 image
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'status': 'error', 'message': 'Failed to decode image'}), 400

        # Encode face
        face_encodings = face_recognition.face_encodings(image)
        if not face_encodings:
            return jsonify({'status': 'error', 'message': 'No face detected in the image'}), 400

        # Save encoding to file
        np.save(os.path.join(KNOWN_FACES_DIR, f"{data['name']}.npy"), face_encodings[0])
        return jsonify({'status': 'success', 'message': f"Face registered for {data['name']}"}), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/verify', methods=['POST'])
def verify():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'status': 'error', 'message': 'Missing image'}), 400

        # Decode the Base64 image
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'status': 'error', 'message': 'Failed to decode image'}), 400

        face_encodings = face_recognition.face_encodings(image)
        if not face_encodings:
            return jsonify({'status': 'error', 'message': 'No face detected in the image'}), 400

        unknown_encoding = face_encodings[0]

        # Compare with known faces
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.endswith(".npy"):
                known_encoding = np.load(os.path.join(KNOWN_FACES_DIR, filename))
                name = os.path.splitext(filename)[0]

                results = face_recognition.compare_faces([known_encoding], unknown_encoding)
                if results[0]:
                    return jsonify({'status': 'success', 'message': f"Match found! This is {name}"}), 200

        return jsonify({'status': 'error', 'message': 'No match found'}), 404

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, port=5000)
