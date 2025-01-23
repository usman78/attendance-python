from flask import Flask, request, jsonify
import face_recognition
import os

app = Flask(__name__)

# Load known faces
known_faces = {}  # This should contain names and face encodings
known_faces_folder = "known_faces"  # Adjust based on your folder structure

# Load all enrolled faces at startup
for filename in os.listdir(known_faces_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        filepath = os.path.join(known_faces_folder, filename)
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            name = filename.split(".")[0]
            known_faces[name] = encodings[0]

@app.route('/verify', methods=['POST'])
def verify_face():
    # Ensure an image is provided
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image provided'})

    file = request.files['image']
    image = face_recognition.load_image_file(file)
    face_encodings = face_recognition.face_encodings(image)

    if not face_encodings:
        return jsonify({'status': 'error', 'message': 'No face detected'})

    # Match with known faces
    student_face = face_encodings[0]
    for name, face_encoding in known_faces.items():
        matches = face_recognition.compare_faces([face_encoding], student_face)
        if True in matches:
            return jsonify({'status': 'success', 'message': f'Face matched for {name}'})

    return jsonify({'status': 'error', 'message': 'Face not recognized'})

if __name__ == '__main__':
    app.run(debug=True)
