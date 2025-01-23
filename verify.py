import face_recognition
import os
import cv2
import sys

KNOWN_FACES_DIR = "known_faces"

# Ensure the directory for known faces exists
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Load the image to verify
image_path = input("Enter the path to the image for verification: ")

# Load the image and extract face encodings
unknown_image = face_recognition.load_image_file(image_path)
unknown_encodings = face_recognition.face_encodings(unknown_image)

# If no face encoding is found in the image
if not unknown_encodings:
    print("No face detected in the image.")
    sys.exit()

# Take the first detected face encoding (if more than one face is found)
unknown_encoding = unknown_encodings[0]

# Loop through the images in the KNOWN_FACES_DIR to find a match
for filename in os.listdir(KNOWN_FACES_DIR):
    known_image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, filename))
    known_encoding = face_recognition.face_encodings(known_image)[0]

    # Compare the faces
    results = face_recognition.compare_faces([known_encoding], unknown_encoding)

    if results[0]:
        print(f"Match found! This is {os.path.splitext(filename)[0]}")
        break
else:
    print("No match found. Face not recognized.")
