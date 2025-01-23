import face_recognition
import os
import numpy as np
import sys

KNOWN_FACES_DIR = "known_faces"

if not os.path.exists(KNOWN_FACES_DIR):
    print("Known faces directory does not exist. Please enroll some faces first.")
    sys.exit()

image_path = input("Enter the path to the image for verification: ")

if not os.path.exists(image_path):
    print("Image file does not exist. Please provide a valid path.")
    sys.exit()

try:
    # Load the image and extract face encodings
    unknown_image = face_recognition.load_image_file(image_path)
    unknown_encodings = face_recognition.face_encodings(unknown_image)

    if not unknown_encodings:
        print("No face detected in the image.")
        sys.exit()

    unknown_encoding = unknown_encodings[0]

    # Load known face encodings and compare
    match_found = False
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(".npy"):
            known_encoding = np.load(os.path.join(KNOWN_FACES_DIR, filename))
            name = os.path.splitext(filename)[0]

            # Compare the faces
            results = face_recognition.compare_faces([known_encoding], unknown_encoding)
            if results[0]:
                print(f"Match found! This is {name}")
                match_found = True
                break

    if not match_found:
        print("No match found. Face not recognized.")

except Exception as e:
    print(f"An error occurred: {str(e)}")
