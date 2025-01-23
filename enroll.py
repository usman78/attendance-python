import face_recognition
import os
import numpy as np

KNOWN_FACES_DIR = "known_faces"

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

name = input("Enter the student's name: ")

image_path = input("Enter the path to the student's image: ")

if not os.path.exists(image_path):
    print("Image file does not exist. Please provide a valid path.")
else:
    try:
        # Load the image and extract face encodings
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if not encodings:
            print("No face detected. Please use a clearer image.")
        else:
            # Save the encoding as a .npy file
            np.save(os.path.join(KNOWN_FACES_DIR, f"{name}.npy"), encodings[0])
            print(f"Enrolled {name} successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

