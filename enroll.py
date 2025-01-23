import face_recognition
import os
import cv2

KNOWN_FACES_DIR = "known_faces"

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

name = input("Enter the student's name: ")
image_path = input("Enter the path to the student's image: ")

image = face_recognition.load_image_file(image_path)
encodings = face_recognition.face_encodings(image)

if encodings:
    cv2.imwrite(os.path.join(KNOWN_FACES_DIR, f"{name}.jpg"), image)
    print(f"Enrolled {name} successfully.")
else:
    print("No face detected. Enrollment failed.")
