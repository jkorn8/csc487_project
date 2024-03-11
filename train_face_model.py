import cv2
import numpy as np
from PIL import Image
import os

# TODO: Replace with Stephens detector
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer.create()

faces = []
ids = []

for folder in os.listdir('./app_data'):
    path = f'./app_data/{folder}'
    if os.path.isdir(path):
        id = int(folder)
        for image in os.listdir(path):
            print(image)
            # Convert image to grayscale
            PIL_img = Image.open(f'{path}/{image}').convert('L')
            img_numpy = np.array(PIL_img, 'uint8')

            # Extract the user ID from the image file name
            all_faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in all_faces:
                # Extract face region and append to the samples
                faces.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)

recognizer.train(faces, np.array(ids))
recognizer.write('trainer.yml')

print(f"Faces trained on: {len(set(ids))}\n")
