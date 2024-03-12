import cv2
import os
import numpy as np
import tensorflow as tf
from deepface import DeepFace
from constants import PATH_TO_FACE_MATCHING, MODEL_INPUT_IMAGE_DIMENSIONS
from helpers import preprocess, deprocess_image

# Files required:
# - an "app_data" directory with folders corresponding to names of people
# - "do_not_delete/deploy.prototxt.txt"
# - "model_final_2.h5"

one_time = False
CAMERA_INDEX = 0
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
PADDING = 100

# Facial Recognition Setup - load images
names = ['Stephen', 'Josh']
face_images = []
for folder in os.listdir(PATH_TO_FACE_MATCHING):
    path = os.path.join(PATH_TO_FACE_MATCHING, folder)
    if os.path.isdir(path):
        id = int(folder)
        for image in os.listdir(path):
            blob = deprocess_image(preprocess(os.path.join(path, image), MODEL_INPUT_IMAGE_DIMENSIONS))
            face_images.append((id, blob))
model_name = 'Facenet'

# Facial Detection setup
face_detector = cv2.dnn.readNetFromCaffe("do_not_delete/deploy.prototxt.txt", "do_not_delete/res10_300x300_ssd_iter_140000.caffemodel")
model = tf.keras.models.load_model('models/model_final_2.h5')

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(3, 640)
cap.set(4, 480)

while cap.isOpened():
    ret, frame = cap.read()

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > .9:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # This is the important part
            for id, face_image in face_images:
                result = DeepFace.verify(face_image, frame[startY:endY, startX:endX], model_name, enforce_detection=False)
                if result['verified']:
                    cv2.putText(frame, names[id], (endX + 5, endY), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_GREEN, 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLOR_GREEN, 5)
                    break
            else:
                cv2.putText(frame, 'Unknown', (endX + 5, endY), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2, cv2.LINE_AA)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLOR_RED, 5)

    cv2.imshow('Verification', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        for i in range(2):
            cv2.waitKey(1)
        break
