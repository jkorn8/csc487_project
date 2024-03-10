import cv2
import os
import numpy as np
import tensorflow as tf
from deepface import DeepFace

# Files required:
# - an "app_data" directory with folders corresponding to names of people
# - "do_not_delete/deploy.prototxt.txt"
# - "model_final_2.h5"

one_time = False
CAMERA_INDEX = 0
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
MODEL_INPUT_IMAGE_DIMENSIONS = (100, 100)
PADDING = 100

# Facial Recognition Setup
pos_images = os.listdir('./app_data/josh')
model_name = 'Facenet'

# Facial Detection setup
face_detector = cv2.dnn.readNetFromCaffe("do_not_delete/deploy.prototxt.txt", "do_not_delete/res10_300x300_ssd_iter_140000.caffemodel")
model = tf.keras.models.load_model('model_final_2.h5')

cap = cv2.VideoCapture(CAMERA_INDEX)


def getDimensions(startX, startY, endX, endY, PADDING, h, w):
    x1 = max(0, startX - PADDING)
    x2 = min(w, endX + PADDING)
    y1 = max(0, startY - PADDING)
    y2 = min(h, endY + PADDING)
    return x1, x2, y1, y2


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
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLOR_RED, 5)
            # This is the important part
            (x1, x2, y1, y2) = getDimensions(startX, startY, endX, endY, PADDING, h, w)
            # cv2.imwrite('./testing.jpg', frame[y1:y2, x1:x2])
            conf = 0
            for pos_image in pos_images:
                result = DeepFace.verify(f'./app_data/josh/{pos_image}', frame[x1:x2, y1:y2], model_name, enforce_detection=False)
                conf += 0.1 if result['verified'] else 0
            if conf > 0.5:
                cv2.putText(frame, 'Josh', (endX + 5, endY), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_GREEN, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'Unknown', (endX + 5, endY), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2, cv2.LINE_AA)

    cv2.imshow('Verification', frame)

    # if cv2.waitKey(10) & 0xFF == ord('v'):
    #     box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    #     (startX, startY, endX, endY) = box.astype("int")
    #     # This is the important part
    #     (x1, x2, y1, y2) = getDimensions(startX, startY, endX, endY, PADDING, h, w)
    #     cv2.imwrite('./testing.jpg', frame[y1:y2, x1:x2])
    #     for pos_image in pos_images:
    #         result = DeepFace.verify(f'./app_data/josh/{pos_image}', './testing.jpg', model_name, enforce_detection=False)
    #         print(result['verified'])

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        for i in range(2):
            cv2.waitKey(1)
        break
