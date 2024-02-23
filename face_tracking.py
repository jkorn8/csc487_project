import cv2
import numpy as np

# Load the DNN Face Detector model
face_detector = cv2.dnn.readNetFromCaffe("do_not_delete/deploy.prototxt.txt", "do_not_delete/res10_300x300_ssd_iter_140000.caffemodel")

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, image = cap.read()

    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_detector.setInput(blob)
    detections = face_detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

    cv2.imshow("Output", image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()