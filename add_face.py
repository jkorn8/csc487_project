import cv2
import os


# TODO: Replace model with stephens detector and increase the number of training images
# Load the pre-trained face cascade classifier
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# if not os.path.exists('./app_data'):
#     os.makedirs('./app_data')

# Create new capture and set dimensions to normalize
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Initialize face capture variables
count = 0
face_id = input('\nEnter user id (MUST be an integer) and press <return> -->  ')
if not os.path.exists(f'./app_data/{face_id}'):
    os.makedirs(f'./app_data/{face_id}')
print("\n[INFO] Initializing face capture. Look at the camera and wait...")

while True:
    # Read a frame from the camera
    _, frame = cap.read()
    faces = faceCascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imwrite(f"./app_data/{face_id}/Users-{face_id}-{count}.jpg", frame[y+2:y + h - 2, x+2:x + w -2])
        count += 1
        cv2.imshow('image', frame)

    if cv2.waitKey(10) & 0xFF < 30:
        break
    elif count >= 1:
        break

print("\n[INFO]Success! Exiting Program.")

cap.release()
cv2.destroyAllWindows()
for i in range(2):
    cv2.waitKey(1)
