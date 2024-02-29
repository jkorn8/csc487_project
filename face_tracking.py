import cv2
import numpy as np
import os
import time

TIMESTEP_FOR_IMAGE_DOWNLOAD = 0
MODEL_INPUT_IMAGE_DIMENSIONS = (250, 250)
CAMERA_INDEX = 0    # May need to change to 1, different index for different devices


def list_possible_camera_indices():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        try:
            if not cap.read()[0]:
                break
            else:
                arr.append(index)
        except Exception as e:
            print(e)
            break
        cap.release()
        index += 1
    return arr


def collect_data_helper(image, coords, output_directory_path, im_num):
    im = cv2.resize(image[coords[1]:coords[3], coords[0]:coords[2]],
                    MODEL_INPUT_IMAGE_DIMENSIONS, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(output_directory_path, str(im_num) + ".jpg"), im)


def face_detection_helper(collect_data, output_directory_path='test_images', max_num_images=10000):
    face_detector = cv2.dnn.readNetFromCaffe("do_not_delete/deploy.prototxt.txt",
                                             "do_not_delete/res10_300x300_ssd_iter_140000.caffemodel")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    im_num = 0
    t1 = time.time()
    while cap.isOpened():
        _, image = cap.read()

        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        face_detector.setInput(blob)
        detections = face_detector.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > .5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 5)

        cv2.imshow("Output", image)

        if collect_data:
            c = -1
            k = -1
            for j in range(0, detections.shape[2]):
                if detections[0, 0, j, 2] > c:
                    c = detections[0, 0, j, 2]
                    k = j
            if c > .5:
                box = detections[0, 0, k, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                t2 = time.time()
                if (t2 - t1) > TIMESTEP_FOR_IMAGE_DOWNLOAD:
                    collect_data_helper(image=image, coords=(max(startX, 0), max(startY, 0), endX, endY),
                                        output_directory_path=output_directory_path, im_num=im_num)
                    im_num += 1
                    t1 = time.time()

        if (cv2.waitKey(10) & 0xFF == ord('q')) or (collect_data and max_num_images <= im_num):
            cap.release()
            cv2.destroyAllWindows()
            for i in range(2):
                cv2.waitKey(1)
            break


def face_detection():
    face_detection_helper(collect_data=False)


def data_collection(output_directory_path, max_num_images=10000):
    face_detection_helper(collect_data=True, output_directory_path=output_directory_path, max_num_images=max_num_images)