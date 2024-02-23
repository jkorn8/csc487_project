import cv2
import numpy as np
import os
import time

TIMESTEP_FOR_IMAGE_DOWNLOAD = .5
MODEL_INPUT_IMAGE_DIMENSIONS = (250, 250)


def collect_data_helper(image, coords, output_directory_path, im_num):
    im = cv2.resize(image[coords[0]:coords[1], coords[2]:coords[3]],
                    MODEL_INPUT_IMAGE_DIMENSIONS, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(output_directory_path, str(im_num) + ".jpg"), im)


def face_detection_helper(collect_data, output_directory_path='test_images', max_num_images=10000):
    face_detector = cv2.dnn.readNetFromCaffe("do_not_delete/deploy.prototxt.txt",
                                             "do_not_delete/res10_300x300_ssd_iter_140000.caffemodel")

    cap = cv2.VideoCapture(0)  # May need to change to 1, different index for different devices
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

            if confidence > TIMESTEP_FOR_IMAGE_DOWNLOAD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 5)
                if collect_data:
                    t2 = time.time()
                    if (t2 - t1) > TIMESTEP_FOR_IMAGE_DOWNLOAD:
                        collect_data_helper(image=image, coords=(startX, startY, endX, endY),
                                            output_directory_path=output_directory_path, im_num=im_num)
                        im_num += 1
                        t1 = time.time()

        cv2.imshow("Output", image)

        if (cv2.waitKey(10) & 0xFF == ord('q')) or (collect_data and max_num_images <= im_num):
            break
    cap.release()
    cv2.destroyAllWindows()


def face_detection():
    face_detection_helper(collect_data=False)


def data_collection(output_directory_path, max_num_images=10000):
    face_detection_helper(collect_data=True, output_directory_path=output_directory_path, max_num_images=max_num_images)


face_detection()
