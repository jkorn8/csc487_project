import cv2
import numpy as np
import os
import time
from helpers import display_progressbar, preprocess, deprocess_image
from constants import MODEL_INPUT_IMAGE_DIMENSIONS
import random
import albumentations as alb
from PIL import Image

TIMESTEP_FOR_IMAGE_DOWNLOAD = 0
CAMERA_INDEX = 0    # May need to change to 1, different index for different devices
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)


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


def collect_data_helper(image, coords, output_directory_path, im_num, num_augmentations_per_image):
    augmentor = alb.Compose([
        alb.HorizontalFlip(p=0.5),
        alb.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
        alb.RandomGamma(p=0.3, gamma_limit=(80, 120)),
        alb.RGBShift(p=0.3, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
        alb.VerticalFlip(p=0.2),
        alb.Rotate(limit=20, p=0.5, interpolation=cv2.INTER_LINEAR)
    ])
    image = image[:, :, ::-1]
    image = cv2.resize(image[coords[1]:coords[3], coords[0]:coords[2]],
                       MODEL_INPUT_IMAGE_DIMENSIONS,
                       interpolation=cv2.INTER_LINEAR)
    for j in range(num_augmentations_per_image):
        aug_img = Image.fromarray(np.array(augmentor(image=image)['image']))
        aug_img.save(
            os.path.join(output_directory_path,
                         "aug" + str(j) + "_" + str(im_num) + ".jpg"))
    image = Image.fromarray(image)
    image.save(os.path.join(output_directory_path, str(im_num) + ".jpg"))
    print("Processed " + str(im_num))


def face_detection_helper(collect_data, output_directory_path='test_images',
                          max_num_images=10000, model=None, starting_index=0, num_augmentations_per_image=10):
    face_detector = cv2.dnn.readNetFromCaffe("do_not_delete/deploy.prototxt.txt",
                                             "do_not_delete/res10_300x300_ssd_iter_140000.caffemodel")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    im_num = 0
    t1 = time.time()
    while cap.isOpened():
        _, image = cap.read()

        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        face_detector.setInput(blob)
        detections = face_detector.forward()

        if collect_data:
            c = -1
            k = -1
            for j in range(0, detections.shape[2]):
                if detections[0, 0, j, 2] > c:
                    c = detections[0, 0, j, 2]
                    k = j
            if c > .8:
                box = detections[0, 0, k, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                t2 = time.time()
                if (t2 - t1) > TIMESTEP_FOR_IMAGE_DOWNLOAD:
                    collect_data_helper(image=image, coords=(max(startX, 0), max(startY, 0), endX, endY),
                                        output_directory_path=output_directory_path, im_num=im_num + starting_index,
                                        num_augmentations_per_image=num_augmentations_per_image)
                    im_num += 1
                    t1 = time.time()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > .8:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                if model is not None:
                    im = cv2.resize(image[max(startY, 0):endY, max(startX, 0):endX],
                                    MODEL_INPUT_IMAGE_DIMENSIONS, interpolation=cv2.INTER_LINEAR)
                    pred = model.predict(np.array([im[:, :, ::-1]]))
                    if pred is not None and pred[2] < .4:
                        if pred[0] > .6:
                            cv2.putText(image, 'Stephen', (endX+5, endY), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, COLOR_GREEN, 2, cv2.LINE_AA)
                            cv2.rectangle(image, (startX, startY), (endX+5, endY), COLOR_GREEN, 5)
                        elif pred[1] > .6:
                            cv2.putText(image, 'Josh', (endX + 5, endY), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, COLOR_GREEN, 2, cv2.LINE_AA)
                            cv2.rectangle(image, (startX, startY), (endX + 5, endY), COLOR_GREEN, 5)
                    else:
                        cv2.putText(image, 'Unknown', (endX, endY), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, COLOR_RED, 2, cv2.LINE_AA)
                        cv2.rectangle(image, (startX, startY), (endX, endY), COLOR_RED, 5)
                else:
                    cv2.rectangle(image, (startX, startY), (endX, endY), COLOR_RED, 5)

        cv2.imshow("Output", image)

        if (cv2.waitKey(10) & 0xFF == ord('q')) or (collect_data and (max_num_images <= len(os.listdir(output_directory_path)))):
            cap.release()
            cv2.destroyAllWindows()
            for i in range(2):
                cv2.waitKey(1)
            break


def process_unknown_celeb_faces(input_path, output_path, face_detector, num_augmentations_per_image):
    augmentor = alb.Compose([
        alb.HorizontalFlip(p=0.5),
        alb.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
        alb.RandomGamma(p=0.3, gamma_limit=(80, 120)),
        alb.RGBShift(p=0.3, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
        alb.VerticalFlip(p=0.2),
        alb.Rotate(limit=20, p=0.5, interpolation=cv2.INTER_LINEAR)
    ])

    t_orig = time.time()
    total_photos = len(os.listdir(input_path))
    total_processed = 0
    t = time.time()
    exceptions = []
    faces = os.listdir(input_path)
    random.shuffle(faces)
    print("Processing {} photos...".format(total_photos))
    for face in faces:
        if str.endswith(face, '.jpg'):
            image = preprocess(input_path=os.path.join(input_path, face),
                               dimensions=MODEL_INPUT_IMAGE_DIMENSIONS)
            image = deprocess_image(image)

            (h, w) = image.shape[:2]

            blob = cv2.dnn.blobFromImage(image, 1.0, (image.shape[0], image.shape[1]), (104.0, 177.0, 123.0))

            face_detector.setInput(blob)
            detections = face_detector.forward()

            confidence = -1
            i = 0
            for j in range(0, detections.shape[2]):
                if detections[0, 0, j, 2] > confidence:
                    confidence = detections[0, 0, j, 2]
                    i = j

            if confidence > .9:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                try:
                    image = cv2.resize(image[max(startY, 0):min(endY, image.shape[1]),
                                       max(startX, 0):min(endX, image.shape[0])],
                                       MODEL_INPUT_IMAGE_DIMENSIONS,
                                       interpolation=cv2.INTER_LINEAR)
                    for j in range(num_augmentations_per_image):
                        aug_img = Image.fromarray(np.array(augmentor(image=image)['image']))
                        aug_img.save(
                            os.path.join(output_path, "aug" + str(j) + "_" + face.split('.')[0] + "_celeba.jpg"))
                    image = Image.fromarray(image)
                    image.save(os.path.join(output_path, face.split('.')[0] + "_celeba.jpg"))
                except Exception as e:
                    exceptions.append(e)
            total_processed += 1
            t = display_progressbar(t1=t, t_orig=t_orig, total_photos=total_photos, total_processed=total_processed)

    print("\r\nDone!")
    if len(exceptions) > 0:
        print("Some exceptions occurred:")
        for e in exceptions:
            print(e)


def process_unknown_digiface_faces(input_path, output_path, face_detector, num_augmentations_per_image):
    augmentor = alb.Compose([
        alb.HorizontalFlip(p=0.5),
        alb.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
        alb.RandomGamma(p=0.3, gamma_limit=(80, 120)),
        alb.RGBShift(p=0.3, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
        alb.VerticalFlip(p=0.2),
        alb.Rotate(limit=20, p=0.5, interpolation=cv2.INTER_LINEAR)
    ])

    t_orig = time.time()
    total_photos = len(os.listdir(input_path)) * 72
    total_processed = 0
    t = time.time()
    exceptions = []
    faces = os.listdir(input_path)
    random.shuffle(faces)
    print("Processing {} photos...".format(total_photos))
    for directory in faces:
        if os.path.isdir(os.path.join(input_path, directory)):
            for face in os.listdir(os.path.join(input_path, directory)):
                if str.endswith(face, '.png'):
                    image = preprocess(input_path=os.path.join(input_path, directory, face), dimensions=MODEL_INPUT_IMAGE_DIMENSIONS)
                    image = deprocess_image(image)

                    (h, w) = image.shape[:2]

                    blob = cv2.dnn.blobFromImage(image, 1.0, (image.shape[0], image.shape[1]), (104.0, 177.0, 123.0))

                    face_detector.setInput(blob)
                    detections = face_detector.forward()

                    confidence = -1
                    i = 0
                    for j in range(0, detections.shape[2]):
                        if detections[0, 0, j, 2] > confidence:
                            confidence = detections[0, 0, j, 2]
                            i = j

                    if confidence > .9:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        try:
                            image = cv2.resize(image[max(startY, 0):min(endY, image.shape[1]),
                                               max(startX, 0):min(endX, image.shape[0])],
                                               MODEL_INPUT_IMAGE_DIMENSIONS,
                                               interpolation=cv2.INTER_LINEAR)
                            for j in range(num_augmentations_per_image):
                                aug_img = Image.fromarray(np.array(augmentor(image=image)['image']))
                                aug_img.save(
                                    os.path.join(output_path,
                                                 directory + "_aug" + str(j) + "_" + face.split('.')[0] + "_digiface.jpg"))
                            image = Image.fromarray(image)
                            image.save(os.path.join(output_path, directory + '_' + face.split('.')[0] + "_digiface.jpg"))
                        except Exception as e:
                            exceptions.append(e)
                    total_processed += 1
                    t = display_progressbar(t1=t, t_orig=t_orig, total_photos=total_photos, total_processed=total_processed)

    print("\r\nDone!")
    if len(exceptions) > 0:
        print("Some exceptions occurred:")
        for e in exceptions:
            print(e)

def face_detection(model):
    face_detection_helper(collect_data=False, model=model)


def data_collection(output_directory_path, max_num_images=10000, starting_index=0, num_augmentations_per_image=10):
    face_detection_helper(collect_data=True, output_directory_path=output_directory_path, max_num_images=max_num_images, starting_index=starting_index, num_augmentations_per_image=num_augmentations_per_image)
