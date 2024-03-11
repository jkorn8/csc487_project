import tensorflow as tf
import os
from constants import UNKNOWN_FACES_PATH, MODEL_INPUT_IMAGE_DIMENSIONS, MODEL_NEGATIVE_FAILED_PATH, MODEL_POSITIVE_FAILED_PATH
from helpers import preprocess, deprocess_image, display_progressbar
import numpy as np
import time
from face_tracking import face_detection
from PIL import Image
import shutil

model = tf.keras.models.load_model('model_best_4.h5')
'''
for file in os.listdir(MODEL_NEGATIVE_FAILED_PATH):
    file_path = os.path.join(MODEL_NEGATIVE_FAILED_PATH, file)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
=======
model = tf.keras.models.load_model('model_final_2.h5')
>>>>>>> 7db2ea87a4597aa020387566c4b1ecabd8bd221a

for file in os.listdir(MODEL_POSITIVE_FAILED_PATH):
    file_path = os.path.join(MODEL_POSITIVE_FAILED_PATH, file)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

num_failures_negative = 0
t_orig = time.time()
total_photos = len(os.listdir(UNKNOWN_FACES_PATH))
total_processed = 0
t = time.time()

for face in os.listdir(UNKNOWN_FACES_PATH):
    face_path = os.path.join(UNKNOWN_FACES_PATH, face)
    im = deprocess_image(preprocess(face_path, MODEL_INPUT_IMAGE_DIMENSIONS))
    pred = model.predict(np.array([im]), verbose=0)
    if pred > .5:
        num_failures_negative += 1
        print("\rNumber of failures for negative images: " + str(num_failures_negative))
        im = Image.fromarray(im)
        im.save(os.path.join(MODEL_NEGATIVE_FAILED_PATH, face))
    total_processed += 1
    t = display_progressbar(t1=t, t_orig=t_orig, total_photos=total_photos, total_processed=total_processed)

num_failures_positive = 0
t_orig = time.time()
total_photos = len(os.listdir(KNOWN_FACE_PATH))
total_processed = 0
t = time.time()

for face in os.listdir(KNOWN_FACE_PATH):
    face_path = os.path.join(KNOWN_FACE_PATH, face)
    im = deprocess_image(preprocess(face_path, MODEL_INPUT_IMAGE_DIMENSIONS))
    pred = model.predict(np.array([im]), verbose=0)
    if pred < .5:
        num_failures_positive += 1
        print("\rNumber of failures for positive images: " + str(num_failures_positive))
        im = Image.fromarray(im)
        im.save(os.path.join(MODEL_POSITIVE_FAILED_PATH, face))
    total_processed += 1
    t = display_progressbar(t1=t, t_orig=t_orig, total_photos=total_photos, total_processed=total_processed)

print("Total failures: " + str(num_failures_negative + num_failures_positive))
'''
face_detection(model=model)
