import os
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
from face_tracking import data_collection, process_unknown_faces
import numpy as np
import albumentations as alb
import matplotlib.pyplot as plt
import time
import tracemalloc
from helpers import display_progressbar, preprocess, get_distribution_array, shuffle_arrays_in_sync, deprocess_image


CELEB_ORIG_PATH = os.path.join('data', 'img_align_celeba')
UNKNOWN_FACES_PATH = os.path.join('data', 'unknown_faces')
KNOWN_FACES_PATH = os.path.join('data', 'known_faces')
KNOWN_FACE_PATH = os.path.join(KNOWN_FACES_PATH, 'stephen') # Change path for different identities

NUM_IMAGES = 5000

PERCENT_TRAINING = 80
PERCENT_TESTING = 19
PERCENT_VALIDATION = 1
NUM_AUGMENTATIONS_PER_IMAGE = 20
MODEL_INPUT_IMAGE_DIMENSIONS = (100, 100)
NUM_EPOCHS = 50

Face_Detector = cv2.dnn.readNetFromCaffe("do_not_delete/deploy.prototxt.txt",
                                             "do_not_delete/res10_300x300_ssd_iter_140000.caffemodel")

if len(os.listdir(UNKNOWN_FACES_PATH)) > 0:
    print("Found files in directory already. To rerun, please empty " + UNKNOWN_FACES_PATH)
else:
    process_unknown_faces(input_path=CELEB_ORIG_PATH, output_path=UNKNOWN_FACES_PATH, face_detector=Face_Detector)

if len(os.listdir(KNOWN_FACE_PATH)) > 0:
    print("Found files in directory already. To rerun, please empty " + KNOWN_FACE_PATH)
else:
    data_collection(KNOWN_FACE_PATH, 10000)

known_faces_dir = os.listdir(KNOWN_FACE_PATH)
unknown_faces_dir = os.listdir(UNKNOWN_FACES_PATH)
num_known_faces = len(known_faces_dir)
nun_unknown_faces = len(unknown_faces_dir)

process_known_face = get_distribution_array(num_known_faces, percent_training=PERCENT_TRAINING, percent_testing=PERCENT_TESTING, percent_validation=PERCENT_VALIDATION)
process_unknown_face = get_distribution_array(nun_unknown_faces, percent_training=PERCENT_TRAINING, percent_testing=PERCENT_TESTING, percent_validation=PERCENT_VALIDATION)

(training_images, training_labels), (test_images, test_labels), (val_images, val_labels) = ([], []), ([], []), ([], [])

t_orig = time.time()
total_photos = min(len(known_faces_dir), NUM_IMAGES)
total_processed = 0
t = time.time()
print("Processing {} known faces...".format(total_photos))
for i in range(len(known_faces_dir)):
    if i > NUM_IMAGES:
        break
    if process_known_face[i] == 0:
        training_images.append(deprocess_image(preprocess(input_path=os.path.join(KNOWN_FACE_PATH, known_faces_dir[i]), dimensions=MODEL_INPUT_IMAGE_DIMENSIONS)))
        training_labels.append(1)
    elif process_known_face[i] == 1:
        val_images.append(deprocess_image(preprocess(input_path=os.path.join(KNOWN_FACE_PATH, known_faces_dir[i]), dimensions=MODEL_INPUT_IMAGE_DIMENSIONS)))
        val_labels.append(1)
    else:
        test_images.append(deprocess_image(preprocess(input_path=os.path.join(KNOWN_FACE_PATH, known_faces_dir[i]), dimensions=MODEL_INPUT_IMAGE_DIMENSIONS)))
        test_labels.append(1)
    total_processed += 1
    t = display_progressbar(t1=t, t_orig=t_orig, total_photos=total_photos, total_processed=total_processed)
display_progressbar(t1=10, t_orig=0, total_photos=total_photos, total_processed=total_photos)
print("\r\nDone!")

t_orig = time.time()
total_photos = min(NUM_IMAGES, 200000)
total_processed = 0
t = time.time()
print("Processing {} unknown faces...".format(total_photos))
for i in range(len(unknown_faces_dir)):
    if i > NUM_IMAGES:
        break
    if process_unknown_face[i] == 0:
        training_images.append(deprocess_image(preprocess(input_path=os.path.join(UNKNOWN_FACES_PATH, unknown_faces_dir[i]), dimensions=MODEL_INPUT_IMAGE_DIMENSIONS)))
        training_labels.append(0)
    elif process_unknown_face[i] == 1:
        val_images.append(deprocess_image(preprocess(input_path=os.path.join(UNKNOWN_FACES_PATH, unknown_faces_dir[i]), dimensions=MODEL_INPUT_IMAGE_DIMENSIONS)))
        val_labels.append(0)
    else:
        test_images.append(deprocess_image(preprocess(input_path=os.path.join(UNKNOWN_FACES_PATH, unknown_faces_dir[i]), dimensions=MODEL_INPUT_IMAGE_DIMENSIONS)))
        test_labels.append(0)
    total_processed += 1
    t = display_progressbar(t1=t, t_orig=t_orig, total_photos=total_photos, total_processed=total_processed)
display_progressbar(t1=10, t_orig=0, total_photos=total_photos, total_processed=total_photos)
print("\r\nDone!")
del known_faces_dir, unknown_faces_dir, num_known_faces, nun_unknown_faces, process_known_face, process_unknown_face, t_orig, total_photos, total_processed, t

augmentor = alb.Compose([alb.HorizontalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.3),
                         alb.RandomGamma(p=0.3),
                         alb.RGBShift(p=.3),
                         alb.VerticalFlip(p=0.2),
                         alb.Rotate(limit=20, p=.5, interpolation=cv2.INTER_LINEAR)])

l = len(training_images)
t_orig = time.time()
total_photos = l
total_processed = 0
t = time.time()
print("Augmenting {} images in training data...".format(total_photos))
for i in range(l):
    for j in range(NUM_AUGMENTATIONS_PER_IMAGE):
        img = (training_images[i]*255).astype(np.uint8)
        training_images.append(np.array(augmentor(image=img)['image'])/255)
        training_labels.append(training_labels[i])
    total_processed += 1
    t = display_progressbar(t1=t, t_orig=t_orig, total_photos=total_photos, total_processed=total_processed)
display_progressbar(t1=10, t_orig=0, total_photos=total_photos, total_processed=total_photos)
print("\r\nDone!")

l = len(test_images)
t_orig = time.time()
total_photos = l
total_processed = 0
t = time.time()
print("Augmenting {} images in test data...".format(total_photos))
for i in range(l):
    for j in range(NUM_AUGMENTATIONS_PER_IMAGE):
        img = (test_images[i]*255).astype(np.uint8)
        test_images.append(np.array(augmentor(image=img)['image'])/255)
        test_labels.append(test_labels[i])
    total_processed += 1
    t = display_progressbar(t1=t, t_orig=t_orig, total_photos=total_photos, total_processed=total_processed)
display_progressbar(t1=10, t_orig=0, total_photos=total_photos, total_processed=total_photos)
print("\r\nDone!")

l = len(val_images)
t_orig = time.time()
total_photos = l
total_processed = 0
t = time.time()
print("Augmenting {} images in validation data...".format(total_photos))
for i in range(l):
    for j in range(NUM_AUGMENTATIONS_PER_IMAGE):
        img = (val_images[i]*255).astype(np.uint8)
        val_images.append(np.array(augmentor(image=img)['image'])/255)
        val_labels.append(val_labels[i])
    total_processed += 1
    t = display_progressbar(t1=t, t_orig=t_orig, total_photos=total_photos, total_processed=total_processed)
display_progressbar(t1=10, t_orig=0, total_photos=total_photos, total_processed=total_photos)
print("\r\nDone!")

print("Shuffling...")
(training_images, training_labels) = shuffle_arrays_in_sync(training_images, training_labels)
(test_images, test_labels) = shuffle_arrays_in_sync(test_images, test_labels)
(val_images, val_labels) = shuffle_arrays_in_sync(val_images, val_labels)
print("Converting to numpy arrays...")
training_images = np.array(training_images)
training_labels = np.array(training_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)
val_images = np.array(val_images)
val_labels = np.array(val_labels)
print("Done!")
print((training_images.shape, training_labels.shape))
print((test_images.shape, test_labels.shape))
print((val_images.shape, val_labels.shape))

model = models.Sequential()

# Block 1
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(MODEL_INPUT_IMAGE_DIMENSIONS[0], MODEL_INPUT_IMAGE_DIMENSIONS[1], 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Block 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Block 3
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Block 4
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten and Dense layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(training_images, training_labels, epochs=NUM_EPOCHS,
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss with Modified Data")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

model.save('model_final.h5')

tracemalloc.stop()