import numpy as np
import sys
import time
import random
from keras.utils import load_img, img_to_array
from keras.applications import vgg19
import os
from deepface import DeepFace
from PIL import Image
from constants import PATH_TO_FACE_MATCHING


def get_progressbar(current, total):
    # Helper function for displaying the process current progress of the process
    # Input:
    #     current (int): current number of processed images
    #     total (int): total number of images to process
    # Output:
    #     A string progressbar displaying the current progress of the process
    progress = round((current/total)*30)
    if progress == 30:
        return '[==============================]'
    return '[' + progress * '=' + '>' + max((29 - progress), 0) * '-' + ']'


def get_minutes_from_seconds(secs):
    # Helper function for displaying the estimated amount of time remaining for process
    # Input:
    #     secs (int): number of seconds
    # Output:
    #     String format of number of minutes and seconds
    m = secs//60
    s = secs % 60
    if len(str(s)) == 2:
        return str(m) + ":" + str(s)
    return str(m) + ":0" + str(s)


def display_progressbar(t1, t_orig, total_photos, total_processed):
    # Helper function for displaying the progressbar and estimated time remaining for operation
    # Input:
    #     t1 (time): time since previous display of progressbar
    #     t_orig (time): time since start of processing data
    #     total_photos (int): total number of photos to process
    #     total_processed (int): total number of photos processed so far
    # Output:
    #     Either old time (if don't display progressbar) or new time (if display progressbar)
    t2 = time.time()
    if t2 - t1 > .5:
        sys.stdout.flush()
        est_seconds = round((total_photos / total_processed) * (t2 - t_orig) - (t2 - t_orig))
        sys.stdout.write('\r' + str(total_processed) + '/' + str(total_photos) + " "
                         + get_progressbar(total_processed, total_photos)
                         + " - ETA: " + get_minutes_from_seconds(est_seconds))
        return time.time()
    return t1


def get_distribution_array(size, percent_training, percent_validation):
    # Helper function that gives array filled with specified number of 1's in random indices
    # Input:
    #     size (int), size of array
    #     percent_distribution (float), percentage (as a decimal) of the array to fill with 1's
    # Output:
    #     Array of size "size", "percent_distribution" percent randomly filled with 1's, rest filled with 0's
    # Training is 0
    # Validation is 1
    if percent_training <= 0 or percent_validation <= 0 or percent_training + percent_validation != 100:
        raise ValueError('percent_training and percent_validation must add to 100')
    number_of_0s = int((size * percent_training) // 1)
    number_of_1s = int((size * percent_validation) // 1)
    arr = [0] * (number_of_0s) + [1] * (number_of_1s)
    while len(arr) < size:
        arr.append(0)
    random.shuffle(arr)
    return arr


'''
Function to shuffle two arrays "in the same way," used to shuffle images list while keeping labels in correct indices
Input:
    l1 (array): first array
    l2 (array): second array
Output:
    None
'''
def shuffle_arrays_in_sync(l1, l2):
    if len(l1) != len(l2):
        raise ValueError("Arrays must be same length")
    arr = []
    l1_final = [0]*len(l1)
    l2_final = [0]*len(l2)
    for i in range(len(l1)):
        arr.append(i)
    for i in range(len(l1)):
        j = random.randint(0, len(arr)-1)
        k = arr.pop(j)
        l1_final[i] = l1[k]
        l2_final[i] = l2[k]
    return l1_final, l2_final


def preprocess(input_path, dimensions):
    img = load_img(input_path, target_size=dimensions)
    img = img_to_array(img)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x):
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def get_max_image_index_for_known_images(known_faces_path):
    largest = 0
    for im in os.listdir(known_faces_path):
        if im.endswith('.jpg'):
            try:
                num = int(im.split('.')[0])
                if num > largest:
                    largest = num
            except ValueError:
                num = int((im.replace('_', '.')).split('.')[1])
                if num > largest:
                    largest = num
            except Exception as e:
                raise e
    return largest


def model_prediction(recoc_pred, match_pred, required_bound, certainty_bound):
    try:
        if match_pred != [None, None]:
            vals = [0, 0, 0]
            vals[0] = ((.75)*recoc_pred[0]) + ((.25)*match_pred[0])
            vals[1] = ((.75)*recoc_pred[1]) + ((.25)*match_pred[1])
            vals[2] = ((.75)*(recoc_pred[2])) + ((.25)*(2 - match_pred[0] - match_pred[1])/2)
        else:
            vals = recoc_pred
        s = vals.copy()
        s = sorted(s)
        s.reverse()
        if abs(s[0] - s[1]) < certainty_bound and s[0] > required_bound:
            return "unsure"
        for i in range(len(vals)-1):
            if vals[i] == s[0] and vals[i] > required_bound:
                return str(i)
        return "unknown"
    except:
        print("Failed")
        return "unknown"


def get_deepface_prediction(image, face_images):
    num_unique_faces = len(os.listdir(PATH_TO_FACE_MATCHING))
    totals = [0] * (num_unique_faces-1)
    validated = [0] * (num_unique_faces-1)
    for id, face in face_images:
        result = DeepFace.verify(face, image, model_name='Facenet', enforce_detection=False, detector_backend='opencv')
        if result['verified']:
            validated[id] += 1
        totals[id] += 1
    out = [0] * (num_unique_faces-1)
    for i in range((num_unique_faces-1)):
        out[i] = validated[i] / totals[i]
    return out


def do_data_preprocessing(num_images, input_dimensions,
                          percent_training, percent_validation,
                          known_face_1_path, known_face_2_path, unknown_face_path,
                          testing_path_0, testing_path_1, testing_path_2,
                          validation_path_0, validation_path_1, validation_path_2):

    known_face_1_dir = os.listdir(known_face_1_path)
    random.shuffle(known_face_1_dir)
    known_face_2_dir = os.listdir(known_face_2_path)
    random.shuffle(known_face_2_dir)
    unknown_faces_dir = os.listdir(unknown_face_path)
    random.shuffle(unknown_faces_dir)
    num_known_faces = num_images
    num_unknown_faces = min(round(num_images*(3)), len(unknown_faces_dir))

    process_known_face_1 = get_distribution_array(num_known_faces, percent_training=percent_training,
                                                percent_validation=percent_validation)
    process_known_face_2 = get_distribution_array(num_known_faces, percent_training=percent_training,
                                                  percent_validation=percent_validation)
    process_unknown_face = get_distribution_array(num_unknown_faces, percent_training=percent_training,
                                                  percent_validation=percent_validation)

    t_orig = time.time()
    total_photos = min(num_known_faces, len(known_face_1_dir))
    total_processed = 0
    t = time.time()
    print("Processing {} known faces in dir 0...".format(total_photos))
    for i in range(len(known_face_1_dir)):
        do_process = process_known_face_1[i]
        if do_process == 0:
            img = deprocess_image(
                preprocess(os.path.join(known_face_1_path, known_face_1_dir[i]), input_dimensions))
            img = Image.fromarray(img)
            img.save(os.path.join(testing_path_0, "known_" + known_face_1_dir[i]))
        else:
            img = deprocess_image(
                preprocess(os.path.join(known_face_1_path, known_face_1_dir[i]), input_dimensions))
            img = Image.fromarray(img)
            img.save(os.path.join(validation_path_0, "known_" + known_face_1_dir[i]))
        total_processed += 1
        t = display_progressbar(t1=t, t_orig=t_orig, total_photos=total_photos, total_processed=total_processed)
    display_progressbar(t1=10, t_orig=0, total_photos=total_photos, total_processed=total_photos)
    print("\r\nDone!")

    t_orig = time.time()
    total_photos = min(num_known_faces, len(known_face_2_dir))
    total_processed = 0
    t = time.time()
    print("Processing {} known faces in dir 1...".format(total_photos))
    for i in range(len(known_face_2_dir)):
        do_process = process_known_face_2[i]
        if do_process == 0:
            img = deprocess_image(
                preprocess(os.path.join(known_face_2_path, known_face_2_dir[i]), input_dimensions))
            img = Image.fromarray(img)
            img.save(os.path.join(testing_path_1, "known_" + known_face_2_dir[i]))
        else:
            img = deprocess_image(
                preprocess(os.path.join(known_face_2_path, known_face_2_dir[i]), input_dimensions))
            img = Image.fromarray(img)
            img.save(os.path.join(validation_path_1, "known_" + known_face_2_dir[i]))
        total_processed += 1
        t = display_progressbar(t1=t, t_orig=t_orig, total_photos=total_photos, total_processed=total_processed)
    display_progressbar(t1=10, t_orig=0, total_photos=total_photos, total_processed=total_photos)
    print("\r\nDone!")

    t_orig = time.time()
    total_photos = num_unknown_faces
    total_processed = 0
    t = time.time()
    print("Processing {} unknown faces...".format(total_photos))
    for i in range(len(unknown_faces_dir)):
        if i > num_unknown_faces:
            break
        if process_unknown_face[i] == 0:
            img = deprocess_image(
                preprocess(os.path.join(unknown_face_path, unknown_faces_dir[i]), input_dimensions))
            img = Image.fromarray(img)
            img.save(os.path.join(testing_path_2, "unknown_" + unknown_faces_dir[i]))
        else:
            img = deprocess_image(
                preprocess(os.path.join(unknown_face_path, unknown_faces_dir[i]), input_dimensions))
            img = Image.fromarray(img)
            img.save(os.path.join(validation_path_2, "unknown_" + unknown_faces_dir[i]))
        total_processed += 1
        t = display_progressbar(t1=t, t_orig=t_orig, total_photos=total_photos, total_processed=total_processed)
    display_progressbar(t1=10, t_orig=0, total_photos=total_photos, total_processed=total_photos)
    print("\r\nDone!")