import numpy as np
import sys
import time
import random
from keras.utils import load_img, img_to_array
from keras.applications import vgg19

'''
Helper function for displaying the process current progress of the process
Input: 
    current (int): current number of processed images
    total (int): total number of images to process
Output: 
    A string progressbar displaying the current progress of the process
'''
def get_progressbar(current, total):
    progress = round((current/total)*30)
    if progress == 30:
        return '[==============================]'
    return '[' + progress * '=' + '>' + max((29 - progress), 0) * '-' + ']'


'''
Helper function for displaying the estimated amount of time remaining for process
Input: 
    secs (int): number of seconds
Output: 
    String format of number of minutes and seconds
'''
def get_minutes_from_seconds(secs):
    m = secs//60
    s = secs % 60
    if len(str(s)) == 2:
        return str(m) + ":" + str(s)
    return str(m) + ":0" + str(s)


'''
Helper function for displaying the progressbar and estimated time remaining for operation
Input:
    t1 (time): time since previous display of progressbar
    t_orig (time): time since start of processing data
    total_photos (int): total number of photos to process
    total_processed (int): total number of photos processed so far
Output:
    Either old time (if don't display progressbar) or new time (if display progressbar)
    
'''
def display_progressbar(t1, t_orig, total_photos, total_processed):
    t2 = time.time()
    if t2 - t1 > .5:
        sys.stdout.flush()
        est_seconds = round((total_photos / total_processed) * (t2 - t_orig) - (t2 - t_orig))
        sys.stdout.write('\r' + str(total_processed) + '/' + str(total_photos) + " "
                         + get_progressbar(total_processed, total_photos)
                         + " - ETA: " + get_minutes_from_seconds(est_seconds))
        return time.time()
    return t1


'''
Helper function that gives array filled with specified number of 1's in random indices
Input: 
    size (int), size of array
    percent_distribution (float), percentage (as a decimal) of the array to fill with 1's
Output: 
    Array of size "size", "percent_distribution" percent randomly filled with 1's, rest filled with 0's
'''
def get_distribution_array(size, percent_training, percent_validation, percent_testing):
    # Training is 0
    # Validation is 1
    # Testing is 2
    if percent_training <= 0 or percent_validation <= 0 or percent_testing <= 0 or percent_training + percent_validation + percent_testing > 100:
        raise ValueError('percent_training, percent_validation, and percent_testing must add to 100')
    number_of_0s = int((size * percent_training) // 1)
    number_of_1s = int((size * percent_validation) // 1)
    number_of_2s = int((size * percent_testing) // 1)
    arr = [0] * (number_of_0s) + [1] * (number_of_1s) + [2] * number_of_2s
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
    x = x/255
    return x
