import os
import numpy as np
import imageio.v3 as iio
import sys
import time
import tensorflow as tf
import random

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
def get_distribution_array(size, percent_distribution):
    if percent_distribution < 0 or percent_distribution > 1:
        raise ValueError('percent_distribution must be between 0 and 1')
    number_of_1s = int((size * percent_distribution) // 1)
    arr = [1] * number_of_1s + [0] * (size-number_of_1s)
    random.shuffle(arr)
    return arr


'''
Function to augment data to add data reflection and/or rotations. Adds this augmented data to the input arrays
Input:
    Images (image tensor list): array of images you wish to augment
    labels (integer list): array of labels for the images
    mirror (bool): whether to do mirror augmentations
    num_rotate (int): non-negative number of rotation augmentations you wish to do per image
Output:
    None
'''
def augment_data(images, labels, mirror=True, num_rotate=1):
    t1 = time.time()
    t_orig = time.time()
    if mirror:
        total_photos = len(images)
        total_processed = 0
        print("Mirroring images...")
        for i in range(len(images)):
            images.append(tf.image.flip_left_right(images[i]))
            labels.append(labels[i])
            total_processed += 1
            t1 = display_progressbar(t1, t_orig, total_photos, total_processed)
        sys.stdout.flush()
        sys.stdout.write("\r")
        print(str(total_processed) + '/' + str(total_photos) + " "
              + get_progressbar(total_processed, total_photos)
              + " - ETA: " + get_minutes_from_seconds(0))
        print("Done mirroring images!")

    if num_rotate > 0:
        rotate_image = tf.keras.layers.RandomRotation(0.05)
        total_photos = len(images)
        total_processed = 0
        print("Rotating images...")
        for i in range(len(images)):
            for j in range(num_rotate):
                images.append(rotate_image(images[i]))
                labels.append(labels[i])
            total_processed += 1
            t1 = display_progressbar(t1, t_orig, total_photos, total_processed)
        sys.stdout.flush()
        sys.stdout.write("\r")
        print(str(total_processed) + '/' + str(total_photos) + " "
              + get_progressbar(total_processed, total_photos)
              + " - ETA: " + get_minutes_from_seconds(0))
        print("Done rotating images!")


'''
Function to shuffle two arrays "in the same way," used to shuffle images list while keeping labels in correct indices
Input:
    l1 (array): first array
    l2 (array): second array
Output:
    None
'''
def shuffle_arrays(l1, l2):
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


'''
Function that retrieves images and processes them as numpy vectors
Input: num_faces (int): number of total identities you wish to process;
    num_unknown (int): number of identities you wish to label as "unknown" (-1);
    percent_training_data (float): percent of data you wish to be training data;
    distribution (string): distribution of training data, must be either "even" or "random"
        if "even", then use every identity, but only use "percent_training_data" percent of photos from each person for training data
        if "random", then randomly select photos from pool of all photos for training data
    add_mirrors (bool): whether augment data by adding mirrored images
    num_rotations (int): non-negative number of augmented data rotations of each image to add to data
Output:

WARNING: Having too much data causes SIGKILL.
'''
def get_data(num_faces=10000, num_unknown=0, percent_training_data=10.0, 
             distribution="even", add_mirrors=True, num_rotations=1):

    # Check for valid input
    if not isinstance(num_faces, int):
        raise ValueError('num_faces must be an integer')
    if not isinstance(num_unknown, int):
        raise ValueError('num_unknown must be an integer')
    if not (isinstance(percent_training_data, float) or isinstance(percent_training_data, int)):
        raise ValueError('percent_training_data must be an float')
    if distribution not in ["even", "random"]:
        raise ValueError('distribution must be "even" or "random"')
    if num_faces < 1 or num_faces > 10000:
        raise ValueError('num_faces must be between 1 and 10000')
    if percent_training_data > 100 or percent_training_data < 0:
        raise ValueError('percent_training_data must be between 0 and 100')
    if num_faces <= num_unknown:
        raise ValueError("Number of total faces must be greater than number of unknown faces")
    if not (isinstance(add_mirrors, bool)):
        raise ValueError('add_mirrors must be True or False')
    if not (isinstance(num_rotations, int)):
        raise ValueError('num_rotations must be an integer')
    if num_rotations < 0:
        raise ValueError('num_rotations must be 0 or greater')

    # Initialize variables
    test_images = []
    test_labels = []
    training_images = []
    training_labels = []
    do_process_photo = []
    t1 = time.time()
    t_orig = time.time()
    total_photos = 72*num_faces   # Should edit if adding/removing photos
    total_processed = 0
    total_people_processed = 0
    j = 0
    identity_mapper = dict()
    if num_unknown == 0:
        size_of_last_layer = num_faces
    else:
        size_of_last_layer = num_faces - num_unknown + 1
    if distribution == "random":
        do_process_photo = get_distribution_array(num_faces*72, 1 - percent_training_data/100)

    if not (os.path.exists('photos') and os.path.isdir('photos')):
        raise FileNotFoundError('Error: Missing "photos" directory')

    print("Beginning processing of images:")

    print("Retrieving images...")
    sys.stdout.write('\r' + str(total_processed) + '/' + str(total_photos)
                     + " " + get_progressbar(total_processed, total_photos))

    # Note: photos directory should hold directories that include directories that include pngs
    #   photos - person 0 - photo 0
    #      |         |------ photo 1
    #      |         ...
    #      |         |------ photo k-1
    #      |----- person 1
    #      ...
    #      |----- person m-1
    identities = os.listdir('photos')
    random.shuffle(identities)
    for i in range(len(identities)):
        if os.path.isdir('photos/' + identities[i]):
            identity_mapper[i] = identities[i]
            if distribution == "even":
                do_process_photo = get_distribution_array(72, 1 - percent_training_data / 100)
            for photo in os.listdir('photos/' + identities[i]):
                if photo.endswith('.png'):
                    if do_process_photo[j] == 1:
                        test_images.append(iio.imread('photos/' + identities[i] + '/' + photo)[:, :, :3])  # Removing transparency layer
                        if num_faces - num_unknown <= total_people_processed:
                            test_labels.append(num_faces - num_unknown)
                        else:
                            test_labels.append(i)
                    else:
                        training_images.append(iio.imread('photos/' + identities[i] + '/' + photo)[:, :, :3])  # Removing transparency layer
                        if num_faces - num_unknown <= total_people_processed:
                            training_labels.append(num_faces - num_unknown)
                        else:
                            training_labels.append(i)
                    total_processed += 1
                    j += 1
                t1 = display_progressbar(t1, t_orig, total_photos, total_processed)
            if distribution == "even":
                j = 0
            total_people_processed += 1

        # Lol this break chain is scuffed but whatever, just need a way to exit once done
        if total_people_processed >= num_faces:
            break

    # Cleanup and get output
    sys.stdout.flush()
    sys.stdout.write("\r")
    print(str(total_processed) + '/' + str(total_photos) + " "
          + get_progressbar(total_processed, total_photos)
          + " - ETA: " + get_minutes_from_seconds(0))
    print("Done Retrieving images!")
    del do_process_photo, t1, t_orig, total_photos, total_processed, total_people_processed, j

    if add_mirrors or (num_rotations == 1):
        augment_data(test_images, test_labels, add_mirrors, num_rotations)
        # augment_data(training_images, training_labels, add_mirrors, num_rotations)
        # This line adds data augmentations to training data too, which we may or may not want to do

    test_images, test_labels = shuffle_arrays(test_images, test_labels)
    training_images, training_labels = shuffle_arrays(training_images, training_labels)
    test_images, test_labels = np.array(test_images), np.array(test_labels)
    training_images, training_labels = np.array(training_images), np.array(training_labels)
    print("Done Processing images!")
    return (test_images, test_labels), (training_images, training_labels), identity_mapper, size_of_last_layer
