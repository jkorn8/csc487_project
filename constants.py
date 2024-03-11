import os

MODEL_INPUT_IMAGE_DIMENSIONS = (100, 100)

# Filepaths
PATH_TO_DATA = os.path.join('..', 'data')
PATH_TO_PROCESSED_DATA = os.path.join(PATH_TO_DATA, 'data')
PATH_TO_FAILED_MODEL_EVALUATION_IMAGES = os.path.join(PATH_TO_DATA, 'failed')
CELEB_ORIG_PATH = os.path.join(PATH_TO_DATA, 'img_align_celeba')
DIGIFACE_ORIG_PATH = os.path.join(PATH_TO_DATA, 'subjects_0-1999_72_imgs')
FACES_PATH = os.path.join(PATH_TO_DATA, 'faces')
TESTING_PATH = os.path.join(PATH_TO_PROCESSED_DATA, 'testing')
VALIDATION_PATH = os.path.join(PATH_TO_PROCESSED_DATA, 'validation')
MODEL_NEGATIVE_FAILED_PATH = os.path.join(PATH_TO_FAILED_MODEL_EVALUATION_IMAGES, 'negative')
MODEL_POSITIVE_FAILED_PATH = os.path.join(PATH_TO_FAILED_MODEL_EVALUATION_IMAGES, 'positive')
UNKNOWN_FACES_PATH = os.path.join(FACES_PATH, 'unknown_faces')
KNOWN_FACE_1_PATH = os.path.join(FACES_PATH, 'stephen')
KNOWN_FACE_2_PATH = os.path.join(FACES_PATH, 'josh')
TESTING_PATH_0 = os.path.join(TESTING_PATH, '0')
TESTING_PATH_1 = os.path.join(TESTING_PATH, '1')
TESTING_PATH_2 = os.path.join(TESTING_PATH, '2')
VALIDATION_PATH_0 = os.path.join(VALIDATION_PATH, '0')
VALIDATION_PATH_1 = os.path.join(VALIDATION_PATH, '1')
VALIDATION_PATH_2 = os.path.join(VALIDATION_PATH, '2')
