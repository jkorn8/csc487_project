import os

MODEL_INPUT_IMAGE_DIMENSIONS = (100, 100)

# Filepaths
TESTING_PATH = os.path.join('data', 'data', 'testing')
VALIDATION_PATH = os.path.join('data', 'data', 'validation')
CELEB_ORIG_PATH = os.path.join('data', 'img_align_celeba')
DIGIFACE_ORIG_PATH = os.path.join('data', 'subjects_0-1999_72_imgs')
FACES_PATH = os.path.join('data', 'faces')
UNKNOWN_FACES_PATH = os.path.join(FACES_PATH, 'unknown_faces')
KNOWN_FACE_PATH = os.path.join(FACES_PATH, 'stephen')   # Change path for different identities
TESTING_PATH_POSITIVE = os.path.join(TESTING_PATH, 'positive')
TESTING_PATH_NEGATIVE = os.path.join(TESTING_PATH, 'negative')
VALIDATION_PATH_POSITIVE = os.path.join(VALIDATION_PATH, 'positive')
VALIDATION_PATH_NEGATIVE = os.path.join(VALIDATION_PATH, 'negative')

