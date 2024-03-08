import tensorflow as tf
from face_tracking import face_detection

model = tf.keras.models.load_model('model_final_1.h5')

face_detection(model=model)
