from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from constants import MODEL_INPUT_IMAGE_DIMENSIONS, TESTING_PATH, VALIDATION_PATH, MODEL_PATH
import tensorflow.keras.callbacks as cb
import os


NUM_EPOCHS = 50

datagen = ImageDataGenerator()
train_generator = datagen.flow_from_directory(
    TESTING_PATH,
    target_size=MODEL_INPUT_IMAGE_DIMENSIONS,
    class_mode='categorical',
    shuffle=True
)
val_generator = datagen.flow_from_directory(
    VALIDATION_PATH,
    target_size=MODEL_INPUT_IMAGE_DIMENSIONS,
    class_mode='categorical',
    shuffle=True
)

model = models.Sequential()

# Block 1
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(MODEL_INPUT_IMAGE_DIMENSIONS[0], MODEL_INPUT_IMAGE_DIMENSIONS[1], 3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

# Block 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

# Block 3
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

# Block 4
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

# Flatten and Dense layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='sigmoid'))

reduce_lr = cb.ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,  # Factor by which the learning rate will be reduced
                              patience=4,  # Number of epochs with no improvement after which learning rate will be reduced
                              verbose=1)
model_checkpoint = cb.ModelCheckpoint(os.path.join(MODEL_PATH, 'model_best_with_josh_3.h5'),
                                      monitor='val_accuracy',  # Choose the metric to monitor (e.g., val_loss, val_accuracy)
                                      mode='max',  # 'max' if monitoring accuracy, 'min' if monitoring loss
                                      save_best_only=True,  # Save only the best model
                                      verbose=1)
csv_logger = cb.CSVLogger('training.log')


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train_generator, epochs=NUM_EPOCHS,
                    validation_data=val_generator,
                    callbacks=[reduce_lr, model_checkpoint, csv_logger])

test_loss, test_acc = model.evaluate(val_generator, verbose=2)
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

model.save(os.path.join(MODEL_PATH, 'model_final_with_josh_3.h5'))
