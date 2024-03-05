import os
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

TESTING_PATH = os.path.join('data', 'data', 'testing')
VALIDATION_PATH = os.path.join('data', 'data', 'validation')

MODEL_INPUT_IMAGE_DIMENSIONS = (100, 100)
NUM_EPOCHS = 50

datagen = ImageDataGenerator()
train_generator = datagen.flow_from_directory(
    TESTING_PATH,
    target_size=(100, 100),
    class_mode='binary'  # Automatically inferred based on the subdirectories
)
val_generator = datagen.flow_from_directory(
    VALIDATION_PATH,
    target_size=(100, 100),
    class_mode='binary',  # Use 'binary' for binary classification or 'categorical' for multiclass
    shuffle=False  # Keep the order of data to evaluate metrics accurately
)

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

history = model.fit(train_generator, epochs=NUM_EPOCHS,
                    validation_data=val_generator)

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

model.save('model_final.h5')