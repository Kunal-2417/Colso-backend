import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Class labels
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
          "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Normalize pixel values to 0-1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshape to match CNN input: (28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Build improved CNN model
model = Sequential([
    # First convolutional block
    Conv2D(32, (3,3), activation='relu', kernel_regularizer=l2(0.001), 
           input_shape=(28,28,1), padding='same'),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', kernel_regularizer=l2(0.001), padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    # Second convolutional block
    Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.001), padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.001), padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    # Third convolutional block
    Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(0.001), padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    # Fully connected layers
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Compile model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Model summary
model.summary()

# Train model
history = model.fit(x_train, y_train_cat, 
                   epochs=50, 
                   batch_size=64,
                   validation_data=(x_test, y_test_cat),
                   callbacks=[early_stopping, reduce_lr])

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test_cat)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save model
model.save("fashion_model_improved.h5")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()