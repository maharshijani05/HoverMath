import os
import sys
import json
import shutil
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configs
IMG_SIZE = 45
BATCH_SIZE = 64
EPOCHS = 10
DATASET_DIR = 'extracted_images'
TRAIN_DIR = 'temp_train'
VAL_DIR = 'temp_val'

# Step 1: Split dataset into train/val
def split_data():
    if os.path.exists(TRAIN_DIR) or os.path.exists(VAL_DIR):
        print("[INFO] Train/Val directories already exist. Skipping split.")
        return

    print("[INFO] Splitting data into training and validation sets...")
    os.makedirs(TRAIN_DIR)
    os.makedirs(VAL_DIR)

    for class_name in os.listdir(DATASET_DIR):
        class_path = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_path): continue

        images = os.listdir(class_path)
        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

        os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, class_name), exist_ok=True)

        for img in train_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(TRAIN_DIR, class_name, img))
        for img in val_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(VAL_DIR, class_name, img))

split_data()

# Step 2: Image generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Step 3: CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(38, activation='softmax')  # 38 classes
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

# Custom progress bar callback
class ProgressBar(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        total = self.params['steps']
        progress = (batch + 1) / total
        bar_len = 30
        filled = int(progress * bar_len)
        bar = '=' * filled + '-' * (bar_len - filled)
        percent = int(progress * 100)
        sys.stdout.write(f'\rEpoch {self.epoch_num + 1}/{EPOCHS} | [{bar}] {percent}%')
        sys.stdout.flush()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_num = epoch
        print()

    def on_epoch_end(self, epoch, logs=None):
        print()

progress_bar = ProgressBar()

# Step 5: Train model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[progress_bar, early_stop, checkpoint]
)

# Step 6: Save final model and class labels
model.save('hovermath_cnn_model.h5')
print("Final model saved as hovermath_cnn_model.h5")

with open('label_map.json', 'w') as f:
    json.dump(train_gen.class_indices, f)
print("Label map saved as label_map.json")
