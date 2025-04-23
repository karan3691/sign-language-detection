import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mediapipe as mp

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 9  # Update this based on your dataset
SIGN_CLASSES = ['hello', 'thank_you', 'yes', 'no', 'please', 'sorry', 'i_love_you', 'help', 'house']
DATASET_PATH = "sign_language_dataset"  # Update with your dataset path
MODEL_PATH = 'sign_language_model.h5'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def create_dataset():
    """
    Process images in the dataset to extract hand landmarks using MediaPipe
    and prepare training data.
    """
    X = []
    y = []
    
    # Initialize MediaPipe hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3  # Lower detection confidence to catch more hands
    )
    
    print("Processing dataset...")
    
    total_images = 0
    processed_images = 0
    
    # Loop through each class folder
    for class_idx, class_name in enumerate(SIGN_CLASSES):
        class_path = os.path.join(DATASET_PATH, class_name)
        
        if not os.path.exists(class_path):
            print(f"Warning: Path {class_path} does not exist. Skipping.")
            continue
        
        print(f"Processing class: {class_name}")
        
        class_images = 0
        class_processed = 0
        
        # Loop through each image in the class folder
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            
            total_images += 1
            class_images += 1
            
            # Read and process image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}. Skipping.")
                continue
            
            # Convert to RGB for MediaPipe
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = hands.process(rgb_img)
            
            # Create a blank image to draw hand landmarks
            h, w, c = img.shape
            hand_img = np.zeros((h, w, c), dtype=np.uint8)
            
            # If hand landmarks detected, draw them
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        hand_img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                
                # Resize image to match model input size
                processed_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                
                # Add to dataset
                X.append(processed_img)
                y.append(class_idx)
                processed_images += 1
                class_processed += 1
            else:
                # Fallback: If no hand landmarks detected, use the original image directly
                # This is useful for images that already contain hand landmarks drawn
                print(f"No landmarks detected in {img_path}, using original image instead.")
                processed_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(processed_img)
                y.append(class_idx)
                processed_images += 1
                class_processed += 1
        
        print(f"  Class {class_name}: Processed {class_processed}/{class_images} images")
    
    print(f"Total: Processed {processed_images}/{total_images} images")
    
    # Close the MediaPipe hands session
    hands.close()
    
    # Convert lists to numpy arrays
    X = np.array(X, dtype='float32')
    y = np.array(y)
    
    # Normalize pixel values to be between 0 and 1
    X = X / 255.0
    
    # One-hot encode the labels
    y = to_categorical(y, num_classes=NUM_CLASSES)
    
    return X, y

def build_model():
    """
    Build a CNN model for sign language classification.
    """
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D((2, 2)),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Fourth convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Flatten and fully connected layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),  # Add dropout to reduce overfitting
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss.
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def train_model():
    """
    Main function to create dataset, build and train the model.
    """
    # Check if dataset directory exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset directory {DATASET_PATH} not found.")
        print("Please organize your dataset with one folder per sign class.")
        return
    
    # Create dataset
    X, y = create_dataset()
    
    # If no images were processed, exit
    if len(X) == 0:
        print("Error: No valid images were processed. Check your dataset.")
        return
    
    print(f"Dataset created: {len(X)} images with shape {X.shape}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Build the model
    model = build_model()
    model.summary()
    
    # Callbacks for training
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    print("Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stopping]
    )
    
    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.2f}")
    print(f"Test loss: {test_loss:.2f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Save the model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    # Check TensorFlow GPU availability
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"TensorFlow is using GPU: {physical_devices}")
    else:
        print("TensorFlow is using CPU. Training may be slower.")
    
    train_model() 