import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Constants
IMG_SIZE = 224
SIGN_CLASSES = ['hello', 'thank_you', 'yes', 'no', 'please', 'sorry', 'i_love_you', 'help', 'house']
MODEL_PATH = 'sign_language_model.h5'

class SignLanguageDetector:
    def __init__(self):
        # Load the trained model if it exists
        if os.path.exists(MODEL_PATH):
            self.model = load_model(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        else:
            self.model = None
            print(f"No model found at {MODEL_PATH}. Run train_model.py first.")
        
        # Initialize MediaPipe Hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # Performance metrics
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        # Debug options
        self.show_processed_image = False
    
    def preprocess_frame(self, frame):
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Create a blank image to draw hand landmarks only
        h, w, c = frame.shape
        hand_image = np.zeros((h, w, c), dtype=np.uint8)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    hand_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
        # Resize the hand image for model input
        processed_img = cv2.resize(hand_image, (IMG_SIZE, IMG_SIZE))
        processed_img = processed_img / 255.0  # Normalize
        
        return processed_img, results, frame
    
    def predict_sign(self, processed_img, results):
        if self.model is None:
            return None, 0, None
        
        # Always make a prediction even if no landmarks detected
        # This helps with the images we've processed as fallback
        img_for_prediction = np.expand_dims(processed_img, axis=0)
        
        # Make prediction
        prediction = self.model.predict(img_for_prediction)
        predicted_class_idx = np.argmax(prediction)
        confidence = prediction[0][predicted_class_idx]
        
        # Return all predictions with confidence, but use a lower threshold
        if confidence > 0.30:  # Lowered threshold to catch more predictions
            return SIGN_CLASSES[predicted_class_idx], confidence, prediction
        
        return None, confidence, prediction
    
    def calculate_fps(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        # Update FPS every second
        if elapsed_time > 1:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
            
        return self.fps
    
    def run_detection(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Variables for smoothing predictions
        prediction_history = []
        MAX_HISTORY = 5
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # Mirror the frame horizontally for a more intuitive view
            frame = cv2.flip(frame, 1)
            
            # Process the frame
            processed_img, results, annotated_frame = self.preprocess_frame(frame)
            
            # Get prediction
            predicted_sign, confidence, predictions = self.predict_sign(processed_img, results)
            
            # Show the prediction confidence even if below threshold (for debugging)
            if confidence > 0:
                top_idx = np.argsort(predictions[0])[-3:][::-1]  # Get top 3 predictions
                prediction_text = []
                for idx in top_idx:
                    cls_name = SIGN_CLASSES[idx]
                    cls_conf = predictions[0][idx]
                    if cls_conf > 0.10:  # Only show if above 10%
                        prediction_text.append(f"{cls_name}: {cls_conf:.2f}")
                
                # Display top predictions for debugging
                y_pos = 30
                for text in prediction_text:
                    cv2.putText(
                        annotated_frame,
                        text,
                        (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )
                    y_pos += 30
            
            # Smooth predictions using a simple history-based approach
            if predicted_sign:
                prediction_history.append(predicted_sign)
                if len(prediction_history) > MAX_HISTORY:
                    prediction_history.pop(0)
                
                # Get the most common prediction from history
                if prediction_history:
                    from collections import Counter
                    most_common = Counter(prediction_history).most_common(1)
                    smoothed_prediction = most_common[0][0]
                    
                    # Display the final prediction on the frame
                    cv2.putText(
                        annotated_frame, 
                        f"DETECTED: {smoothed_prediction.replace('_', ' ')}",
                        (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )
            
            # Calculate and display FPS
            fps = self.calculate_fps()
            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.2f}",
                (500, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            
            # Display the resulting frame
            cv2.imshow('Sign Language Detection', annotated_frame)
            
            # Display the processed hand landmarks image if enabled
            if self.show_processed_image:
                # Convert back to 0-255 range for visualization
                vis_img = (processed_img * 255).astype(np.uint8)
                cv2.imshow('Hand Landmarks (Model Input)', vis_img)
            
            # Toggle processed image view with 'p' key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.show_processed_image = not self.show_processed_image
                if not self.show_processed_image:
                    cv2.destroyWindow('Hand Landmarks (Model Input)')
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    detector = SignLanguageDetector()
    detector.run_detection() 