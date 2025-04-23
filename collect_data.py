import cv2
import os
import numpy as np
import mediapipe as mp
import time
import argparse

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Constants
DATASET_PATH = "sign_language_dataset"
IMG_SIZE = 224
SIGN_CLASSES = ['hello', 'thank_you', 'yes', 'no', 'please', 'sorry', 'i_love_you', 'help', 'house']

def create_directory_structure():
    """Create directory structure for the dataset."""
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
        print(f"Created main dataset directory: {DATASET_PATH}")
    
    # Create subdirectories for each sign class
    for class_name in SIGN_CLASSES:
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)
            print(f"Created directory for {class_name}")

def collect_data(sign_class, num_samples=100, delay=30):
    """
    Collect training data for a specific sign class using webcam.
    
    Args:
        sign_class: The sign class to collect data for
        num_samples: Number of images to collect
        delay: Delay between captures in frames (at 30fps, delay=30 means ~1 second)
    """
    if sign_class not in SIGN_CLASSES:
        print(f"Error: {sign_class} is not in the list of sign classes.")
        print(f"Available classes: {SIGN_CLASSES}")
        return
    
    # Create class directory if it doesn't exist
    class_path = os.path.join(DATASET_PATH, sign_class)
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    
    # Count existing samples
    existing_samples = len([name for name in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, name))])
    print(f"Found {existing_samples} existing samples for {sign_class}")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize MediaPipe Hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Variables for tracking
    samples_collected = 0
    frame_count = 0
    countdown_active = False
    countdown_start = 0
    countdown_duration = 3  # seconds
    
    print(f"Starting data collection for {sign_class}...")
    print("Press 'c' to start/stop countdown for collection")
    print("Press 'q' to quit")
    
    while cap.isOpened() and samples_collected < num_samples:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Mirror the frame horizontally for a more intuitive view
        frame = cv2.flip(frame, 1)
        
        # Convert the frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)
        
        # Create a copy for visualization
        vis_frame = frame.copy()
        
        # Draw hand landmarks on the visualization frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    vis_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Display collection status
        status_text = f"Class: {sign_class} | Collected: {samples_collected}/{num_samples}"
        cv2.putText(vis_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Handle countdown
        if countdown_active:
            elapsed = time.time() - countdown_start
            if elapsed < countdown_duration:
                # Display countdown
                seconds_left = int(countdown_duration - elapsed)
                cv2.putText(vis_frame, f"Starting in {seconds_left}...", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Countdown finished, start collection
                countdown_active = False
                collecting = True
                frame_count = 0
                cv2.putText(vis_frame, "Collecting...", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Create a blank image with only hand landmarks for storage
                h, w, c = frame.shape
                hand_img = np.zeros((h, w, c), dtype=np.uint8)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            hand_img,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                    # Save the image
                    img_filename = f"{sign_class}_{existing_samples + samples_collected + 1}.jpg"
                    img_path = os.path.join(class_path, img_filename)
                    
                    # Resize image
                    hand_img_resized = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                    cv2.imwrite(img_path, hand_img_resized)
                    
                    samples_collected += 1
                    print(f"Saved sample {samples_collected}/{num_samples}")
                else:
                    cv2.putText(vis_frame, "No hand detected!", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(vis_frame, "Press 'c' to start capture", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Show the frame
        cv2.imshow('Sign Language Data Collection', vis_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and not countdown_active:
            countdown_active = True
            countdown_start = time.time()
            print("Starting countdown...")
        
        # Increment frame count for delay
        frame_count += 1
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print(f"Data collection complete. Collected {samples_collected} samples for {sign_class}.")

def main():
    parser = argparse.ArgumentParser(description="Collect sign language training data")
    parser.add_argument("--sign", type=str, help="Sign class to collect data for", required=True)
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to collect")
    
    args = parser.parse_args()
    
    # Create directory structure
    create_directory_structure()
    
    # Collect data for the specified sign
    collect_data(args.sign, args.samples)

if __name__ == "__main__":
    main() 