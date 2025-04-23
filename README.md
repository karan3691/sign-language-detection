# Real-time Sign Language Detection System

A computer vision-based system that detects and interprets sign language gestures in real-time, using OpenCV for video processing, MediaPipe for hand landmark detection, and a custom CNN model trained with TensorFlow and Keras.

## Features

- Real-time hand gesture detection and recognition
- Support for 9 common signs: hello, thank you, yes, no, please, sorry, I love you, help, house
- Low-latency processing suitable for real-time applications
- Custom data collection tool for expanding the training dataset
- CNN-based model with 92% accuracy

## System Requirements

- Python 3.7+
- Webcam
- GPU recommended for training (CPU sufficient for inference)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/karan3691/sign-language-detection.git
   cd sign-language-detection
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Important Note About the Model File

**The trained model file (`sign_language_model.h5`) is not included in this repository** due to GitHub's file size limitations. You need to generate your own model by following these steps:

1. Collect training data using the provided tool
2. Train the model using the training script

This process is detailed in the Quick Start section below.

## Quick Start

1. **Collect training data** for each sign class:
   ```
   python collect_data.py --sign hello --samples 100
   python collect_data.py --sign thank_you --samples 100
   python collect_data.py --sign yes --samples 100
   python collect_data.py --sign no --samples 100
   python collect_data.py --sign please --samples 100
   python collect_data.py --sign sorry --samples 100
   python collect_data.py --sign i_love_you --samples 100
   python collect_data.py --sign help --samples 100
   python collect_data.py --sign house --samples 100
   ```
   
   You can adjust the number of samples as needed. More samples generally lead to better accuracy.

2. **Train the model**:
   ```
   python train_model.py
   ```
   This will process the collected data, train the CNN model, and save it as `sign_language_model.h5`.

3. **Run the detection system**:
   ```
   python Sign_language_detection.py
   ```
   This will start real-time sign language detection using your webcam.

## Usage Tips

- Press 'p' while the detection is running to toggle the visualization of the hand landmarks used as model input
- Press 'q' to quit the application
- If your signs aren't being detected well, try collecting more training data or ensuring better lighting conditions

## Project Structure

- `Sign_language_detection.py`: Main script for real-time sign language detection
- `train_model.py`: Script for training the CNN model on collected data
- `collect_data.py`: Tool for collecting training data using a webcam
- `sign_language_dataset/`: Directory for storing training data
- `sign_language_model.h5`: Trained model file (created after training)

## How It Works

1. **Hand Detection**: MediaPipe is used to detect hand landmarks in real-time video frames.
2. **Feature Extraction**: The detected hand landmarks are drawn on a blank image to create a standardized representation.
3. **Classification**: A CNN model processes the hand landmark images and classifies them into one of the predefined sign classes.
4. **Smooth Prediction**: A simple temporal smoothing algorithm is applied to reduce flickering in predictions.

## Model Architecture

The system uses a CNN with the following architecture:
- 4 convolutional blocks with increasing filter sizes (32, 64, 128, 128)
- Max pooling after each convolutional layer
- Dense layer with 512 units and dropout for regularization
- Softmax output layer for classification

During training, the model achieves approximately 92% accuracy on the test set after 10 epochs when trained with sufficient data.

## Extending the System

To add support for additional signs:

1. Add the new sign class to the `SIGN_CLASSES` list in all three Python files.
2. Collect training data for the new sign using `collect_data.py`.
3. Retrain the model using `train_model.py`.

## Requirements

```
tensorflow>=2.5.0
opencv-python>=4.5.3
mediapipe>=0.8.7
numpy>=1.19.5
matplotlib>=3.4.3
scikit-learn>=0.24.2
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The MediaPipe team for their excellent hand tracking solution
- The TensorFlow and Keras teams for their deep learning frameworks 