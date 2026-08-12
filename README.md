# Drowsiness Detection Using Eye Aspect Ratio (EAR)

## Project Overview

Drowsiness Detection is a real-time computer vision project designed to detect signs of driver fatigue. The system uses a webcam to monitor the driver's eyes, detects facial landmarks using **Dlib**, and calculates the **Eye Aspect Ratio (EAR)** using Python and OpenCV.

When the eyes remain closed for a predefined number of consecutive frames, the system identifies possible drowsiness and displays an alert on the screen.

## Objectives

* Detect driver drowsiness in real time.
* Monitor eye closure using facial landmarks.
* Calculate Eye Aspect Ratio (EAR).
* Provide an alert when prolonged eye closure is detected.
* Demonstrate the application of Computer Vision in automotive safety.

## Technologies Used

* **Python**
* **OpenCV**
* **Dlib**
* **NumPy**
* **SciPy**
* **Computer Vision**

## How It Works

1. The webcam captures the driver's video.
2. Dlib detects the driver's face.
3. Facial landmarks are identified using the 68-point landmark model.
4. The coordinates of both eyes are extracted.
5. The Eye Aspect Ratio (EAR) is calculated.
6. If the EAR remains below a predefined threshold for consecutive frames, drowsiness is detected.
7. A visual warning is displayed on the screen.

## Eye Aspect Ratio (EAR)

The Eye Aspect Ratio is used to measure the level of eye openness.

```text
EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
```

A higher EAR generally indicates an open eye, while a lower EAR indicates that the eye is closing.

## Project Structure

```text
drowsiness-detection-eye-aspect-ratio/
│
├── drowsiness_detection.py
├── requirements.txt
├── README.md
├── shape_predictor_68_face_landmarks.dat
└── screenshots/
    └── demo.png
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/drowsiness-detection-eye-aspect-ratio.git
```

Navigate to the project folder:

```bash
cd drowsiness-detection-eye-aspect-ratio
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Required Dependencies

The `requirements.txt` file contains:

```text
opencv-python
dlib
numpy
scipy
```

## Running the Project

Make sure your webcam is connected and run:

```bash
python drowsiness_detection.py
```

Press **Q** to close the application.

## Key Features

* Real-time webcam monitoring
* Face detection
* 68-point facial landmark detection
* Eye landmark tracking
* EAR calculation
* Drowsiness detection
* Visual warning system

## Applications

This project can be applied to:

* Driver monitoring systems
* Automotive safety applications
* Fatigue monitoring
* Real-time computer vision systems
* Intelligent transportation systems

## Future Enhancements

* Add an audio alarm for drowsiness alerts.
* Add yawning detection.
* Implement head-pose estimation.
* Improve detection using machine learning/deep learning.
* Develop a graphical user interface.
* Integrate the system with an automotive embedded platform.

## Author

**Shreya**

Electronics and Communication Engineering
Interested in AI/ML, Computer Vision, and Automotive Technologies.
