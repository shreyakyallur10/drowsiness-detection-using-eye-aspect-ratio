import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import time

# -----------------------------
# Configuration
# -----------------------------

EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 20

# -----------------------------
# Initialize Dlib
# -----------------------------

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)

# Eye landmark indexes
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))


# -----------------------------
# Calculate Eye Aspect Ratio
# -----------------------------

def calculate_ear(eye):

    # Vertical eye distances
    vertical_1 = distance.euclidean(eye[1], eye[5])
    vertical_2 = distance.euclidean(eye[2], eye[4])

    # Horizontal eye distance
    horizontal = distance.euclidean(eye[0], eye[3])

    # EAR formula
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear


# -----------------------------
# Extract eye coordinates
# -----------------------------

def get_eye_points(shape, eye_indices):

    points = []

    for index in eye_indices:
        x = shape.part(index).x
        y = shape.part(index).y
        points.append((x, y))

    return np.array(points)


# -----------------------------
# Start webcam
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

frame_counter = 0
alarm_start_time = None

print("Drowsiness Detection Started")
print("Press 'q' to exit")


# -----------------------------
# Main loop
# -----------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (800, 600))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    for face in faces:

        # Detect facial landmarks
        shape = predictor(gray, face)

        # Get eye coordinates
        left_eye = get_eye_points(shape, LEFT_EYE)
        right_eye = get_eye_points(shape, RIGHT_EYE)

        # Calculate EAR
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        # Average EAR
        ear = (left_ear + right_ear) / 2.0

        # Draw eye contours
        cv2.polylines(
            frame,
            [left_eye],
            True,
            (0, 255, 0),
            2
        )

        cv2.polylines(
            frame,
            [right_eye],
            True,
            (0, 255, 0),
            2
        )

        # Display EAR value
        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # -----------------------------
        # Drowsiness detection
        # -----------------------------

        if ear < EAR_THRESHOLD:

            frame_counter += 1

            # Start timer
            if alarm_start_time is None:
                alarm_start_time = time.time()

            # Check consecutive frames
            if frame_counter >= CONSECUTIVE_FRAMES:

                cv2.putText(
                    frame,
                    "DROWSINESS DETECTED!",
                    (150, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    frame,
                    "PLEASE WAKE UP!",
                    (200, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3
                )

                # Visual alert
                cv2.rectangle(
                    frame,
                    (0, 0),
                    (799, 599),
                    (0, 0, 255),
                    5
                )

        else:

            # Reset counter when eyes are open
            frame_counter = 0
            alarm_start_time = None

    # Display webcam feed
    cv2.imshow(
        "Drowsiness Detection System",
        frame
    )

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Release resources
# -----------------------------

cap.release()
cv2.destroyAllWindows()

print("Drowsiness Detection Stopped.")