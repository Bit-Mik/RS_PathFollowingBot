import cv2
import numpy as np

# Define the black color threshold
# You can adjust the threshold to detect different shades of black
black_threshold = 50

# Start capturing the video input from the default camera (0)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to detect black areas
    _, black_areas = cv2.threshold(gray_frame, black_threshold, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the black areas
    contours, _ = cv2.findContours(black_areas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)  # Green color for contours

    # Display the resulting frame with detected black areas
    cv2.imshow('Black Area Detection', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
