import os
from ultralytics import YOLO
import cv2

# Authentication Information
username = 'admin'
password = 'Louismoret2002%40'

# Video stream URL with authentication
stream_url = f'rtsp://{username}:{password}@192.168.1.108'

# Initializing video capture from stream address
cap = cv2.VideoCapture(stream_url)

# Check if the capture has been opened correctly
if not cap.isOpened():
    print("Error: Unable to access the video stream.")
    exit()

# Load the YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train17', 'weights', 'last.pt')
model = YOLO(model_path)  # Load a custom model
threshold = 0.8  # Detection threshold

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read the video stream.")
        break

    # Applying the model on the frame
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Detection', frame)

    # Wait for a delay of 1 ms and close the window if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close open windows
cap.release()
cv2.destroyAllWindows()

