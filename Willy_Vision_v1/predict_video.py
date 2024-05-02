import os
from ultralytics import YOLO
import cv2

VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'IMG_2003.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

# Check if the video has been opened correctly
if not ret:
    print("Error: Unable to open the video.")
    exit()

H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train17', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # Load a custom model

threshold = 0.8  # Lower detection threshold

while ret:
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

    # Save the processed frame
    out.write(frame)

    # Read the next frame
    ret, frame = cap.read()

    # Wait for a 1 ms delay and close the window if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close open windows
cap.release()
out.release()
cv2.destroyAllWindows()

