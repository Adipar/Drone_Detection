from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("C:/Users/aadip/OneDrive/Documents/autonomous drones/pt file/Drone.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Iterate through results
    for result in results:
        annotated_frame = result.plot()  # Annotate the frame with detection results

        # Display the annotated frame
        cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
