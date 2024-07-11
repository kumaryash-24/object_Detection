from ultralytics import YOLO
import cv2

# Load the model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "traffic.mp4"  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = "output_video.avi"  # Output path for the annotated video
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Annotate the frame with bounding boxes
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Display the annotated frame in real-time
    cv2.imshow('Object Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Annotated video saved as {output_path}.")
