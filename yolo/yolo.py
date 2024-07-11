from ultralytics import YOLO
import cv2
from PIL import Image

# Load the model
model = YOLO('yolov8n.pt')

# Perform inference on the image
results = model("bus.jpg")

# Save the image with bounding boxes
annotated_frame = results[0].plot()
cv2.imwrite("output.jpg", annotated_frame)

# Open the saved image using PIL and display it
img = Image.open("output.jpg")
img.show()

print("Image saved as output.jpg and displayed.")
