from ultralytics import YOLO
import cv2
import pygame
import numpy as np

# Initialize pygame
pygame.init()

# Load the model
model = YOLO('yolov8n.pt')

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set up the display window
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("YOLO Detection")

running = True
while running:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    # Perform inference on the frame
    results = model(frame)

    # Plot the results on the frame
    annotated_frame = results[0].plot()

    # Convert the frame to RGB (pygame uses RGB)
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    frame_rgb = np.rot90(frame_rgb)  # Rotate frame if needed
    frame_rgb = pygame.surfarray.make_surface(frame_rgb)

    # Display the frame
    screen.blit(frame_rgb, (0, 0))
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False

# Release the webcam and close all windows
cap.release()
pygame.quit()
