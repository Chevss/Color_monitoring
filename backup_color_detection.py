import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
from ultralytics import YOLO
import torch

# Set dark mode style for the plot
plt.style.use('dark_background')

# Video Parameters
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

# Function to initialize the camera
def initialize_camera(camera_choice):
    if camera_choice == 0:
        return cv2.VideoCapture(0)
    elif camera_choice == 1:
        return cv2.VideoCapture(1)
    else:
        raise ValueError("Invalid camera choice")

# Ask the user to choose the camera
camera_choice = int(input("Choose camera (0 for webcam, 1 for other camera, 2 for RTSP camera): "))
cap = initialize_camera(camera_choice)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Set device to Intel Arc GPU if available, otherwise use CPU
device = torch.device("xpu" if torch.cuda.is_available() else "cpu")

# Load the YOLOv8 model
model = YOLO('best.pt').to(device)

# Initialize deque to store color counts
color_counts = {}
frame_buffer = deque(maxlen=100)  # Initialize frame buffer

# Initialize plots
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1)
lines_live = {}
lines_static = {}

ax1.set_xlim(0, 100)
ax1.set_ylim(0, 1)  # Set y-axis limit to 0.5 for normalized values
ax2.legend(loc='upper right')
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 1)  # Set y-axis limit to 0.5 for normalized values

last_alert_times = {}
last_static_update_time = time.time()

# Variables for drawing the rectangle
drawing = False
box_start = (0, 0)
box_end = (0, 0)
detection_box_start = (0, 0)
detection_box_end = (0, 0)

# Initialize lists to store the highest values for the static plot
static_counts = {}

# Variable to track the current detection mode
detect_all_colors = True
colors_to_detect = []

# Predefined color ranges
predefined_colors = {
    'red': ((0, 100, 100), (10, 255, 255)),
    'green': ((50, 100, 100), (70, 255, 255)),
    'blue': ((100, 100, 100), (130, 255, 255)),
    'yellow': ((20, 100, 100), (30, 255, 255)),
    'orange': ((10, 100, 100), (20, 255, 255)),
    'purple': ((130, 100, 100), (160, 255, 255)),
    'brown': ((10, 100, 20), (20, 255, 200)),
    'white': ((0, 0, 200), (180, 20, 255))
}

def draw_rectangle(event, x, y, flags, param):
    global box_start, box_end, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        box_start = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            box_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        box_end = (x, y)

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', draw_rectangle)

def get_color_from_roi(frame, box_start, box_end):
    roi = frame[box_start[1]:box_end[1], box_start[0]:box_end[0]]
    if roi.size == 0:
        return None
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_color = np.mean(hsv_roi, axis=(0, 1))
    return avg_color

def select_colors():
    print("Select up to 5 colors from the following list:")
    for i, color in enumerate(predefined_colors.keys()):
        print(f"{i}: {color}")
    selected_colors = []
    while len(selected_colors) < 5:
        choice = input("Enter the number of the color to select (or 'c' to calibrate a new color): ")
        if choice.isdigit() and int(choice) in range(len(predefined_colors)):
            color_name = list(predefined_colors.keys())[int(choice)]
            selected_colors.append((predefined_colors[color_name][0], predefined_colors[color_name][1], color_name))
        elif choice == 'c':
            print("Draw a box around the color to calibrate.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                if drawing or (box_start != box_end):
                    cv2.rectangle(frame, box_start, box_end, (0, 255, 0), 2)
                cv2.imshow('Frame', frame)
                if not drawing and box_start != box_end:
                    avg_color = get_color_from_roi(frame, box_start, box_end)
                    if avg_color is not None:
                        lower = tuple(np.maximum([avg_color[0] - 10, 100, 100], 0))
                        upper = tuple(np.minimum([avg_color[0] + 10, 255, 255], 255))
                        color_name = input("Enter name for the new color: ")
                        selected_colors.append((lower, upper, color_name))
                        break
                    else:
                        print("Invalid ROI. Please try again.")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            print("Invalid choice. Please try again.")
    return selected_colors

use_leaf_detection = input("Do you want to use leaf detection? (y/n): ").lower() == 'y'

colors_to_detect = select_colors()
for color in colors_to_detect:
    color_counts[color] = deque(maxlen=100)
    static_counts[color] = []
    lines_live[color] = ax1.plot([], [], label=color[2], color=color[2])[0]
    lines_static[color] = ax2.plot([], [], label=color[2], color=color[2])[0]
    last_alert_times[color] = time.time()
ax2.legend(loc='upper right')

print("Draw a box around the area where colors will be detected.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    if drawing or (box_start != box_end):
        cv2.rectangle(frame, box_start, box_end, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    if not drawing and box_start != box_end:
        detection_box_start = box_start
        detection_box_end = box_end
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame_buffer.append(frame)

    detection_roi = frame[detection_box_start[1]:detection_box_end[1], detection_box_start[0]:detection_box_end[0]]
    hsv_frame = cv2.cvtColor(detection_roi, cv2.COLOR_BGR2HSV)

    if use_leaf_detection:
        # Run YOLOv8 inference
        results = model(detection_roi)

        # Draw YOLOv8 results on the frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = box.conf[0]
                label = model.names[class_id]
                cv2.rectangle(detection_roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(detection_roi, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Extract the region of interest (ROI) within the bounding box
                roi = detection_roi[y1:y2, x1:x2]
                hsv_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                alerts = []
                current_time = time.time()  # Define current_time here
                for color in colors_to_detect:
                    lower, upper, _ = color
                    mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
                    color_count = cv2.countNonZero(mask)
                    normalized_count = color_count / (roi.shape[0] * roi.shape[1])  # Normalize the count

                    # Append the normalized count to the deque
                    color_counts[color].append(normalized_count)

                    # Check if the normalized count exceeds the threshold
                    if normalized_count > 0.1 and (current_time - last_alert_times[color]) >= 5:
                        alerts.append(f"{color[2]} {normalized_count:.2f}")
                        last_alert_times[color] = current_time

                    # Update live plot data
                    lines_live[color].set_xdata(range(len(color_counts[color])))
                    lines_live[color].set_ydata(color_counts[color])

                if alerts:
                    print(f"Alert: {', '.join(alerts)} has exceeded the threshold")

                ax1.relim()
                ax1.autoscale_view()
                plt.draw()
                plt.pause(0.01)

                # Update static plot data every 20 seconds
                if (current_time - last_static_update_time) >= 5:
                    for color in colors_to_detect:
                        highest_value = max(color_counts[color]) if color_counts[color] else 0
                        static_counts[color].append(highest_value)
                        lines_static[color].set_xdata(range(len(static_counts[color])))
                        lines_static[color].set_ydata(static_counts[color])
                    ax2.relim()
                    ax2.autoscale_view()
                    last_static_update_time = current_time
    else:
        alerts = []
        current_time = time.time()  # Define current_time here
        for color in colors_to_detect:
            lower, upper, _ = color
            mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
            color_count = cv2.countNonZero(mask)
            normalized_count = color_count / (detection_roi.shape[0] * detection_roi.shape[1])  # Normalize the count

            # Append the normalized count to the deque
            color_counts[color].append(normalized_count)

            # Check if the normalized count exceeds the threshold
            if normalized_count > 0.1 and (current_time - last_alert_times[color]) >= 5:
                alerts.append(f"{color[2]} {normalized_count:.2f}")
                last_alert_times[color] = current_time

            # Update live plot data
            lines_live[color].set_xdata(range(len(color_counts[color])))
            lines_live[color].set_ydata(color_counts[color])

        if alerts:
            print(f"Alert: {', '.join(alerts)} has exceeded the threshold")

        ax1.relim()
        ax1.autoscale_view()
        plt.draw()
        plt.pause(0.01)

        # Update static plot data every 20 seconds
        if (current_time - last_static_update_time) >= 5:
            for color in colors_to_detect:
                highest_value = max(color_counts[color]) if color_counts[color] else 0
                static_counts[color].append(highest_value)
                lines_static[color].set_xdata(range(len(static_counts[color])))
                lines_static[color].set_ydata(static_counts[color])
            ax2.relim()
            ax2.autoscale_view()
            last_static_update_time = current_time

    # Draw the detection box on the frame
    if detection_box_start != detection_box_end:
        cv2.rectangle(frame, detection_box_start, detection_box_end, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
