import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
from ultralytics import YOLO
import torch
from color_calibration import ColorCalibrator

# Set dark mode style for the plot
plt.style.use('dark_background')

# Video Parameters
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

def initialize_camera(camera_choice):
    if camera_choice == 0:
        return cv2.VideoCapture(0)
    elif camera_choice == 1:
        return cv2.VideoCapture(1)
    elif camera_choice == 2:
        return cv2.VideoCapture(2)
    elif camera_choice == 3:
        # List of possible URLs to try
        urls = [
            "http://192.168.1.39" 
        ]
        
        for url in urls:
            print(f"Attempting to connect to: {url}")
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                print(f"Successfully connected to: {url}")
                return cap
            else:
                print(f"Failed to connect to: {url}")
        
        print("Could not connect to camera using any of the tried URLs")
        return None
    else:
        raise ValueError("Invalid camera choice")

# Function to check if the camera is opened successfully
def check_camera_connection(cap):
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return False
    return True

# Ask the user to choose the camera
while True:
    camera_choice = int(input("Choose camera (0 for webcam, 1 for other camera, 2 for RTSP camera at 192.168.1.39): "))
    cap = initialize_camera(camera_choice)
    if check_camera_connection(cap):
        break
    else:
        print("Please choose a valid camera option.")

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

# Variables for drawing the circle
drawing = False
circle_center = (0, 0)
circle_radius = 0
detection_circle_center = (0, 0)
detection_circle_radius = 0

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

# Initialize deque to store color transitions
color_transitions = {color: deque(maxlen=100) for color in predefined_colors.keys()}

def draw_circle(event, x, y, flags, param):
    """Handle mouse events to draw a circle."""
    global circle_center, circle_radius, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        circle_center = (x, y)
        circle_radius = 0
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            circle_radius = int(((x - circle_center[0]) ** 2 + (y - circle_center[1]) ** 2) ** 0.5)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        circle_radius = int(((x - circle_center[0]) ** 2 + (y - circle_center[1]) ** 2) ** 0.5)

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', draw_circle)

def get_color_from_roi(frame, center, radius):
    """Extract the average HSV color from a circular ROI."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    circular_roi = cv2.bitwise_and(frame, frame, mask=mask)
    hsv_roi = cv2.cvtColor(circular_roi, cv2.COLOR_BGR2HSV)
    avg_color = np.mean(hsv_roi[hsv_roi > 0].reshape(-1, 3), axis=0)  # Calculate the average HSV color
    return avg_color

def select_colors():
    global circle_center, circle_radius  # Ensure these variables are accessible and modifiable
    print("Select at least one color from the following list:")
    for i, color in enumerate(predefined_colors.keys()):
        print(f"{i}: {color}")
    selected_colors = []
    while True:
        choice = input("Enter the number of the color to select (or 'c' to calibrate a new color, 'x' to finish): ")
        if choice.isdigit() and int(choice) in range(len(predefined_colors)):
            color_name = list(predefined_colors.keys())[int(choice)]
            selected_colors.append((predefined_colors[color_name][0], predefined_colors[color_name][1], color_name))
        elif choice == 'c':
            print("Draw a circle around the color to calibrate.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                temp_frame = frame.copy()  # Create a copy to avoid permanent drawing
                if drawing or circle_radius > 0:
                    cv2.circle(temp_frame, circle_center, circle_radius, (0, 255, 0), 2)
                cv2.imshow('Frame', temp_frame)
                if not drawing and circle_radius > 0:
                    avg_color = get_color_from_roi(frame, circle_center, circle_radius)
                    if avg_color is not None:
                        # Further increase sensitivity by narrowing the HSV bounds
                        lower = tuple(np.maximum(avg_color - [1, 10, 10], [0, 0, 0]).astype(int))
                        upper = tuple(np.minimum(avg_color + [1, 10, 10], [179, 255, 255]).astype(int))
                        print(f"Detected color range in HSV: Lower={lower}, Upper={upper}")  # Display the range
                        color_name = input("Enter name for the new color: ")
                        selected_colors.append((lower, upper, color_name))
                        circle_center, circle_radius = (0, 0), 0  # Reset the circle coordinates
                        break
                    else:
                        print("Invalid ROI. Please try again.")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        elif choice == 'x':
            if len(selected_colors) >= 1:
                break
            else:
                print("You must select at least one color.")
        else:
            print("Invalid choice. Please try again.")
    return selected_colors

colors_to_detect = select_colors()
for color in colors_to_detect:
    color_counts[color] = deque(maxlen=100)
    static_counts[color] = []
    lines_live[color] = ax1.plot([], [], label=color[2], color=color[2])[0]
    lines_static[color] = ax2.plot([], [], label=color[2], color=color[2])[0]
    last_alert_times[color] = time.time()
ax2.legend(loc='upper right')

print("Draw a circle around the area where colors will be detected.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    if drawing or circle_radius > 0:
        cv2.circle(frame, circle_center, circle_radius, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    if not drawing and circle_radius > 0:
        detection_circle_center = circle_center
        detection_circle_radius = circle_radius
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

def track_color_transitions(hsv_frame, center, radius, colors_to_detect):
    global color_transitions  # Ensure we modify the global dictionary
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    circular_roi = cv2.bitwise_and(hsv_frame, hsv_frame, mask=mask)
    area = np.pi * radius ** 2  # Calculate the area of the circular ROI
    for color in colors_to_detect:
        lower, upper, color_name = color
        # Dynamically add missing colors to color_transitions
        if color_name not in color_transitions:
            color_transitions[color_name] = deque(maxlen=100)
        mask = cv2.inRange(circular_roi, np.array(lower), np.array(upper))
        color_count = cv2.countNonZero(mask)
        normalized_count = min(color_count / area, 1.0)  # Normalize the count to a range of 0.0 to 1.0
        color_transitions[color_name].append(normalized_count)

# Ensure dynamically added colors are initialized in color_transitions
for color in colors_to_detect:
    _, _, color_name = color
    if color_name not in color_transitions:
        color_transitions[color_name] = deque(maxlen=100)

class ColorDetector:
    def __init__(self):
        self.calibrator = ColorCalibrator()

    def detect_color(self, rgb_values):
        """Detect the closest calibrated color."""
        calibrated_colors = self.calibrator.get_colors()
        closest_color = None
        min_distance = float('inf')

        for name, color in calibrated_colors.items():
            distance = sum((rgb_values[i] - color[i]) ** 2 for i in range(3)) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_color = name

        return closest_color

# Main loop to detect colors
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame_buffer.append(frame)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    track_color_transitions(hsv_frame, detection_circle_center, detection_circle_radius, colors_to_detect)

    alerts = []
    current_time = time.time()
    area = np.pi * detection_circle_radius ** 2  # Calculate the area of the circular ROI
    for color in colors_to_detect:
        lower, upper, color_name = color
        mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
        color_count = cv2.countNonZero(mask)
        normalized_count = min(color_count / area, 1.0)  # Normalize the count to a range of 0.0 to 1.0

        # Append the normalized count to the deque
        color_counts[color].append(normalized_count)

        # Check if the normalized count exceeds a threshold
        if normalized_count > 0.1:
            alerts.append(f"{color_name}: {normalized_count:.2f}")

        # Update live plot data
        lines_live[color].set_xdata(range(len(color_counts[color])))
        lines_live[color].set_ydata(color_counts[color])

    if alerts:
        print(f"Detected: {', '.join(alerts)}")

    ax1.relim()
    ax1.autoscale_view()
    plt.draw()
    plt.pause(0.01)

    # Update static plot data every 5 seconds
    if (current_time - last_static_update_time) >= 5:
        for color in colors_to_detect:
            highest_value = max(color_counts[color]) if color_counts[color] else 0
            static_counts[color].append(highest_value)
            lines_static[color].set_xdata(range(len(static_counts[color])))
            lines_static[color].set_ydata(static_counts[color])
        ax2.relim()
        ax2.autoscale_view()
        last_static_update_time = current_time

    # Draw the detection circle on the frame
    if detection_circle_radius > 0:
        cv2.circle(frame, detection_circle_center, detection_circle_radius, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Plot color transitions
plt.figure()
for color_name, transitions in color_transitions.items():
    plt.plot(transitions, label=color_name)
plt.xlabel('Time')
plt.ylabel('Normalized Color Count')
plt.title('Color Transitions Over Time')
plt.legend()
plt.show()

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
