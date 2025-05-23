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

# Create ColorCalibrator class
class ColorCalibrator:
    def __init__(self):
        self.calibrated_colors = {}
    
    def calibrate_color(self, hsv_lower, hsv_upper, color_name):
        """Add a new calibrated color to the dictionary"""
        self.calibrated_colors[color_name] = (hsv_lower, hsv_upper)
        return True
    
    def get_colors(self):
        """Return all calibrated colors"""
        return self.calibrated_colors

class ColorDetector:
    def __init__(self, calibrator=None):
        self.calibrator = calibrator if calibrator else ColorCalibrator()

    def detect_color(self, hsv_value):
        """Detect if the HSV value matches any calibrated color."""
        calibrated_colors = self.calibrator.get_colors()
        for name, (lower, upper) in calibrated_colors.items():
            # Check if the HSV value is within the range
            if all(lower[i] <= hsv_value[i] <= upper[i] for i in range(3)):
                return name
        return None

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
ax1.set_title('Live Color Detection')
ax1.set_xlabel('Time')
ax1.set_ylabel('Normalized Color Count')

ax2.set_xlim(0, 100)
ax2.set_ylim(0, 1)  # Set y-axis limit to 0.5 for normalized values
ax2.set_title('Static Color Detection (Max Values)')
ax2.set_xlabel('Time Interval (5s)')
ax2.set_ylabel('Max Normalized Color Count')

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
    
    # Only consider non-zero pixels (inside the circle)
    non_zero_pixels = hsv_roi[mask > 0]
    if len(non_zero_pixels) > 0:
        avg_color = np.mean(non_zero_pixels, axis=0)
        return avg_color
    return None

def select_colors():
    global circle_center, circle_radius
    calibrator = ColorCalibrator()
    
    # Add predefined colors to the calibrator
    for color_name, (lower, upper) in predefined_colors.items():
        calibrator.calibrate_color(lower, upper, color_name)
    
    print("Select at least one color from the following list:")
    for i, color in enumerate(predefined_colors.keys()):
        print(f"{i}: {color}")
    
    selected_colors = []
    while True:
        choice = input("Enter the number of the color to select (or 'c' to calibrate a new color, 'x' to finish): ")
        if choice.isdigit() and int(choice) in range(len(predefined_colors)):
            color_name = list(predefined_colors.keys())[int(choice)]
            lower, upper = predefined_colors[color_name]
            selected_colors.append((lower, upper, color_name))
        elif choice == 'c':
            print("Draw a circle around the color to calibrate.")
            circle_center, circle_radius = (0, 0), 0  # Reset circle
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                temp_frame = frame.copy()
                if drawing or circle_radius > 0:
                    cv2.circle(temp_frame, circle_center, circle_radius, (0, 255, 0), 2)
                cv2.imshow('Frame', temp_frame)
                if not drawing and circle_radius > 0:
                    avg_color = get_color_from_roi(frame, circle_center, circle_radius)
                    if avg_color is not None and not np.isnan(avg_color).any():
                        # Create HSV bounds for the new color with wider ranges for better detection
                        lower = tuple(np.maximum(avg_color - [10, 50, 50], [0, 0, 0]).astype(int))
                        upper = tuple(np.minimum(avg_color + [10, 50, 50], [179, 255, 255]).astype(int))
                        print(f"Detected HSV color: {avg_color}")
                        print(f"Using HSV range: Lower={lower}, Upper={upper}")
                        color_name = input("Enter name for the new color: ")
                        
                        # Add to calibrator
                        calibrator.calibrate_color(lower, upper, color_name)
                        
                        # Add to selected colors
                        selected_colors.append((lower, upper, color_name))
                        
                        # Also update the predefined_colors dictionary for future reference
                        predefined_colors[color_name] = (lower, upper)
                        
                        # Reset the circle
                        circle_center, circle_radius = (0, 0), 0
                        break
                    else:
                        print("Invalid ROI or no color detected. Please try again.")
                        circle_center, circle_radius = (0, 0), 0  # Reset circle
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        elif choice == 'x':
            if len(selected_colors) >= 1:
                break
            else:
                print("You must select at least one color.")
        else:
            print("Invalid choice. Please try again.")
    return selected_colors, calibrator

def track_color_transitions(hsv_frame, center, radius, colors_to_detect, color_detector):
    global color_transitions
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    circular_roi = cv2.bitwise_and(hsv_frame, hsv_frame, mask=mask)
    area = np.pi * radius ** 2
    
    # Track each color
    for color in colors_to_detect:
        lower, upper, color_name = color
        mask = cv2.inRange(circular_roi, np.array(lower), np.array(upper))
        color_count = cv2.countNonZero(mask)
        normalized_count = min(color_count / area, 1.0)
        
        # Make sure the color exists in the transitions dictionary
        if color_name not in color_transitions:
            color_transitions[color_name] = deque(maxlen=100)
        
        color_transitions[color_name].append(normalized_count)

# Get colors to detect and calibrator
colors_to_detect, color_calibrator = select_colors()

# Initialize color detector with the calibrator
color_detector = ColorDetector(color_calibrator)

# Set up the plots
for color in colors_to_detect:
    color_counts[color] = deque(maxlen=100)
    static_counts[color] = []
    color_name = color[2]
    lines_live[color] = ax1.plot([], [], label=color_name, color=color_name.lower() if color_name.lower() in ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'brown'] else 'white')[0]
    lines_static[color] = ax2.plot([], [], label=color_name, color=color_name.lower() if color_name.lower() in ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'brown'] else 'white')[0]
    last_alert_times[color] = time.time()

ax1.legend(loc='upper right')
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

# Ensure all colors are in the transitions dictionary
for color in colors_to_detect:
    _, _, color_name = color
    if color_name not in color_transitions:
        color_transitions[color_name] = deque(maxlen=100)

# Main loop to detect colors
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame_buffer.append(frame)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Track color transitions using the color detector
    track_color_transitions(hsv_frame, detection_circle_center, detection_circle_radius, colors_to_detect, color_detector)

    alerts = []
    current_time = time.time()
    area = np.pi * detection_circle_radius ** 2
    
    # Create a mask for the detection circle
    detection_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(detection_mask, detection_circle_center, detection_circle_radius, 255, -1)
    
    # Process each color
    for color in colors_to_detect:
        lower, upper, color_name = color
        mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
        
        # Apply the detection circle mask
        mask = cv2.bitwise_and(mask, mask, mask=detection_mask)
        
        color_count = cv2.countNonZero(mask)
        normalized_count = min(color_count / area, 1.0)

        # Append the normalized count to the deque
        color_counts[color].append(normalized_count)

        # Check if the normalized count exceeds a threshold
        if normalized_count >= 0:
            alerts.append(f"{color_name}: {normalized_count:.2f}")

        # Update live plot data
        lines_live[color].set_xdata(range(len(color_counts[color])))
        lines_live[color].set_ydata(color_counts[color])
        
        # Visualize the detected color in the frame
        color_display = frame.copy()
        color_display[mask > 0] = [255, 255, 255]  # Highlight detected pixels
        cv2.addWeighted(color_display, 0.3, frame, 0.7, 0, frame)

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

    # Add text showing the detected colors in the frame
    y_offset = 30
    for alert in alerts:
        cv2.putText(frame, alert, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Plot color transitions
plt.figure()
for color_name, transitions in color_transitions.items():
    if transitions:  # Only plot if there's data
        plt.plot(list(transitions), label=color_name)
plt.xlabel('Time')
plt.ylabel('Normalized Color Count')
plt.title('Color Transitions Over Time')
plt.legend()
plt.show()

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()