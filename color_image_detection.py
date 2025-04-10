import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# Image Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 1280, 720

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

# Variables for drawing circles
drawing = False
current_circle = {"center": (0, 0), "radius": 0}
detection_circles = []  # List to store multiple detection circles

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

def draw_circle(event, x, y, flags, param):
    """Handle mouse events to draw a circle."""
    global current_circle, drawing, detection_circles
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_circle["center"] = (x, y)
        current_circle["radius"] = 0
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_circle["radius"] = int(((x - current_circle["center"][0]) ** 2 + 
                                         (y - current_circle["center"][1]) ** 2) ** 0.5)
    
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            current_circle["radius"] = int(((x - current_circle["center"][0]) ** 2 + 
                                         (y - current_circle["center"][1]) ** 2) ** 0.5)
            
            # Only add the circle if it has a valid radius and we're in the main detection window
            if current_circle["radius"] > 5:
                # Create a new circle with a unique ID
                new_circle = {
                    "id": len(detection_circles) + 1,
                    "center": current_circle["center"],
                    "radius": current_circle["radius"],
                    "results": {}
                }
                detection_circles.append(new_circle)
                
                # Reset current circle for next drawing
                current_circle = {"center": (0, 0), "radius": 0}

def draw_calibration_circle(event, x, y, flags, param):
    """Separate function for drawing circles during calibration."""
    global current_circle, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_circle["center"] = (x, y)
        current_circle["radius"] = 0
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_circle["radius"] = int(((x - current_circle["center"][0]) ** 2 + 
                                         (y - current_circle["center"][1]) ** 2) ** 0.5)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_circle["radius"] = int(((x - current_circle["center"][0]) ** 2 + 
                                     (y - current_circle["center"][1]) ** 2) ** 0.5)

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

def select_image():
    """Open a file dialog to select an image."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return file_path

def select_colors(image, existing_calibrator=None):
    global current_circle
    
    # Initialize calibrator - either use existing one or create a new one
    calibrator = existing_calibrator if existing_calibrator else ColorCalibrator()
    
    # If using a new calibrator, add predefined colors
    if not existing_calibrator:
        # Add predefined colors to the calibrator
        for color_name, (lower, upper) in predefined_colors.items():
            calibrator.calibrate_color(lower, upper, color_name)
    
    print("Select at least one color from the following list:")
    for i, color in enumerate(predefined_colors.keys()):
        print(f"{i}: {color}")
    
    # Show existing calibrated colors if any
    calibrated_colors = calibrator.get_colors()
    if calibrated_colors:
        print("\nCurrently selected colors:")
        for color_name in calibrated_colors:
            print(f"- {color_name}")
    
    selected_colors = []
    while True:
        choice = input("Enter the number of the color to select (or 'c' to calibrate a new color, 'x' to finish): ")
        if choice.isdigit() and int(choice) in range(len(predefined_colors)):
            color_name = list(predefined_colors.keys())[int(choice)]
            lower, upper = predefined_colors[color_name]
            selected_colors.append((lower, upper, color_name))
            print(f"Added {color_name} to detection list")
        elif choice == 'c':
            print("Draw a circle around the color to calibrate.")
            current_circle = {"center": (0, 0), "radius": 0}  # Reset circle
            
            # Create a window for color calibration
            cv2.namedWindow('Calibrate Color')
            cv2.setMouseCallback('Calibrate Color', draw_calibration_circle)  # Use separate calibration function
            
            while True:
                temp_image = image.copy()
                if drawing or current_circle["radius"] > 0:
                    cv2.circle(temp_image, current_circle["center"], current_circle["radius"], (0, 255, 0), 2)
                
                # Add instructions text
                cv2.putText(temp_image, "Draw a circle around the color", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(temp_image, "Press 'r' to reset, 'Enter' to confirm, 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Calibrate Color', temp_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):  # Reset the circle if 'r' is pressed
                    current_circle = {"center": (0, 0), "radius": 0}
                elif key == 13 and current_circle["radius"] > 0:  # Enter key and valid circle
                    avg_color = get_color_from_roi(image, current_circle["center"], current_circle["radius"])
                    if avg_color is not None and not np.isnan(avg_color).any():
                        # Create HSV bounds for the new color with wider ranges for better detection
                        lower = tuple(np.maximum(avg_color - [8, 25, 25], [0, 0, 0]).astype(int))
                        upper = tuple(np.minimum(avg_color + [8, 25, 25], [179, 255, 255]).astype(int))
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
                        current_circle = {"center": (0, 0), "radius": 0}
                        break
                    else:
                        print("Invalid ROI or no color detected. Please try again.")
                        current_circle = {"center": (0, 0), "radius": 0}  # Reset circle
                elif key == ord('q'):
                    break
            
            cv2.destroyWindow('Calibrate Color')
        
        elif choice == 'x':
            if len(selected_colors) >= 1:
                break
            else:
                print("You must select at least one color.")
        else:
            print("Invalid choice. Please try again.")
    
    return selected_colors, calibrator

def detect_colors_in_circles(image, detection_circles, colors_to_detect):
    """Detect colors within multiple specified circles in the image."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    display_image = image.copy()
    
    # Process each detection circle
    for circle in detection_circles:
        circle_id = circle["id"]
        center = circle["center"]
        radius = circle["radius"]
        
        # Create a mask for this detection circle
        detection_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(detection_mask, center, radius, 255, -1)
        
        # Calculate circle area
        area = np.pi * radius ** 2
        
        # Store results for this circle
        color_results = {}
        
        # Process each color for this circle
        for color in colors_to_detect:
            lower, upper, color_name = color
            mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            
            # Apply the detection circle mask
            mask = cv2.bitwise_and(mask, mask, mask=detection_mask)
            
            color_count = cv2.countNonZero(mask)
            normalized_count = min(color_count / area, 1.0) if area > 0 else 0
            
            color_results[color_name] = normalized_count
            
            # Highlight detected color in this circle
            temp_image = image.copy()
            temp_image[mask > 0] = [255, 255, 255]  # Highlight detected pixels
            cv2.addWeighted(temp_image, 0.3, display_image, 0.7, 0, display_image)
        
        # Save results for this circle
        circle["results"] = color_results
        
        # Draw the circle on the display image with its ID
        cv2.circle(display_image, center, radius, (0, 255, 0), 2)
        cv2.putText(display_image, f"#{circle_id}", 
                    (center[0] - 15, center[1] - radius - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display HSV values at the center of the circle
        hsv_at_center = get_color_from_roi(image, center, 2)  # Small radius to get center color
        if hsv_at_center is not None:
            hsv_text = f"HSV: ({int(hsv_at_center[0])},{int(hsv_at_center[1])},{int(hsv_at_center[2])})"
            cv2.putText(display_image, hsv_text, 
                        (center[0] - 10, center[1] + radius + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return display_image

def display_results_sidebar(image, detection_circles, colors_to_detect):
    """Display color detection results in a sidebar."""
    # Create a sidebar for results
    h, w = image.shape[:2]
    sidebar_width = 300
    result_image = np.zeros((h, w + sidebar_width, 3), dtype=np.uint8)
    result_image[:, :w] = image
    
    # Fill sidebar with dark gray
    result_image[:, w:] = (50, 50, 50)
    
    # Add title to sidebar
    cv2.putText(result_image, "Color Detection Results", 
                (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display results for each circle
    y_offset = 70
    for circle in detection_circles:
        circle_id = circle["id"]
        results = circle["results"]
        
        # Display circle ID
        cv2.putText(result_image, f"Circle #{circle_id}:", 
                    (w + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        
        # Display HSV value at center
        center = circle["center"]
        hsv_at_center = get_color_from_roi(image, center, 2)  # Small radius to get center color
        if hsv_at_center is not None:
            hsv_text = f"HSV: ({int(hsv_at_center[0])},{int(hsv_at_center[1])},{int(hsv_at_center[2])})"
            cv2.putText(result_image, hsv_text, 
                        (w + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 20
        
        # Display color percentages
        for color_name, percentage in results.items():
            # Get color RGB value for visualization
            color_rgb = get_color_rgb(color_name)
            
            # # Draw percentage bar
            # bar_length = int(percentage * 150)  # Scale to 150px max
            # cv2.rectangle(result_image, 
            #               (w + 20, y_offset - 15), 
            #               (w + 20 + bar_length, y_offset - 5), 
            #               color_rgb, -1)
            
            # Add text
            text = f"{color_name}: {percentage:.2f}"
            cv2.putText(result_image, text, 
                        (w + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        y_offset += 20  # Add space between circles
    
    return result_image

def get_color_rgb(color_name):
    """Return RGB value for common colors."""
    color_map = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'orange': (0, 165, 255),
        'purple': (255, 0, 255),
        'brown': (42, 42, 165),
        'white': (255, 255, 255)
    }
    return color_map.get(color_name, (200, 200, 200))  # Default to light gray

def load_and_process_new_image(colors_to_detect, color_calibrator):
    """Load a new image and reset detection circles."""
    global detection_circles, current_circle
    
    # Reset detection circles
    detection_circles = []
    current_circle = {"center": (0, 0), "radius": 0}
    
    # Select a new image
    image_path = select_image()
    if not image_path or not os.path.exists(image_path):
        print("No image selected or invalid image path.")
        return None, False
    
    # Load the image
    new_image = cv2.imread(image_path)
    if new_image is None:
        print(f"Failed to load image: {image_path}")
        return None, False
    
    # Resize the image
    new_image = cv2.resize(new_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    
    return new_image, True

def main():
    global current_circle, detection_circles
    
    # Reset detection circles at the start
    detection_circles = []
    
    # Select an image
    image_path = select_image()
    if not image_path or not os.path.exists(image_path):
        print("No image selected or invalid image path.")
        return
    
    # Load the image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Resize the image
    original_image = cv2.resize(original_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    
    # Get initial colors to detect
    colors_to_detect, color_calibrator = select_colors(original_image.copy())
    
    # Initialize color detector with the calibrator
    color_detector = ColorDetector(color_calibrator)
    
    # Create a window for drawing the detection circles
    cv2.namedWindow('Color Detection')
    cv2.setMouseCallback('Color Detection', draw_circle)
    
    show_results = False
    
    while True:
        # Make a copy of the original image for display
        display_image = original_image.copy()
        
        # Draw all existing detection circles
        for circle in detection_circles:
            cv2.circle(display_image, circle["center"], circle["radius"], (0, 255, 0), 2)
            # Add circle ID
            cv2.putText(display_image, f"#{circle['id']}", 
                        (circle["center"][0] - 15, circle["center"][1] - circle["radius"] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw the circle currently being drawn
        if drawing or (current_circle["radius"] > 0 and current_circle["center"] != (0, 0)):
            cv2.circle(display_image, current_circle["center"], current_circle["radius"], (255, 255, 0), 2)
        
        # If showing results, process and display color detection
        if show_results and detection_circles:
            # Detect colors in all circles
            processed_image = detect_colors_in_circles(original_image, detection_circles, colors_to_detect)
            
            # Create results sidebar
            result_image = display_results_sidebar(processed_image, detection_circles, colors_to_detect)
            
            display_image = result_image
        
        # Add instructions text
        if not show_results:
            cv2.putText(display_image, "Draw circles around areas to detect colors", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, "Press 'Enter' to analyze, 'c' to clear all circles, 'n' for new image, 'q' to quit", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, "Press 's' to select different colors", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(display_image, "Press 'r' to reset and draw new circles, 'n' for new image", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, "Press 's' to select different colors, 'q' to quit", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the image
        cv2.imshow('Color Detection', display_image)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Process key presses
        if key == ord('q'):
            break
        elif key == ord('r') or key == ord('c'):
            # Reset all circles
            detection_circles = []
            current_circle = {"center": (0, 0), "radius": 0}
            show_results = False
        elif key == ord('n'):
            # Load a new image
            print("Loading new image...")
            new_image, success = load_and_process_new_image(colors_to_detect, color_calibrator)
            if success:
                original_image = new_image
                show_results = False
                print("New image loaded successfully.")
        elif key == ord('s'):
            # Go back to color selection
            print("\nReselecting colors to detect...")
            # Pass the existing calibrator to maintain custom colors
            colors_to_detect, color_calibrator = select_colors(original_image.copy(), color_calibrator)
            # Reset circles and results when colors change
            detection_circles = []
            current_circle = {"center": (0, 0), "radius": 0}
            show_results = False
            print("Colors updated successfully. Draw new detection circles.")
        elif key == 13:  # Enter key
            if detection_circles:
                show_results = True
            else:
                print("Draw at least one circle before analyzing.")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()