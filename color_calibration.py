# ...existing code...

class ColorCalibrator:
    def __init__(self):
        self.calibrated_colors = {}  # Store colors as a dictionary {name: RGB values}
        self.load_colors()

    def calibrate_color(self, name, rgb_values):
        """Add or update a color calibration."""
        self.calibrated_colors[name] = rgb_values
        self.save_colors()

    def save_colors(self):
        """Save calibrated colors to a file."""
        with open("calibrated_colors.json", "w") as file:
            json.dump(self.calibrated_colors, file)

    def load_colors(self):
        """Load calibrated colors from a file."""
        try:
            with open("calibrated_colors.json", "r") as file:
                self.calibrated_colors = json.load(file)
        except FileNotFoundError:
            self.calibrated_colors = {}

    def get_colors(self):
        """Return all calibrated colors."""
        return self.calibrated_colors

# Example usage:
# calibrator = ColorCalibrator()
# calibrator.calibrate_color("new_color", [255, 200, 100])
# print(calibrator.get_colors())
