import cv2
import numpy as np

class GridDetector:
    def __init__(self, lower_orange=np.array([5, 100, 100]), upper_orange=np.array([15, 255, 255])):
        # Initialize color thresholds
        self.lower_orange = lower_orange
        self.upper_orange = upper_orange

    # Calculate the intersection point of two line segments
    def get_line_intersection(self, line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None  # Parallel lines have no intersection

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return int(px), int(py)

    # Extend a line segment
    def extend_line(self, x1, y1, x2, y2, scale=1000):
        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        extension_factor = scale / line_length

        new_x1 = int(x1 - (x2 - x1) * extension_factor)
        new_y1 = int(y1 - (y2 - y1) * extension_factor)
        new_x2 = int(x2 + (x2 - x1) * extension_factor)
        new_y2 = int(y2 + (y2 - y1) * extension_factor)

        return new_x1, new_y1, new_x2, new_y2

    # Merge multiple line segments and choose the longest
    def merge_lines(self, lines):
        if len(lines) == 0:
            return None

        longest_line = max(lines, key=lambda l: np.sqrt((l[2] - l[0]) ** 2 + (l[3] - l[1]) ** 2))
        return longest_line

    # Calculate division points between two points
    def calculate_division_points(self, P1, P2, num_divisions):
        points = []
        for i in range(1, num_divisions):
            x = P1[0] + i * (P2[0] - P1[0]) // num_divisions
            y = P1[1] + i * (P2[1] - P1[1]) // num_divisions
            points.append((x, y))
        return points

    # Draw division points and return their coordinates
    def draw_division_points(self, image, corners):
        points_coordinates = []  # Store coordinates of all points

        # Assign corners as P1-P4 (top-left, top-right, bottom-left, bottom-right)
        P1 = corners[0]
        P2 = corners[1]
        P3 = corners[2]
        P4 = corners[3]

        # Add corner points to the coordinates list
        points_coordinates.extend([P1, P2, P3, P4])

        # Display coordinates of the corners on the image
        for idx, point in enumerate([P1, P2, P3, P4]):
            cv2.putText(image, f"P{idx+1} ({point[0]}, {point[1]})", (point[0] + 5, point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Calculate division points
        div_points_p1_p2 = self.calculate_division_points(P1, P2, 4)
        div_points_p1_p3 = self.calculate_division_points(P1, P3, 4)
        div_points_p2_p4 = self.calculate_division_points(P2, P4, 4)
        div_points_p3_p4 = self.calculate_division_points(P3, P4, 2)

        # Draw division points and store their coordinates
        for points, label in [(div_points_p1_p2, "P1-P2"), (div_points_p1_p3, "P1-P3"), 
                              (div_points_p2_p4, "P2-P4"), (div_points_p3_p4, "P3-P4")]:
            for idx, point in enumerate(points):
                cv2.circle(image, point, 5, (0, 255, 0), -1)
                cv2.putText(image, f"Div {label}-{idx+1} ({point[0]}, {point[1]})", (point[0] + 5, point[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                points_coordinates.append(point)

        return points_coordinates

    # Detect and merge lines, return the image and coordinates of 23 points
    def detect_and_merge_lines(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, self.lower_orange, self.upper_orange)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return image, []

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        top_region = (int(x + 0.2 * w), y - int(0.3 * h), int(x + 0.8 * w), y + int(0.2 * h))
        bottom_region = (int(x + 0.2 * w), y + h - int(0.2 * h), int(x + 0.8 * w), y + h + int(0.3 * h))
        left_region = (x - int(0.3 * w), int(y + 0.2 * h), x + int(0.2 * w), int(y + 0.8 * h))
        right_region = (x + w - int(0.2 * w), int(y + 0.2 * h), x + w + int(0.3 * w), int(y + 0.8 * h))

        regions = [(top_region, 'horizontal'), (bottom_region, 'horizontal'),
                   (left_region, 'vertical'), (right_region, 'vertical')]

        all_lines = []
        for region, direction in regions:
            x_start = max(0, region[0])
            y_start = max(0, region[1])
            x_end = min(image.shape[1], region[2])
            y_end = min(image.shape[0], region[3])

            mask_region = mask[y_start:y_end, x_start:x_end]

            if mask_region.size == 0:
                continue

            mask_region_smooth = cv2.GaussianBlur(mask_region, (5, 5), 0)
            edges = cv2.Canny(mask_region_smooth, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=15)

            if lines is not None:
                lines = [line[0] for line in lines]
                merged_line = self.merge_lines(lines)
                if merged_line is not None:
                    extended_line = self.extend_line(*merged_line)
                    cv2.line(image, (extended_line[0] + x_start, extended_line[1] + y_start),
                              (extended_line[2] + x_start, extended_line[3] + y_start), (255, 0, 0), 2)
                    all_lines.append([extended_line[0] + x_start, extended_line[1] + y_start,
                                      extended_line[2] + x_start, extended_line[3] + y_start])

        if len(all_lines) >= 4:
            corners = []
            for i in range(len(all_lines)):
                for j in range(i + 1, len(all_lines)):
                    intersection = self.get_line_intersection(all_lines[i], all_lines[j])
                    if intersection is not None:
                        corners.append(intersection)

            if len(corners) >= 4:
                points_coordinates = self.draw_division_points(image, corners[:4])
                return image, points_coordinates

        return image, []

    # Check if a point is inside a quadrilateral
    def is_point_in_rect(self, p, rect):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(p, rect[0], rect[1])
        d2 = sign(p, rect[1], rect[3])
        d3 = sign(p, rect[3], rect[2])
        d4 = sign(p, rect[2], rect[0])

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0) or (d4 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0) or (d4 > 0)

        return not (has_neg and has_pos)

    # Get coordinates of all grid vertices and label the center of each grid
    def get_grid_coordinates(self, image, points_coordinates):
        grids = [
            [points_coordinates[0], points_coordinates[4], points_coordinates[7], points_coordinates[14]],  # Grid 1
            [points_coordinates[4], points_coordinates[5], points_coordinates[14], points_coordinates[15]],  # Grid 2
            [points_coordinates[5], points_coordinates[6], points_coordinates[15], points_coordinates[16]],  # Grid 3
            [points_coordinates[6], points_coordinates[1], points_coordinates[16], points_coordinates[10]],   # Grid 4
            [points_coordinates[7], points_coordinates[14], points_coordinates[8], points_coordinates[17]],  # Grid 5
            [points_coordinates[14], points_coordinates[15], points_coordinates[17], points_coordinates[18]], # Grid 6
            [points_coordinates[15], points_coordinates[16], points_coordinates[18], points_coordinates[19]], # Grid 7
            [points_coordinates[16], points_coordinates[10], points_coordinates[19], points_coordinates[11]],  # Grid 8
            [points_coordinates[8], points_coordinates[17], points_coordinates[9], points_coordinates[20]], # Grid 9
            [points_coordinates[17], points_coordinates[18], points_coordinates[20], points_coordinates[20]], # Grid 10
            [points_coordinates[18], points_coordinates[19], points_coordinates[21], points_coordinates[22]], # Grid 11
            [points_coordinates[19], points_coordinates[11], points_coordinates[22], points_coordinates[12]],  # Grid 12
            [points_coordinates[9], points_coordinates[21], points_coordinates[2], points_coordinates[13]],   # Grid 13
            [points_coordinates[21], points_coordinates[12], points_coordinates[13], points_coordinates[3]]   # Grid 14
        ]
        for idx, grid in enumerate(grids):
            center_x = int((grid[0][0] + grid[1][0] + grid[2][0] + grid[3][0]) / 4)
            center_y = int((grid[0][1] + grid[1][1] + grid[2][1] + grid[3][1]) / 4)
            cv2.putText(image, f"Grid {idx + 1}", (center_x - 20, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return grids

    # Find the grid that contains the given point
    def find_grid_for_point(self, point, grid_points, image):
        grids = self.get_grid_coordinates(image, grid_points)
        for idx, grid in enumerate(grids):
            if self.is_point_in_rect(point, grid):
                return f"Point is in grid {idx + 1}"
        return "Point is not in any grid"

    # Main method to process image and find grid
    def process_image_and_find_grid(self, image, point):
        result_image, points_coordinates = self.detect_and_merge_lines(image)
        grid_result = self.find_grid_for_point(point, points_coordinates, result_image)
        cv2.circle(result_image, point, 8, (0, 0, 255), -1)
        return result_image, grid_result
0