import cv2
import numpy as np
import os

# Calculate the intersection point of two line segments
def get_line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Parallel lines have no intersection

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return int(px), int(py)

# Extend a line segment
def extend_line(x1, y1, x2, y2, scale=1000):
    line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    extension_factor = scale / line_length

    new_x1 = int(x1 - (x2 - x1) * extension_factor)
    new_y1 = int(y1 - (y2 - y1) * extension_factor)
    new_x2 = int(x2 + (x2 - x1) * extension_factor)
    new_y2 = int(y2 + (y2 - y1) * extension_factor)

    return new_x1, new_y1, new_x2, new_y2

# Merge multiple line segments and choose the longest
def merge_lines(lines):
    if len(lines) == 0:
        return None

    longest_line = max(lines, key=lambda l: np.sqrt((l[2] - l[0]) ** 2 + (l[3] - l[1]) ** 2))
    return longest_line

# Calculate division points between two points
def calculate_mask_center(mask):
    moments = cv2.moments(mask)
    if moments['m00'] != 0:
        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])
        return center_x, center_y
    else:
        return None

# Calculate division points between two points
def calculate_division_points(P1, P2, num_divisions):
    points = []
    for i in range(1, num_divisions):
        x = P1[0] + i * (P2[0] - P1[0]) // num_divisions
        y = P1[1] + i * (P2[1] - P1[1]) // num_divisions
        points.append((x, y))
    return points

# Draw division points
def draw_division_points(image, corners):
    # Assign the four corners as P1-P4 (top-left, top-right, bottom-left, bottom-right)
    P1 = corners[0]
    P2 = corners[1]
    P3 = corners[2]
    P4 = corners[3]

    # Calculate division points: P1-P2 and P1-P3 are divided into four parts, P3-P4 into two
    div_points_p1_p2 = calculate_division_points(P1, P2, 4)
    div_points_p1_p3 = calculate_division_points(P1, P3, 4)
    div_points_p2_p4 = calculate_division_points(P2, P4, 4)
    div_points_p3_p4 = calculate_division_points(P3, P4, 2)

    # Draw division points
    for idx, point in enumerate(div_points_p1_p2):
        cv2.circle(image, point, 5, (0, 255, 0), -1)  # Green point
        cv2.putText(image, f"Div P1-P2-{idx+1}", (point[0] + 5, point[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for idx, point in enumerate(div_points_p1_p3):
        cv2.circle(image, point, 5, (0, 255, 0), -1)  # Green point
        cv2.putText(image, f"Div P1-P3-{idx+1}", (point[0] + 5, point[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    for idx, point in enumerate(div_points_p2_p4):
        cv2.circle(image, point, 5, (0, 255, 0), -1)  # Green point
        cv2.putText(image, f"Div P2-P4-{idx+1}", (point[0] + 5, point[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for idx, point in enumerate(div_points_p3_p4):
        cv2.circle(image, point, 5, (0, 255, 0), -1)  # Green point
        cv2.putText(image, f"Div P3-P4-{idx+1}", (point[0] + 5, point[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        
    # Connect division points on the third row (third division point of P1-P3 and P2-P4)
    for i in range(1, 4):
        start_point = div_points_p1_p3[i - 1]
        end_point = div_points_p2_p4[i - 1]
        cv2.line(image, start_point, end_point, (255, 0, 0), 2)  # Blue lines to connect

    # Create lists to store division points of each row
    row1_division_points = []
    row2_division_points = []
    row3_division_points = []
    
    # Further divide the connected points on the third row into four divisions and name them by row
    for row_index in range(1, 4):  # Three rows
        start_point = div_points_p1_p3[row_index - 1]
        end_point = div_points_p2_p4[row_index - 1]
        division_points_on_third_row = calculate_division_points(start_point, end_point, 4)
    
        # Store points in different lists based on row number and draw the division points on the image
        if row_index == 1:
            row1_division_points = division_points_on_third_row
        elif row_index == 2:
            row2_division_points = division_points_on_third_row
        elif row_index == 3:
            row3_division_points = division_points_on_third_row
    
        # Draw division points on the image and label by row
        for idx, div_point in enumerate(division_points_on_third_row):
            cv2.circle(image, div_point, 5, (255, 255, 0), -1)  # Yellow point
            cv2.putText(image, f"Row{row_index} Div-{idx+1}", (div_point[0] + 5, div_point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    # Connect the following points
    # 1. P1, Div P1-P2-1, Div P1-P2-2, Div P1-P2-3, P2
    cv2.line(image, P1, div_points_p1_p2[0], (255, 0, 0), 2)
    cv2.line(image, div_points_p1_p2[0], div_points_p1_p2[1], (255, 0, 0), 2)
    cv2.line(image, div_points_p1_p2[1], div_points_p1_p2[2], (255, 0, 0), 2)
    cv2.line(image, div_points_p1_p2[2], P2, (255, 0, 0), 2)

    # 2. P1, Div P1-P3-1, Div P1-P3-2, Div P1-P3-3, P3
    cv2.line(image, P1, div_points_p1_p3[0], (255, 0, 0), 2)
    cv2.line(image, div_points_p1_p3[0], div_points_p1_p3[1], (255, 0, 0), 2)
    cv2.line(image, div_points_p1_p3[1], div_points_p1_p3[2], (255, 0, 0), 2)
    cv2.line(image, div_points_p1_p3[2], P3, (255, 0, 0), 2)

    # 3. P2, Div P2-P4-1, Div P2-P4-2, Div P2-P4-3, P4
    cv2.line(image, P2, div_points_p2_p4[0], (255, 0, 0), 2)
    cv2.line(image, div_points_p2_p4[0], div_points_p2_p4[1], (255, 0, 0), 2)
    cv2.line(image, div_points_p2_p4[1], div_points_p2_p4[2], (255, 0, 0), 2)
    cv2.line(image, div_points_p2_p4[2], P4, (255, 0, 0), 2)

    # 4. P3, Div P3-P4-1, P4
    cv2.line(image, P3, div_points_p3_p4[0], (255, 0, 0), 2)
    cv2.line(image, div_points_p3_p4[0], P4, (255, 0, 0), 2)

    # 5. Div P1-P2-1, Row3 Div-1 (connect points top to bottom)
    cv2.line(image, div_points_p1_p2[0], row1_division_points[0], (255, 0, 0), 2)
    cv2.line(image, row1_division_points[0], row2_division_points[0], (255, 0, 0), 2)
    cv2.line(image, row2_division_points[0], row3_division_points[0], (255, 0, 0), 2)
    

    # 6. Div P1-P2-2, Row3 Div-2 (connect points top to bottom)
    cv2.line(image, div_points_p1_p2[1], row1_division_points[1], (255, 0, 0), 2)
    cv2.line(image, row1_division_points[1], row2_division_points[1], (255, 0, 0), 2)
    cv2.line(image, row2_division_points[1], row3_division_points[1], (255, 0, 0), 2)

    # 7. Div P1-P2-3, Row3 Div-3 (connect points top to bottom)
    cv2.line(image, div_points_p1_p2[2], row1_division_points[2], (255, 0, 0), 2)
    cv2.line(image, row1_division_points[2], row2_division_points[2], (255, 0, 0), 2)
    cv2.line(image, row2_division_points[2], row3_division_points[2], (255, 0, 0), 2)

    # 8. Row3 Div-2, Div P3-P4-1
    cv2.line(image, division_points_on_third_row[1], div_points_p3_p4[0], (255, 0, 0), 2)

    return image


# Filter out points outside the image range
def filter_points_in_image(points, image_shape):
    filtered_points = []
    height, width = image_shape[:2]

    for point in points:
        x, y = point
        if 0 <= x < width and 0 <= y < height:
            filtered_points.append(point)

    return filtered_points

# Detect lines, merge them, find corners, and draw them on the image
def detect_and_merge_lines(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        # print("No orange area found")
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    mask_center = calculate_mask_center(mask)
    if mask_center:
        center_x, center_y = mask_center
        cv2.circle(image, (center_x, center_y), 5, (255, 255, 255), -1)
        cv2.putText(image, f"Center ({center_x}, {center_y})", (center_x + 10, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Define four extended regions
    top_region = (int(x + 0.2 * w), y - int(0.3 * h), int(x + 0.8 * w), y + int(0.2 * h))
    bottom_region = (int(x + 0.2 * w), y + h - int(0.2 * h), int(x + 0.8 * w), y + h + int(0.3 * h))
    left_region = (x - int(0.3 * w), int(y + 0.2 * h), x + int(0.2 * w), int(y + 0.8 * h))
    right_region = (x + w - int(0.2 * w), int(y + 0.2 * h), x + w + int(0.3 * w), int(y + 0.8 * h))

    regions = [(top_region, 'horizontal'), (bottom_region, 'horizontal'),
               (left_region, 'vertical'), (right_region, 'vertical')]

    height, width = mask.shape[:2]  # Get image dimensions for boundary checking

    for region, _ in regions:
        # Boundary check to ensure the region does not exceed image dimensions
        x_start = max(0, region[0])
        y_start = max(0, region[1])
        x_end = min(width, region[2])
        y_end = min(height, region[3])

        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    all_lines = []
    for region, direction in regions:
        x_start = max(0, region[0])
        y_start = max(0, region[1])
        x_end = min(width, region[2])
        y_end = min(height, region[3])

        # Extract the mask region
        mask_region = mask[y_start:y_end, x_start:x_end]

        if mask_region.size == 0:  # Skip if the region is empty
            # print(f"{direction.capitalize()} region is empty, skipping.")
            continue

        # Use smoothing to reduce edge noise
        mask_region_smooth = cv2.GaussianBlur(mask_region, (5, 5), 0)
        edges = cv2.Canny(mask_region_smooth, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=15)

        if lines is not None:
            lines = [line[0] for line in lines]
            merged_line = merge_lines(lines)
            if merged_line is not None:
                # print(f"{direction.capitalize()} region detected and merged 1 line")
                extended_line = extend_line(*merged_line)
                cv2.line(image, (extended_line[0] + x_start, extended_line[1] + y_start),
                          (extended_line[2] + x_start, extended_line[3] + y_start), (255, 0, 0), 2)
                all_lines.append([extended_line[0] + x_start, extended_line[1] + y_start, 
                                  extended_line[2] + x_start, extended_line[3] + y_start])

    if len(all_lines) >= 4:
        corners = []
        for i in range(len(all_lines)):
            for j in range(i + 1, len(all_lines)):
                intersection = get_line_intersection(all_lines[i], all_lines[j])
                if intersection is not None:
                    corners.append(intersection)

        filtered_corners = filter_points_in_image(corners, image.shape)

        for idx, point in enumerate(filtered_corners[:4]):
            cv2.circle(image, point, 5, (0, 0, 255), -1)
            cv2.putText(image, f"P{idx + 1} ({point[0]}, {point[1]})", (point[0] + 5, point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if len(filtered_corners) == 4:
            draw_division_points(image, filtered_corners)

    return image


lower_orange = np.array([5, 100, 100])  # Lower bound for HSV color
upper_orange = np.array([15, 255, 255])  # Upper bound for HSV color

# image = cv2.imread('./frames_output_text/1.000.jpg')
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

# result_image = detect_and_merge_lines(image, mask)

# cv2.imshow('Merged Lines with Corners and Center', result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Define input and output folder paths
input_folder = './frames_output_text'
output_folder = './output_grid'  

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Process jpg and png images
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to read image: {filename}")
            continue

        # Convert to HSV image and create a mask for the orange region
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

        # Process the image and draw division points and lines
        result_image = detect_and_merge_lines(image, mask)

        # Save the processed image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, result_image)
        # print(f"Saved processed image: {output_path}")

print("Batch processing completed!")





# import cv2
# import numpy as np

# def get_line_intersection(line1, line2):
#     x1, y1, x2, y2 = line1
#     x3, y3, x4, y4 = line2

#     denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
#     if denom == 0:
#         return None  # Parallel lines have no intersection

#     px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
#     py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
#     return int(px), int(py)

# def extend_line(x1, y1, x2, y2, scale=1000):
#     line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#     extension_factor = scale / line_length

#     new_x1 = int(x1 - (x2 - x1) * extension_factor)
#     new_y1 = int(y1 - (y2 - y1) * extension_factor)
#     new_x2 = int(x2 + (x2 - x1) * extension_factor)
#     new_y2 = int(y2 + (y2 - y1) * extension_factor)

#     return new_x1, new_y1, new_x2, new_y2

# def merge_lines(lines):
#     if len(lines) == 0:
#         return None

#     longest_line = max(lines, key=lambda l: np.sqrt((l[2] - l[0]) ** 2 + (l[3] - l[1]) ** 2))
#     return longest_line

# def calculate_mask_center(mask):
#     moments = cv2.moments(mask)
#     if moments['m00'] != 0:
#         center_x = int(moments['m10'] / moments['m00'])
#         center_y = int(moments['m01'] / moments['m00'])
#         return center_x, center_y
#     else:
#         return None

# def calculate_division_points(P1, P2, num_divisions):
#     points = []
#     for i in range(1, num_divisions):
#         x = P1[0] + i * (P2[0] - P1[0]) // num_divisions
#         y = P1[1] + i * (P2[1] - P1[1]) // num_divisions
#         points.append((x, y))
#     return points


# def draw_division_points(image, corners):
#     # Assign the four corners as P1-P4 (top-left, top-right, bottom-left, bottom-right)
#     P1 = corners[0]
#     P2 = corners[1]
#     P3 = corners[2]
#     P4 = corners[3]

#     # Calculate division points: P1-P2 and P1-P3 are divided into four parts, P3-P4 into two
#     div_points_p1_p2 = calculate_division_points(P1, P2, 4)
#     div_points_p1_p3 = calculate_division_points(P1, P3, 4)
#     div_points_p2_p4 = calculate_division_points(P2, P4, 4)
#     div_points_p3_p4 = calculate_division_points(P3, P4, 2)

#     # Draw division points
#     for idx, point in enumerate(div_points_p1_p2):
#         cv2.circle(image, point, 5, (0, 255, 0), -1)  # Green point
#         cv2.putText(image, f"Div P1-P2-{idx+1}", (point[0] + 5, point[1] - 5), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     for idx, point in enumerate(div_points_p1_p3):
#         cv2.circle(image, point, 5, (0, 255, 0), -1)  # Green point
#         cv2.putText(image, f"Div P1-P3-{idx+1}", (point[0] + 5, point[1] - 5), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
#     for idx, point in enumerate(div_points_p2_p4):
#         cv2.circle(image, point, 5, (0, 255, 0), -1)  # Green point
#         cv2.putText(image, f"Div P2-P4-{idx+1}", (point[0] + 5, point[1] - 5), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     for idx, point in enumerate(div_points_p3_p4):
#         cv2.circle(image, point, 5, (0, 255, 0), -1)  # Green point
#         cv2.putText(image, f"Div P3-P4-{idx+1}", (point[0] + 5, point[1] - 5), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        
#     # Connect division points on the third row (third division point of P1-P3 and P2-P4)
#     for i in range(1, 4):
#         start_point = div_points_p1_p3[i - 1]
#         end_point = div_points_p2_p4[i - 1]
#         cv2.line(image, start_point, end_point, (255, 0, 0), 2)  # Blue lines to connect

#     # Create lists to store division points of each row
#     row1_division_points = []
#     row2_division_points = []
#     row3_division_points = []
    
#     # Further divide the connected points on the third row into four divisions and name them by row
#     for row_index in range(1, 4):  # Three rows
#         start_point = div_points_p1_p3[row_index - 1]
#         end_point = div_points_p2_p4[row_index - 1]
#         division_points_on_third_row = calculate_division_points(start_point, end_point, 4)
    
#         # Store points in different lists based on row number and draw the division points on the image
#         if row_index == 1:
#             row1_division_points = division_points_on_third_row
#         elif row_index == 2:
#             row2_division_points = division_points_on_third_row
#         elif row_index == 3:
#             row3_division_points = division_points_on_third_row
    
#         # Draw division points on the image and label by row
#         for idx, div_point in enumerate(division_points_on_third_row):
#             cv2.circle(image, div_point, 5, (255, 255, 0), -1)  # Yellow point
#             cv2.putText(image, f"Row{row_index} Div-{idx+1}", (div_point[0] + 5, div_point[1] - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


#     # Connect the following points
#     # 1. P1, Div P1-P2-1, Div P1-P2-2, Div P1-P2-3, P2
#     cv2.line(image, P1, div_points_p1_p2[0], (255, 0, 0), 2)
#     cv2.line(image, div_points_p1_p2[0], div_points_p1_p2[1], (255, 0, 0), 2)
#     cv2.line(image, div_points_p1_p2[1], div_points_p1_p2[2], (255, 0, 0), 2)
#     cv2.line(image, div_points_p1_p2[2], P2, (255, 0, 0), 2)

#     # 2. P1, Div P1-P3-1, Div P1-P3-2, Div P1-P3-3, P3
#     cv2.line(image, P1, div_points_p1_p3[0], (255, 0, 0), 2)
#     cv2.line(image, div_points_p1_p3[0], div_points_p1_p3[1], (255, 0, 0), 2)
#     cv2.line(image, div_points_p1_p3[1], div_points_p1_p3[2], (255, 0, 0), 2)
#     cv2.line(image, div_points_p1_p3[2], P3, (255, 0, 0), 2)

#     # 3. P2, Div P2-P4-1, Div P2-P4-2, Div P2-P4-3, P4
#     cv2.line(image, P2, div_points_p2_p4[0], (255, 0, 0), 2)
#     cv2.line(image, div_points_p2_p4[0], div_points_p2_p4[1], (255, 0, 0), 2)
#     cv2.line(image, div_points_p2_p4[1], div_points_p2_p4[2], (255, 0, 0), 2)
#     cv2.line(image, div_points_p2_p4[2], P4, (255, 0, 0), 2)

#     # 4. P3, Div P3-P4-1, P4
#     cv2.line(image, P3, div_points_p3_p4[0], (255, 0, 0), 2)
#     cv2.line(image, div_points_p3_p4[0], P4, (255, 0, 0), 2)

#     # 5. Div P1-P2-1, Row3 Div-1 (connect points top to bottom)
#     cv2.line(image, div_points_p1_p2[0], row1_division_points[0], (255, 0, 0), 2)
#     cv2.line(image, row1_division_points[0], row2_division_points[0], (255, 0, 0), 2)
#     cv2.line(image, row2_division_points[0], row3_division_points[0], (255, 0, 0), 2)
    
#     # 6. Div P1-P2-2, Row3 Div-2 (connect points top to bottom)
#     cv2.line(image, div_points_p1_p2[1], row1_division_points[1], (255, 0, 0), 2)
#     cv2.line(image, row1_division_points[1], row2_division_points[1], (255, 0, 0), 2)
#     cv2.line(image, row2_division_points[1], row3_division_points[1], (255, 0, 0), 2)

#     # 7. Div P1-P2-3, Row3 Div-3 (connect points top to bottom)
#     cv2.line(image, div_points_p1_p2[2], row1_division_points[2], (255, 0, 0), 2)
#     cv2.line(image, row1_division_points[2], row2_division_points[2], (255, 0, 0), 2)
#     cv2.line(image, row2_division_points[2], row3_division_points[2], (255, 0, 0), 2)

#     # 8. Row3 Div-2, Div P3-P4-1
#     cv2.line(image, division_points_on_third_row[1], div_points_p3_p4[0], (255, 0, 0), 2)

#     return image


# # Detect and merge lines on an image based on the mask
# def detect_and_merge_lines(image, mask):
#     # Find contours in the orange region
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if len(contours) == 0:
#         print("No orange region found")
#         return image

#     # Find the largest contour and get the bounding box
#     largest_contour = max(contours, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(largest_contour)

#     # Calculate the center of the mask
#     mask_center = calculate_mask_center(mask)
#     if mask_center:
#         center_x, center_y = mask_center
#         # Draw the centroid on the image
#         cv2.circle(image, (center_x, center_y), 5, (255, 255, 255), -1)  # Green center point
#         cv2.putText(image, f"Center ({center_x}, {center_y})", (center_x + 10, center_y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     # Manually set the corner points P1-P4 (replace with detected corner points)
#     P1 = (649, 246)  # Replace with actual detected corner point coordinates
#     P2 = (1206, 263)
#     P3 = (559, 717)
#     P4 = (1247, 732)

#     corners = [P1, P2, P3, P4]

#     # Draw the corner points on the image
#     for idx, point in enumerate([P1, P2, P3, P4]):
#         cv2.circle(image, point, 5, (0, 0, 255), -1)  # Red corner point
#         cv2.putText(image, f"P{idx + 1} ({point[0]}, {point[1]})", (point[0] + 5, point[1] - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     # Draw division points
#     image = draw_division_points(image, corners)

#     return image


# lower_orange = np.array([5, 100, 100])  # Lower bound for HSV color
# upper_orange = np.array([15, 255, 255])  # Upper bound for HSV color

# # Read the image
# image = cv2.imread('./frames_output_text_orangemask/1.000.jpg')

# # Create a mask to mark the orange region
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

# # Detect and merge lines, filter corner points, and display division points in the central 40% of the region
# result_image = detect_and_merge_lines(image, mask)

# # Display the final result
# cv2.imshow('Final Result with Division Points', result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()







# import cv2
# import numpy as np

# # Calculate the intersection point of two line segments
# def get_line_intersection(line1, line2):
#     x1, y1, x2, y2 = line1
#     x3, y3, x4, y4 = line2

#     denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
#     if denom == 0:
#         return None  # Parallel lines have no intersection

#     px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
#     py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
#     return int(px), int(py)

# # Extend a line segment
# def extend_line(x1, y1, x2, y2, scale=1000):
#     line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#     extension_factor = scale / line_length

#     new_x1 = int(x1 - (x2 - x1) * extension_factor)
#     new_y1 = int(y1 - (y2 - y1) * extension_factor)
#     new_x2 = int(x2 + (x2 - x1) * extension_factor)
#     new_y2 = int(y2 + (y2 - y1) * extension_factor)

#     return new_x1, new_y1, new_x2, new_y2

# # Merge multiple line segments and choose the longest
# def merge_lines(lines):
#     if len(lines) == 0:
#         return None

#     longest_line = max(lines, key=lambda l: np.sqrt((l[2] - l[0]) ** 2 + (l[3] - l[1]) ** 2))
#     return longest_line

# # Calculate the center of a mask
# def calculate_mask_center(mask):
#     moments = cv2.moments(mask)
#     if moments['m00'] != 0:
#         center_x = int(moments['m10'] / moments['m00'])
#         center_y = int(moments['m01'] / moments['m00'])
#         return center_x, center_y
#     else:
#         return None

# # Detect and merge lines on an image based on the mask
# def detect_and_merge_lines(image, mask):
#     # Find contours in the orange region
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if len(contours) == 0:
#         print("No orange region found")
#         return image

#     # Find the largest contour and get the bounding box
#     largest_contour = max(contours, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(largest_contour)

#     # Calculate the center of the mask
#     mask_center = calculate_mask_center(mask)
#     if mask_center:
#         center_x, center_y = mask_center
#         # Draw the centroid on the image
#         cv2.circle(image, (center_x, center_y), 5, (255, 255, 255), -1)  # Green center point
#         cv2.putText(image, f"Center ({center_x}, {center_y})", (center_x + 10, center_y - 10),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     # Expand the size of the bounding box
#     top_region = (int(x + 0.2 * w), y - int(0.3 * h), int(x + 0.8 * w), y + int(0.2 * h))  # Top extension
#     bottom_region = (int(x + 0.2 * w), y + h - int(0.2 * h), int(x + 0.8 * w), y + h + int(0.3 * h))  # Bottom extension
#     left_region = (x - int(0.3 * w), int(y + 0.2 * h), x + int(0.2 * w), int(y + 0.8 * h))  # Left extension
#     right_region = (x + w - int(0.2 * w), int(y + 0.2 * h), x + w + int(0.3 * w), int(y + 0.8 * h))  # Right extension

#     regions = [(top_region, 'horizontal'), (bottom_region, 'horizontal'),
#                 (left_region, 'vertical'), (right_region, 'vertical')]

#     # Draw these detection regions on the image (for visualization)
#     for region, _ in regions:
#         cv2.rectangle(image, (region[0], region[1]), (region[2], region[3]), (0, 255, 0), 2)

#     all_lines = []
#     # Perform line detection on each region
#     for region, direction in regions:
#         # Extract the mask for the region
#         mask_region = mask[region[1]:region[3], region[0]:region[2]]

#         # Use smoothing to reduce edge noise
#         mask_region_smooth = cv2.GaussianBlur(mask_region, (5, 5), 0)

#         # Canny edge detection
#         edges = cv2.Canny(mask_region_smooth, 50, 150, apertureSize=3)

#         # Hough transform to detect lines, adjust parameters for accuracy
#         lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=15)

#         if lines is not None:
#             lines = [line[0] for line in lines]
#             merged_line = merge_lines(lines)
#             if merged_line is not None:
#                 print(f"{direction.capitalize()} region detected and merged 1 line")

#                 # Extend the line and draw it
#                 x1, y1, x2, y2 = merged_line
#                 extended_line = extend_line(x1, y1, x2, y2)
#                 cv2.line(image, (extended_line[0] + region[0], extended_line[1] + region[1]),
#                           (extended_line[2] + region[0], extended_line[3] + region[1]), (255, 0, 0), 2)
#                 all_lines.append([extended_line[0] + region[0], extended_line[1] + region[1], 
#                                   extended_line[2] + region[0], extended_line[3] + region[1]])

#     # Calculate and draw corner points
#     if len(all_lines) >= 4:
#         corners = []
#         for i in range(len(all_lines)):
#             for j in range(i + 1, len(all_lines)):
#                 intersection = get_line_intersection(all_lines[i], all_lines[j])
#                 if intersection is not None:
#                     corners.append(intersection)

#         # Filter out corner points outside the image boundaries
#         filtered_corners = filter_points_in_image(corners, image.shape)

#         # Draw valid corner points
#         for idx, point in enumerate(filtered_corners[:4]):
#             cv2.circle(image, point, 5, (0, 0, 255), -1)  # Red corner point
#             cv2.putText(image, f"P{idx + 1} ({point[0]}, {point[1]})", (point[0] + 5, point[1] - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#         print(f"Number of valid corner points: {len(filtered_corners)}")
#     else:
#         print("Not enough lines detected to calculate corner points.")

#     return image

# # Filter out points outside the image boundaries
# def filter_points_in_image(points, image_shape):
#     filtered_points = []
#     height, width = image_shape[:2]

#     for point in points:
#         x, y = point
#         if 0 <= x < width and 0 <= y < height:
#             filtered_points.append(point)

#     return filtered_points


# lower_orange = np.array([5, 100, 100])  # Lower bound for HSV color
# upper_orange = np.array([15, 255, 255])  # Upper bound for HSV color

# # Read the image
# image = cv2.imread('./frames_output_text/1.000.jpg') 

# # Create a mask to mark the orange region
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

# # Detect and merge lines in the central 40% region, filter corner points, and display the centroid
# result_image = detect_and_merge_lines(image, mask)

# # Display the final result
# cv2.imshow('Merged Lines with Corners and Center', result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()







