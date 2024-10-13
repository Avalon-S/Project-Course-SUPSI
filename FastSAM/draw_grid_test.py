import cv2
import numpy as np
import os

# 计算两条线段的交点
def get_line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # 平行线无交点

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return int(px), int(py)

# 将线段延长
def extend_line(x1, y1, x2, y2, scale=1000):
    line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    extension_factor = scale / line_length

    new_x1 = int(x1 - (x2 - x1) * extension_factor)
    new_y1 = int(y1 - (y2 - y1) * extension_factor)
    new_x2 = int(x2 + (x2 - x1) * extension_factor)
    new_y2 = int(y2 + (y2 - y1) * extension_factor)

    return new_x1, new_y1, new_x2, new_y2

# 合并多条线段，选择最长的
def merge_lines(lines):
    if len(lines) == 0:
        return None

    longest_line = max(lines, key=lambda l: np.sqrt((l[2] - l[0]) ** 2 + (l[3] - l[1]) ** 2))
    return longest_line

# 计算二值图像的质心
def calculate_mask_center(mask):
    moments = cv2.moments(mask)
    if moments['m00'] != 0:
        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])
        return center_x, center_y
    else:
        return None

# 计算两点之间的等分点
def calculate_division_points(P1, P2, num_divisions):
    points = []
    for i in range(1, num_divisions):
        x = P1[0] + i * (P2[0] - P1[0]) // num_divisions
        y = P1[1] + i * (P2[1] - P1[1]) // num_divisions
        points.append((x, y))
    return points

# 绘制等分点并返回它们的坐标
def draw_division_points(image, corners):
    points_coordinates = []  # 用于存储所有点的坐标

    # 将四个角点分别为P1-P4（左上，右上，左下，右下）
    P1 = corners[0]
    P2 = corners[1]
    P3 = corners[2]
    P4 = corners[3]

    # 先将角点存储到坐标列表中
    points_coordinates.extend([P1, P2, P3, P4])

    # 在图像上显示角点的坐标
    for idx, point in enumerate([P1, P2, P3, P4]):
        cv2.putText(image, f"P{idx+1} ({point[0]}, {point[1]})", (point[0] + 5, point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 计算等分点：P1-P2和P1-P3四等分，P3-P4二等分
    div_points_p1_p2 = calculate_division_points(P1, P2, 4)
    div_points_p1_p3 = calculate_division_points(P1, P3, 4)
    div_points_p2_p4 = calculate_division_points(P2, P4, 4)
    div_points_p3_p4 = calculate_division_points(P3, P4, 2)

    # 绘制等分点，并存储它们的坐标
    for idx, point in enumerate(div_points_p1_p2):
        cv2.circle(image, point, 5, (0, 255, 0), -1)  # 绿色点
        cv2.putText(image, f"Div P1-P2-{idx+1} ({point[0]}, {point[1]})", (point[0] + 5, point[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        points_coordinates.append(point)  # 存储坐标

    for idx, point in enumerate(div_points_p1_p3):
        cv2.circle(image, point, 5, (0, 255, 0), -1)  # 绿色点
        cv2.putText(image, f"Div P1-P3-{idx+1} ({point[0]}, {point[1]})", (point[0] + 5, point[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        points_coordinates.append(point)  # 存储坐标
        
    for idx, point in enumerate(div_points_p2_p4):
        cv2.circle(image, point, 5, (0, 255, 0), -1)  # 绿色点
        cv2.putText(image, f"Div P2-P4-{idx+1} ({point[0]}, {point[1]})", (point[0] + 5, point[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        points_coordinates.append(point)  # 存储坐标

    for idx, point in enumerate(div_points_p3_p4):
        cv2.circle(image, point, 5, (0, 255, 0), -1)  # 绿色点
        cv2.putText(image, f"Div P3-P4-{idx+1} ({point[0]}, {point[1]})", (point[0] + 5, point[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        points_coordinates.append(point)  # 存储坐标
        
    # 连接第三行的等分点 (P1-P3 和 P2-P4 的第三个点)
    for i in range(1, 4):
        start_point = div_points_p1_p3[i - 1]
        end_point = div_points_p2_p4[i - 1]
        cv2.line(image, start_point, end_point, (255, 0, 0), 2)  # 蓝色线连接

    # 创建用于存储各行分割点的列表
    row1_division_points = []
    row2_division_points = []
    row3_division_points = []
    
    # 进一步对第三行的连接进行四等分并为不同的行命名
    for row_index in range(1, 4):  # 三行
        start_point = div_points_p1_p3[row_index - 1]
        end_point = div_points_p2_p4[row_index - 1]
        division_points_on_third_row = calculate_division_points(start_point, end_point, 4)
    
        # 根据行号，将点存储到不同的列表中，并在图像上绘制等分点
        if row_index == 1:
            row1_division_points = division_points_on_third_row
        elif row_index == 2:
            row2_division_points = division_points_on_third_row
        elif row_index == 3:
            row3_division_points = division_points_on_third_row
    
        # 在图像上绘制等分点，并按行号命名
        for idx, div_point in enumerate(division_points_on_third_row):
            cv2.circle(image, div_point, 5, (255, 255, 0), -1)  # 黄色点
            cv2.putText(image, f"Row{row_index} Div-{idx+1} ({div_point[0]}, {div_point[1]})", (div_point[0] + 5, div_point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            points_coordinates.append(div_point)  # 存储坐标

    # 连接以下几组点
    # 1. P1，Div P1-P2-1, Div P1-P2-2, Div P1-P2-3, P2
    cv2.line(image, P1, div_points_p1_p2[0], (255, 0, 0), 2)
    cv2.line(image, div_points_p1_p2[0], div_points_p1_p2[1], (255, 0, 0), 2)
    cv2.line(image, div_points_p1_p2[1], div_points_p1_p2[2], (255, 0, 0), 2)
    cv2.line(image, div_points_p1_p2[2], P2, (255, 0, 0), 2)

    # 2. P1，Div P1-P3-1，Div P1-P3-2，Div P1-P3-3，P3
    cv2.line(image, P1, div_points_p1_p3[0], (255, 0, 0), 2)
    cv2.line(image, div_points_p1_p3[0], div_points_p1_p3[1], (255, 0, 0), 2)
    cv2.line(image, div_points_p1_p3[1], div_points_p1_p3[2], (255, 0, 0), 2)
    cv2.line(image, div_points_p1_p3[2], P3, (255, 0, 0), 2)

    # 3. P2，Div P2-P4-1，Div P2-P4-2，Div P2-P4-3，P4
    cv2.line(image, P2, div_points_p2_p4[0], (255, 0, 0), 2)
    cv2.line(image, div_points_p2_p4[0], div_points_p2_p4[1], (255, 0, 0), 2)
    cv2.line(image, div_points_p2_p4[1], div_points_p2_p4[2], (255, 0, 0), 2)
    cv2.line(image, div_points_p2_p4[2], P4, (255, 0, 0), 2)

    # 4. P3，Div P3-P4-1，P4
    cv2.line(image, P3, div_points_p3_p4[0], (255, 0, 0), 2)
    cv2.line(image, div_points_p3_p4[0], P4, (255, 0, 0), 2)

    # 5. Div P1-P2-1，Row3 Div-1（上下两个点）
    cv2.line(image, div_points_p1_p2[0], row1_division_points[0], (255, 0, 0), 2)
    cv2.line(image, row1_division_points[0], row2_division_points[0], (255, 0, 0), 2)
    cv2.line(image, row2_division_points[0], row3_division_points[0], (255, 0, 0), 2)
    

    # 6. Div P1-P2-2，Row3 Div-2（上下两个点）
    cv2.line(image, div_points_p1_p2[1], row1_division_points[1], (255, 0, 0), 2)
    cv2.line(image, row1_division_points[1], row2_division_points[1], (255, 0, 0), 2)
    cv2.line(image, row2_division_points[1], row3_division_points[1], (255, 0, 0), 2)

    # 7. Div P1-P2-3，Row3 Div-3（上下两个点）
    cv2.line(image, div_points_p1_p2[2], row1_division_points[2], (255, 0, 0), 2)
    cv2.line(image, row1_division_points[2], row2_division_points[2], (255, 0, 0), 2)
    cv2.line(image, row2_division_points[2], row3_division_points[2], (255, 0, 0), 2)

    # 8. Row3 Div-2，Div P3-P4-1
    cv2.line(image, division_points_on_third_row[1], div_points_p3_p4[0], (255, 0, 0), 2)

    # 返回所有点的坐标
    return points_coordinates


# 过滤掉图像范围外的点
def filter_points_in_image(points, image_shape):
    filtered_points = []
    height, width = image_shape[:2]

    for point in points:
        x, y = point
        if 0 <= x < width and 0 <= y < height:
            filtered_points.append(point)

    return filtered_points

# 在detect_and_merge_lines中调用时获取23个点
def detect_and_merge_lines(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return image, []

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    mask_center = calculate_mask_center(mask)
    if mask_center:
        center_x, center_y = mask_center
        cv2.circle(image, (center_x, center_y), 5, (255, 255, 255), -1)
        cv2.putText(image, f"Center ({center_x}, {center_y})", (center_x + 10, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    top_region = (int(x + 0.2 * w), y - int(0.3 * h), int(x + 0.8 * w), y + int(0.2 * h))
    bottom_region = (int(x + 0.2 * w), y + h - int(0.2 * h), int(x + 0.8 * w), y + h + int(0.3 * h))
    left_region = (x - int(0.3 * w), int(y + 0.2 * h), x + int(0.2 * w), int(y + 0.8 * h))
    right_region = (x + w - int(0.2 * w), int(y + 0.2 * h), x + w + int(0.3 * w), int(y + 0.8 * h))

    regions = [(top_region, 'horizontal'), (bottom_region, 'horizontal'),
               (left_region, 'vertical'), (right_region, 'vertical')]

    height, width = mask.shape[:2]

    for region, _ in regions:
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

        mask_region = mask[y_start:y_end, x_start:x_end]

        if mask_region.size == 0:
            continue

        mask_region_smooth = cv2.GaussianBlur(mask_region, (5, 5), 0)
        edges = cv2.Canny(mask_region_smooth, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=15)

        if lines is not None:
            lines = [line[0] for line in lines]
            merged_line = merge_lines(lines)
            if merged_line is not None:
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

        if len(filtered_corners) == 4:
            # 传递角点并获取 23 个点的坐标
            points_coordinates = draw_division_points(image, filtered_corners)
            return image, points_coordinates

    return image, []


# 辅助函数，判断点是否在四边形内
def is_point_in_rect(p, rect):
    """
    判断点 p 是否在由 rect 定义的四边形内，按照左上，右上，左下，右下顺序
    :param p: 输入的点 (x, y)
    :param rect: 四边形的四个顶点 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    :return: True 如果点在四边形内, 否则 False
    """
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    # 使用新的顶点顺序：左上(x1, y1), 右上(x2, y2), 左下(x3, y3), 右下(x4, y4)
    d1 = sign(p, rect[0], rect[1])  # 左上 -> 右上
    d2 = sign(p, rect[1], rect[3])  # 右上 -> 右下
    d3 = sign(p, rect[3], rect[2])  # 右下 -> 左下
    d4 = sign(p, rect[2], rect[0])  # 左下 -> 左上

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0) or (d4 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0) or (d4 > 0)

    return not (has_neg and has_pos)


# 获取所有格子的顶点坐标
def get_grid_coordinates(image, points_coordinates):
    grids = [
        [points_coordinates[0], points_coordinates[4], points_coordinates[7], points_coordinates[14]],  # 格子1
        [points_coordinates[4], points_coordinates[5], points_coordinates[14], points_coordinates[15]],  # 格子2
        [points_coordinates[5], points_coordinates[6], points_coordinates[15], points_coordinates[16]],  # 格子3
        [points_coordinates[6], points_coordinates[1], points_coordinates[16], points_coordinates[10]],   # 格子4
        [points_coordinates[7], points_coordinates[14], points_coordinates[8], points_coordinates[17]],  # 格子5
        [points_coordinates[14], points_coordinates[15], points_coordinates[17], points_coordinates[18]], # 格子6
        [points_coordinates[15], points_coordinates[16], points_coordinates[18], points_coordinates[19]], # 格子7
        [points_coordinates[16], points_coordinates[10], points_coordinates[19], points_coordinates[11]],  # 格子8
        [points_coordinates[8], points_coordinates[17], points_coordinates[9], points_coordinates[20]], # 格子9
        [points_coordinates[17], points_coordinates[18], points_coordinates[20], points_coordinates[20]], # 格子10
        [points_coordinates[18], points_coordinates[19], points_coordinates[21], points_coordinates[22]], # 格子11
        [points_coordinates[19], points_coordinates[11], points_coordinates[22], points_coordinates[12]],  # 格子12
        [points_coordinates[9], points_coordinates[21], points_coordinates[2], points_coordinates[13]],   # 格子13
        [points_coordinates[21], points_coordinates[12], points_coordinates[13], points_coordinates[3]]   # 格子14
    ]
    
    # 遍历每个格子，并在图像的格子中心显示格子编号
    for idx, grid in enumerate(grids):
        # 计算格子中心
        center_x = int((grid[0][0] + grid[1][0] + grid[2][0] + grid[3][0]) / 4)
        center_y = int((grid[0][1] + grid[1][1] + grid[2][1] + grid[3][1]) / 4)
        
        # 在格子中心绘制编号
        cv2.putText(image, f"Grid {idx + 1}", (center_x - 20, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return grids



# 判断输入坐标在哪个小格子中
def find_grid_for_point(point, grid_points,image):
    """
    根据14个格子顶点，找到输入点所在的小格子
    :param point: 输入点 (x, y)
    :param grid_points: 23个点坐标的列表
    :return: 找到的格子编号 (或者其它信息)
    """
    grids = get_grid_coordinates(image,grid_points)  # 获取每个格子的顶点
    
    # 遍历每个格子，判断输入点在哪个格子内
    for idx, grid in enumerate(grids):
        if is_point_in_rect(point, grid):
            return f"Point is in grid {idx+1}"
    
    return "Point is not in any grid"

# 示例调用
# 替换显示图像的部分
# 示例调用
lower_orange = np.array([5, 100, 100])  # HSV 颜色的下限
upper_orange = np.array([15, 255, 255])  # HSV 颜色的上限

# 读取图片
image = cv2.imread('./frames_output_text/1.000.jpg')  # 替换为你的图像路径
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

# 检测和合并线段并获取23个点的坐标
result_image, points_coordinates = detect_and_merge_lines(image, mask)

# 输出点的坐标
print("23 个点的坐标：")
for i, point in enumerate(points_coordinates):
    print(f"Point {i+1}: {point}")

# 输入坐标
x = int(input("请输入坐标 x: "))
y = int(input("请输入坐标 y: "))
input_point = (x, y)

# 在图像上显示输入的坐标点
cv2.circle(result_image, input_point, 8, (0, 0, 255), -1)  # 在图像上标记输入点（红色圆点）
cv2.putText(result_image, f"Input ({x}, {y})", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# 查找坐标所在的格子
grid_result = find_grid_for_point(input_point, points_coordinates,result_image)
print(grid_result)

# 在图像上显示格子编号
if "grid" in grid_result:
    cv2.putText(result_image, grid_result, (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# 显示图像
cv2.imshow('Result', result_image)
cv2.imwrite("grid_test.jpg",result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# result_image = detect_and_merge_lines(image, mask)

# cv2.imshow('Merged Lines with Corners and Center1111', result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 定义输入和输出文件夹路径
# input_folder = './frames_output_text'  # 替换为你的输入文件夹路径
# output_folder = './output_grid'  # 替换为你希望保存图像的文件夹路径

# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # 遍历输入文件夹中的所有文件
# for filename in os.listdir(input_folder):
#     if filename.endswith(".jpg") or filename.endswith(".png"):  # 处理jpg和png格式的图片
#         image_path = os.path.join(input_folder, filename)
#         image = cv2.imread(image_path)

#         if image is None:
#             print(f"无法读取图像: {filename}")
#             continue

#         # 转换为HSV图像并创建橙色区域的mask
#         hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

#         # 处理图像并绘制分割点和直线
#         result_image = detect_and_merge_lines(image, mask)

#         # 保存处理后的图像到输出文件夹
#         output_path = os.path.join(output_folder, filename)
#         cv2.imwrite(output_path, result_image)
#         # print(f"保存处理后的图像: {output_path}")

# print("批量处理完成！")