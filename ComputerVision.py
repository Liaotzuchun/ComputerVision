import cv2
import numpy as np

# 自动检测黑线的左右端点，并计算旋转角度
def detect_black_line_and_correct_rotation(image):
    # 设定固定区域的坐标 (y1, y2, x1, x2)
    region_top = 100  # 区域顶部Y坐标
    region_bottom = 150  # 区域底部Y坐标
    region_left = 50  # 区域左侧X坐标
    region_right = 400  # 区域右侧X坐标

    # 裁剪出指定区域
    region = image[region_top:region_bottom, region_left:region_right]

    # 转换为灰度图
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # 提高对比度，便于线条检测
    gray = cv2.equalizeHist(gray)

    # 边缘检测，检测黑线边缘
    edges = cv2.Canny(gray, 50, 150)

    # 使用 Hough 变换检测线条
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=50)
    
    if lines is not None:
        leftmost_point = None
        rightmost_point = None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 找到区域内最左边和最右边的点
            if leftmost_point is None or x1 < leftmost_point[0]:
                leftmost_point = (x1 + region_left, y1 + region_top)
            if leftmost_point is None or x2 < leftmost_point[0]:
                leftmost_point = (x2 + region_left, y2 + region_top)
            if rightmost_point is None or x1 > rightmost_point[0]:
                rightmost_point = (x1 + region_left, y1 + region_top)
            if rightmost_point is None or x2 > rightmost_point[0]:
                rightmost_point = (x2 + region_left, y2 + region_top)
        
        # 确保检测到左右端点后，计算旋转角度
        if leftmost_point and rightmost_point:
            (x1, y1) = leftmost_point
            (x2, y2) = rightmost_point
            
            # 计算垂直差异，使用 atan2 计算角度
            delta_y = y2 - y1
            delta_x = x2 - x1
            angle = np.degrees(np.arctan2(delta_y, delta_x))
            print(f"Calculated angle for rotation: {-angle} degrees")

            # 旋转整个图像，使黑线水平
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            rotated_image = cv2.warpAffine(image, M, (w, h))
            return rotated_image
    
    # 如果没有找到线条，返回原图
    return image

# 读取图像
image = cv2.imread('002.BMP')

# 自动检测固定区域内的黑线并旋转图像
rotated_image = detect_black_line_and_correct_rotation(image)

# 在旋转后的图像上继续执行亮点检测
gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

# 二值化处理
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
kernel = np.ones((3, 3), np.uint8)

# 设置 iterations=3 进行形态学开操作，清除噪声和杂点
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=3)

# 检测轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 初始化亮点坐标列表
center_list = []

# 遍历每个轮廓，获取亮点的圆心坐标
for contour in contours:
    if 200 > cv2.contourArea(contour) > 5:  # 根据面积过滤不需要的轮廓
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        center_list.append(center)

# 输出亮点的圆心坐标列表
print("亮点的坐标列表：")
print(center_list)

# 获取亮点的坐标范围
x_values = [center[0] for center in center_list]
y_values = [center[1] for center in center_list]
x_min, x_max = min(x_values), max(x_values)
y_min, y_max = min(y_values), max(y_values)

# 设置误差容忍范围
error_margin = 5  # 设置误差为5个像素

# 计算网格大小，使得亮点区域划分为 5x12 的矩阵
grid_size_x = 12  # 列数
grid_size_y = 5   # 行数
grid_x_step = (x_max - x_min) // (grid_size_x - 1)  # 列的步长
grid_y_step = (y_max - y_min) // (grid_size_y - 1)  # 行的步长

# 初始化 5x12 的矩阵，默认值为 0
matrix = np.zeros((grid_size_y, grid_size_x), dtype=int)

# 遍历亮点坐标，将其转换为矩阵中的相对位置
for center in center_list:
    grid_x = (center[0] - x_min + error_margin) // grid_x_step
    grid_y = (center[1] - y_min + error_margin) // grid_y_step

    # 防止超出范围
    if 0 <= grid_x < grid_size_x and 0 <= grid_y < grid_size_y:
        matrix[grid_y, grid_x] = 1

# 打印 5x12 的矩阵
print("亮点矩阵 (5x12)：")
for row in matrix:
    print(row)

# 标记亮点在图像上的位置
for center in center_list:
    cv2.circle(rotated_image, center, 5, (0, 255, 0), 2)  # 绿色圆圈标记圆心

# 保存旋转并标记亮点的图像
cv2.imwrite("rotated_image_with_centers.png", rotated_image)

# 直接显示旋转并标记亮点的图像
cv2.imshow("Rotated Image with Centers", rotated_image)
cv2.waitKey(0)  # 显示5秒后自动关闭窗口
cv2.destroyAllWindows()
