import cv2
import numpy as np

# 初始化全局变量
drawing = False  # 是否开始绘制矩形
ix, iy = -1, -1  # 矩形的初始坐标
rect = None  # 最终绘制的矩形坐标

# 回调函数，用于响应鼠标事件
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect

    if event == cv2.EVENT_LBUTTONDOWN:  # 当按下鼠标左键时
        drawing = True
        ix, iy = x, y  # 记录起始点
    elif event == cv2.EVENT_MOUSEMOVE:  # 当鼠标移动时
        if drawing == True:
            rect = (ix, iy, x, y)  # 实时更新矩形坐标
    elif event == cv2.EVENT_LBUTTONUP:  # 当松开鼠标左键时
        drawing = False
        rect = (ix, iy, x, y)  # 最终确定矩形坐标

# 读取图像
image = cv2.imread('b.webp')
clone = image.copy()

# 设置鼠标回调函数
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)

while True:
    # 创建一个副本，每次循环都在原始图像上绘制矩形
    img = image.copy()

    # 如果用户正在绘制矩形，则显示当前的矩形
    if rect is not None:
        x1, y1, x2, y2 = rect
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 显示图像
    cv2.imshow('image', img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 按 'q' 键退出
        break
    elif key == ord('a'):  # 按 'a' 键检测矩形区域内的亮点
        if rect is not None:
            x1, y1, x2, y2 = rect
            roi = clone[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]  # 裁剪矩形区域
            
            # 转换为灰度图像
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # 二值化处理，只在该区域内检测亮点
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            # 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            points_by_x = {}
            points_by_y = {}  # 新增的字典來存儲 Y 軸的資料
            # 在原图上绘制矩形区域内的轮廓
            for contour in contours:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cx += min(x1, x2)
                    cy += min(y1, y2)
                # X 軸處理
                added_x = False
                for key_x in points_by_x:
                    if abs(cx - key_x) <= 5:
                        points_by_x[key_x].append((cx, cy))
                        added_x = True
                        break
                if not added_x:
                    points_by_x[cx] = [(cx, cy)]

                # Y 軸處理
                added_y = False
                for key_y in points_by_y:
                    if abs(cy - key_y) <= 5:
                        points_by_y[key_y].append((cx, cy))
                        added_y = True
                        break
                if not added_y:
                    points_by_y[cy] = [(cx, cy)]    
                # 调整轮廓坐标回原图的坐标系
                contour[:, :, 0] += min(x1, x2)
                contour[:, :, 1] += min(y1, y2)
                print(contour)
            cv2.drawContours(image, contours, -1, (0, 0, 255), 2)



            # 按 X 軸輸出
            print("X軸的點：")
            i = 1
            for x in sorted(points_by_x):
                print(f"X={x}: {len(points_by_x[x])} points")
                i+=1

            # 按 Y 軸輸出
            j = 1
            print("\nY軸的點：")
            for y in sorted(points_by_y):
                print(f"Y={y}: {len(points_by_y[y])} points")
                j+=1

            cv2.imshow('Detected Dots', image)

cv2.destroyAllWindows()