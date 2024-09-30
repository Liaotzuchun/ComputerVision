import cv2
import numpy as np

drawing = False  
ix, iy = -1, -1  
rect = None  

# 回调函数，用于响应鼠标事件
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect

    if event == cv2.EVENT_LBUTTONDOWN: 
        drawing = True
        ix, iy = x, y  
    elif event == cv2.EVENT_MOUSEMOVE:  
        if drawing == True:
            rect = (min(ix, x), min(iy, y), max(ix, y), max(iy, y))  # 保证矩形始终从左上到右下
    elif event == cv2.EVENT_LBUTTONUP: 
        drawing = False
        rect = (min(ix, x), min(iy, y), max(ix, y), max(iy, y))  

image = cv2.imread('002.BMP')
clone = image.copy()

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)

while True:
    img = image.copy()

    if rect is not None:
        x1, y1, x2, y2 = rect
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('image', img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 按 'q' 键退出
        break
    elif key == ord('a'):  # 按 'a' 键检测矩形区域内的亮点
        if rect is not None:
            x1, y1, x2, y2 = rect
            roi = clone[y1:y2, x1:x2]  # 裁剪矩形区域
            
            # 将裁剪的区域转为灰度图
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 进行二值化处理以检测亮点
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # 检测轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            points_by_x = {}
            points_by_y = {}  
            
            for contour in contours:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cx += x1  # 将坐标映射回原图的坐标系
                    cy += y1

                    # 标记亮点
                    cv2.circle(img, (cx, cy), 3, (0, 0, 255), 3)
                
                added_x = False
                for key_x in points_by_x:
                    if abs(cx - key_x) <= 1:
                        points_by_x[key_x].append((cx, cy))
                        added_x = True
                        break
                if not added_x:
                    points_by_x[cx] = [(cx, cy)]

                added_y = False
                for key_y in points_by_y:
                    if abs(cy - key_y) <= 1:
                        points_by_y[key_y].append((cx, cy))
                        added_y = True
                        break
                if not added_y:
                    points_by_y[cy] = [(cx, cy)]    

            image = cv2.drawContours(image, contours, -1, (0, 0, 200), 3)

            # 打印亮点信息
            print("X轴的点：")
            for i, x in enumerate(sorted(points_by_x)):
                print(f"X={i+1}: {len(points_by_x[x])} points")

            print("\nY轴的点：")
            for j, y in enumerate(sorted(points_by_y)):
                print(f"Y={j+1}: {len(points_by_y[y])} points")

            cv2.imshow('Detected Dots', image)
            
cv2.destroyAllWindows()
