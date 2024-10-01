import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import xml.etree.ElementTree as ET
from PIL import Image, ImageTk

# 全局變量，用於存儲基準圖數據
settings = {
    'rotation_angle': None,
    'origin': None
}

# 保存基準圖設置至 XML 文件
def save_settings_to_xml(settings, file_name='settings.xml'):
    root = ET.Element("Settings")
    angle = ET.SubElement(root, "RotationAngle")
    angle.text = str(settings['rotation_angle'])
    origin = ET.SubElement(root, "Origin")
    origin.text = f"{settings['origin'][0]},{settings['origin'][1]}"

    tree = ET.ElementTree(root)
    tree.write(file_name)

# 從 XML 文件加載設置
def load_settings_from_xml(file_name='settings.xml'):
    global settings
    tree = ET.parse(file_name)
    root = tree.getroot()
    settings['rotation_angle'] = float(root.find("RotationAngle").text)
    origin_text = root.find("Origin").text.split(',')
    settings['origin'] = (int(origin_text[0]), int(origin_text[1]))

# 拍攝基準圖並進行設定處理
def capture_reference_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture image")
        return

    # 保存基準圖
    reference_image_path = 'reference_image.jpg'
    cv2.imwrite(reference_image_path, frame)

    # 對基準圖進行處理，找到旋轉角度和原點位置
    rotated_image, rotation_angle, origin = process_reference_image(frame)

    # 更新設置並保存到 XML
    settings['rotation_angle'] = rotation_angle
    settings['origin'] = origin
    save_settings_to_xml(settings)

    print(f"Settings saved: Rotation angle = {rotation_angle}, Origin = {origin}")

    # 顯示旋轉後的圖像
    show_image(rotated_image)

    cap.release()

# 基準圖的處理：計算旋轉角度，定義原點
def process_reference_image(image):
    # 假設定義的區域來檢測黑線
    region_top = 100
    region_bottom = 150
    region_left = 50
    region_right = 400

    region = image[region_top:region_bottom, region_left:region_right]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=50)

    if lines is not None:
        x1, y1, x2, y2 = lines[0][0]
        delta_y = y2 - y1
        delta_x = x2 - x1
        angle = np.degrees(np.arctan2(delta_y, delta_x))
        print(f"Calculated rotation angle: {angle} degrees")
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h))

        # 假設原點是圖片的某個區域
        origin_x = 100  # 這裡的數值可以根據需要來調整
        origin_y = 100
        return rotated_image, angle, (origin_x, origin_y)
    return image, 0, (0, 0)

# 顯示圖片
def show_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(image)
    im_tk = ImageTk.PhotoImage(im_pil)
    image_label.config(image=im_tk)
    image_label.image = im_tk

# 開始過程：只允許在基準設置完成後運行
def start_process():
    try:
        load_settings_from_xml()
        print(f"Loaded settings: Rotation angle = {settings['rotation_angle']}, Origin = {settings['origin']}")
        # 在此處添加拍攝圖片並使用基準圖設置進行處理的邏輯
    except FileNotFoundError:
        print("Error: Settings file not found. Please define the reference image first.")
        return

# GUI 設置
root = tk.Tk()
root.title("Image Processing with Reference Setup")

# Image display
image_label = tk.Label(root)
image_label.pack()

# 設置按鈕
btn_capture_reference = tk.Button(root, text="Capture Reference Image", command=capture_reference_image)
btn_capture_reference.pack()

# 開始過程按鈕
btn_start_process = tk.Button(root, text="Start Process", command=start_process)
btn_start_process.pack()

root.mainloop()
