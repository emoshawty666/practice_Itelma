import os
import cv2
import numpy as np
from shapely.geometry import Polygon, LineString
from ultralytics import YOLO

# Загрузка модели
model = YOLO('D:\\PracticeItelma\\runs\\obb\\train21\\weights\\best.pt')

# Папки с изображениями
input_folder = 'images'
output_folder = 'results'
os.makedirs(output_folder, exist_ok=True)

# Функция отрисовки поворотного бокса
def draw_rotated_bbox(image, bbox, label_text=""):
    bbox = bbox.cpu().numpy()
    cx, cy, w, h, angle_rad = bbox
    angle_deg = angle_rad * 180 / np.pi

    rect = ((cx, cy), (w, h), angle_deg)
    box = cv2.boxPoints(rect).astype(int)
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

    if label_text:
        offset = 30
        dx = offset * np.cos(angle_rad)
        dy = offset * np.sin(angle_rad)
        text_x = int(cx + dx)
        text_y = int(cy + dy)

        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image,
                      (text_x - 2, text_y - text_h - 2),
                      (text_x + text_w + 2, text_y + 2),
                      (255, 255, 255), -1)
        cv2.putText(image, label_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 100), 2)

# Обработка изображений из папки
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        # Инференс
        results = model(img, imgsz=640, iou=0.4, conf=0.5, verbose=False)
        result = results[0]

        if result.obb is None or result.obb.xywhr is None:
            print(f"[{filename}] - нет детекций.")
            continue

        boxes = result.obb.xywhr
        zone_boxes = []
        line_boxes = []

        # Классификация боксов по форме
        for bbox in boxes:
            bbox_np = bbox.cpu().numpy()
            cx, cy, w, h, angle = bbox_np
            angle_deg = angle * 180 / np.pi
            rect = ((cx, cy), (w, h), angle_deg)
            box_pts = cv2.boxPoints(rect).astype(int)

            label = f"{angle_deg:.2f} deg"
            draw_rotated_bbox(img, bbox, label_text=label)

            if max(w, h) / min(w, h) > 2.0:
                line_boxes.append(box_pts)
            else:
                zone_boxes.append(box_pts)

        # Поиск пересечений нижней грани zone с line
        for z_box in zone_boxes:
            for l_box in line_boxes:
                zone_poly = Polygon(z_box)
                line_poly = Polygon(l_box)

                # Нижняя грань zone — 2 точки с макс. Y
                z_sorted = sorted(z_box, key=lambda p: p[1], reverse=True)
                bottom_edge = LineString([z_sorted[0], z_sorted[1]])

                if line_poly.intersects(bottom_edge):
                    inter = line_poly.intersection(bottom_edge)

                    # Поддержка разных типов геометрии
                    geoms = []
                    if inter.geom_type == 'Point':
                        geoms = [inter]
                    elif inter.geom_type in ['MultiPoint', 'GeometryCollection']:
                        geoms = list(inter.geoms)
                    elif inter.geom_type == 'LineString':
                        # Используем начальную точку линии
                        geoms = [inter]

                    for pt in geoms:
                        if pt.is_empty:
                            continue
                        x, y = map(int, pt.coords[0])
                        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                        cv2.putText(img, f"Y = {y}", (x + 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Сохранение результата
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)
        print(f"[{filename}] - обработано.")
