import cv2
import numpy as np

def get_shape_type(contour):
    arc_len = cv2.arcLength(contour, True)
    if arc_len == 0:
        return None
        
    approx = cv2.approxPolyDP(contour, 0.04 * arc_len, True)
    vertices = len(approx)
    
    if vertices == 4:
        return "Прямоугольники"
    elif vertices > 4:
        return "Круги"
    return None

def main():
    img = cv2.imread("C:/Users/vital\Desktop/figures_and_colors/balls_and_rects.png")
    if img is None:
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    color_ranges = [
        ("Красный", [0, 50, 50], [10, 255, 255]),
        ("Оранжевый", [11, 50, 50], [25, 255, 255]),
        ("Жёлтый", [26, 50, 50], [35, 255, 255]),
        ("Зелёный", [36, 50, 50], [85, 255, 255]),
        ("Синий", [86, 50, 50], [130, 255, 255]),
        ("Фиолетовый", [131, 50, 50], [160, 255, 255]),
        ("Розовый", [161, 50, 50], [180, 255, 255])
    ]

    summary = {}
    total_count = 0

    for name, low, high in color_ranges:
        mask = cv2.inRange(hsv, np.array(low), np.array(high))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = 0
        rects = 0
        
        for c in contours:
            if cv2.contourArea(c) < 15:
                continue
                
            label = get_shape_type(c)
            if label == "Круги":
                circles += 1
                total_count += 1
            elif label == "Прямоугольники":
                rects += 1
                total_count += 1
        
        if circles > 0 or rects > 0:
            summary[name] = (circles, rects)

    print(f"Общее количество фигур: {total_count}")
    for shade, counts in summary.items():
        print(f"Оттенок: {shade}")
        print(f"  Кругов: {counts[0]}")
        print(f"  Прямоугольников: {counts[1]}")

if __name__ == "__main__":
    main()