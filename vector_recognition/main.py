import numpy as np
import cv2
import os
import shutil
from skimage.measure import label, regionprops
from pathlib import Path

CANVAS_SIZE = 256
REF_VEC_SIZE = 20
root_dir = Path(__file__).parent.absolute()
out_dir = root_dir / "vector_recognition"

if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(exist_ok=True)

def preprocess(path):
    """Продвинутая бинаризация: понимает прозрачность и любые фоны"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Не могу прочитать {path}")
        
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        if np.min(alpha) < 255: 
            return (alpha > 50).astype(int)
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    if np.mean(gray) > 127: 
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    else:                 
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
    return (thresh > 0).astype(int)

def get_vec(region):
    """Делает вектор 20x20"""
    sym = (region.image * 255).astype('uint8')
    res = cv2.resize(sym, (REF_VEC_SIZE, REF_VEC_SIZE), interpolation=cv2.INTER_AREA)
    return res.flatten().astype(float)

ref_labels = ["A", "B", "8", "0", "1", "W", "X", "star", "minus", "slash"]

raw_ref_bins = regionprops(label(preprocess(root_dir / "alphabet-small.png")))
ref_bins = [r for r in raw_ref_bins if r.area > 10] 
ref_bins.sort(key=lambda x: x.bbox[1]) # Слева направо

references = []
for i, reg in enumerate(ref_bins):
    if i < len(ref_labels):
        references.append({"label": ref_labels[i], "vector": get_vec(reg)})

print(f"Загружено эталонов: {len(references)} из {len(ref_labels)}")
if not references:
    print("ОШИБКА: Эталоны не найдены! Проверь картинку alphabet-small.png")
    exit()

target_bins = regionprops(label(preprocess(root_dir / "alphabet.png")))
print(f"Найдено объектов: {len(target_bins)}. Идет классификация...")

count = 0
for reg in target_bins:
    if reg.area < 10: continue 
    
    t_vec = get_vec(reg)
    match_name = "unknown"
    min_dist = float('inf')
    
    for r in references:
        dist = np.linalg.norm(t_vec - r['vector'])
        if dist < min_dist:
            min_dist = dist
            match_name = r['label'] 

    sym_img = (reg.image * 255).astype('uint8')
    h, w = sym_img.shape
    scale = (CANVAS_SIZE * 0.6) / max(h, w)
    sym_res = cv2.resize(sym_img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype='uint8')
    y_off = (CANVAS_SIZE - sym_res.shape[0]) // 2
    x_off = (CANVAS_SIZE - sym_res.shape[1]) // 2
    canvas[y_off:y_off+sym_res.shape[0], x_off:x_off+sym_res.shape[1]] = sym_res
    
    cv2.putText(canvas, f"Class: {match_name}", (15, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
    
    char_dir = out_dir / match_name
    char_dir.mkdir(exist_ok=True)
    
    count += 1
    cv2.imwrite(str(char_dir / f"symbol_{count}.png"), canvas)

print(f"Успех! Сохранено нормальных картинок: {count}. Проверяй папку '{out_dir.name}'")