import numpy as np
import cv2
from skimage.measure import label, regionprops
from pathlib import Path

REF_VEC_SIZE = 20
root_dir = Path(__file__).parent.absolute()

def preprocess(path):
    """Продвинутая бинаризация: понимает прозрачность и любые фоны"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"ОШИБКА: Не могу найти файл '{path.name}'. Точно положил его рядом со скриптом?")
        
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        if np.min(alpha) < 255:  
            return (alpha > 50).astype(int)
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    if np.mean(gray) > 127: # Светлый фон
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    else:                   # Темный фон
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
    return (thresh > 0).astype(int)

def get_vec(region):
    """Делает вектор 20x20 для сравнения"""
    sym = (region.image * 255).astype('uint8')
    res = cv2.resize(sym, (REF_VEC_SIZE, REF_VEC_SIZE), interpolation=cv2.INTER_AREA)
    return res.flatten().astype(float)

ref_labels = ["A", "B", "8", "0", "1", "W", "X", "star", "minus", "slash", "P", "D"]

try:
    raw_ref_bins = regionprops(label(preprocess(root_dir / "alphabet_ext.png")))
except FileNotFoundError as e:
    print(e)
    exit()

ref_bins = [r for r in raw_ref_bins if r.area > 5] 
ref_bins.sort(key=lambda x: x.bbox[1]) 

references = []
for i, reg in enumerate(ref_bins):
    if i < len(ref_labels):
        references.append({"label": ref_labels[i], "vector": get_vec(reg)})

if len(references) != len(ref_labels):
    print(f"ВНИМАНИЕ: Найдено {len(references)} эталонов, а ожидалось {len(ref_labels)}.")
    print("Проверь картинку alphabet_ext.png, возможно там грязь.")

try:
    target_bins = regionprops(label(preprocess(root_dir / "symbols.png")))
except FileNotFoundError as e:
    print(e)
    exit()

freq_dict = {name: 0 for name in ref_labels}
freq_dict["unknown"] = 0 

print(f"Найдено объектов на изображении: {len(target_bins)}. Идет классификация...")

count = 0
for reg in target_bins:
    if reg.area < 5: continue 
    
    t_vec = get_vec(reg)
    match_name = "unknown"
    min_dist = float('inf')
    
    for r in references:
        dist = np.linalg.norm(t_vec - r['vector'])
        if dist < min_dist:
            min_dist = dist
            match_name = r['label']
    
    freq_dict[match_name] += 1
    count += 1

print("\n" + "="*35)
print(" ЧАСТОТНЫЙ СЛОВАРЬ СИМВОЛОВ ")
print("="*35)

if freq_dict["unknown"] == 0:
    del freq_dict["unknown"] 

for char, freq in freq_dict.items():
    print(f"Символ {char:<7} : {freq:>3} шт.")

print("-" * 35)
print(f"ВСЕГО РАСПОЗНАНО: {count} шт.")
print("="*35)

dict_file = root_dir / "frequency_dict.txt"
with open(dict_file, "w", encoding="utf-8") as f:
    f.write("Частотный словарь символов:\n")
    for char, freq in freq_dict.items():
         f.write(f"{char}: {freq}\n")
    f.write(f"Всего: {count}\n")

print(f"\n[+] Словарь также сохранен в файл: {dict_file.name}")
