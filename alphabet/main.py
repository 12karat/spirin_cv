import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

save_path = Path(__file__).parent


def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2))
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled)


def extractor(region):
    cy, cx = region.centroid_local
    cy /= region.image.shape[0]
    cx /= region.image.shape[1]
    perimeter = region.perimeter / region.image.size
    holes = count_holes(region) / 100.0
    vlines = (np.sum(region.image, 0) == region.image.shape[1]).sum()
    hlines = (np.sum(region.image, 1) == region.image.shape[0]).sum()
    eccentricity = region.eccentricity
    aspect = region.image.shape[0] / region.image.shape[1]
    solidity = region.solidity
    extent = region.extent
    h, w = region.image.shape[0], region.image.shape[1]
    left = region.image[:, :w // 2].mean() if w > 1 else 0.5
    right = region.image[:, w // 2:].mean() if w > 1 else 0.5
    left_right = left / (right + 1e-9)
    top = region.image[:h // 2, :].sum() if h > 1 else 1.0
    bot = region.image[h // 2:, :].sum() if h > 1 else 1.0
    top_bot = top / (bot + 1e-9)
    return np.array([cy, cx, holes, eccentricity, aspect, solidity, extent, left_right, top_bot])


def classificator(region, templates):
    features = extractor(region)
    result = ""
    min_d = 10 ** 16
    for symbol, t in templates.items():
        d = ((t - features) ** 2).sum() ** 0.5
        if d < min_d:
            result = symbol
            min_d = d
    return result


template = imread("alphabet_ext.png")
print(template.shape)
template = template.sum(2)
binary = template != 765.
labeled = label(binary)
props = regionprops(labeled)

props = [p for p in props if p.area > 100]
props = sorted(props, key=lambda r: r.centroid[1])

templates = {}
for region, symbol in zip(props, ["A", "B", "8", "0", "1", "W", "X", "*", "-", "/", "P", "D"]):
    templates[symbol] = extractor(region)

print(type(props[0]))
print(props[0].area, props[0].centroid, props[0].label)
print(classificator(props[0], templates))

image = imread("symbols.png")
abinary = image.mean(2) > 0
alabeled = label(abinary)
print("Всего:", np.max(alabeled))
aprops = regionprops(alabeled)

result = {}
image_path = save_path / "out"
image_path.mkdir(exist_ok=True)

plt.figure(figsize=(5, 7))
for region in aprops:
    symbol = classificator(region, templates)
    if symbol not in result:
        result[symbol] = 0
    result[symbol] += 1
    plt.cla()
    plt.title(f"Class - '{symbol}'")
    plt.imshow(region.image)
    plt.savefig(image_path / f"image_{region.label}.png")

print(result)
for sym in ["A", "B", "8", "0", "1", "W", "X", "*", "-", "/", "P", "D"]:
    print(f"{sym}: {result.get(sym, 0)}")
print(f"Всего: {sum(result.values())}")
print(f"Процент распознавания: {(1 - result.get('?', 0) / len(aprops)) * 100}")
plt.imshow(abinary)
plt.show()