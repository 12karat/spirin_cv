import numpy as np
from skimage.measure import label

data = np.load("stars.npy")
regions = label(data)

pluses = 0
crosses = 0

for i in range(1, regions.max() + 1):
    star = (regions == i)
    
    if star.sum() == 9:
        y, x = np.where(star)
        
        if (y.max() - y.min() + 1) == 5:
            if data[y.min(), x.min()]:
                crosses += 1
            else:
                pluses += 1

print(f"Количество плюсов: {pluses}")
print(f"Количество крестов: {crosses}")
print(f"Количество звездочек: {pluses + crosses}")
