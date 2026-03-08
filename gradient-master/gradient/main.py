import numpy as np
import matplotlib.pyplot as plt

def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1

size = 100
image = np.zeros((size, size, 3), dtype="uint8")

color1 = np.array([0, 128, 255])   
color2 = np.array([255, 128, 0])  
 
max_sum = (size - 1) + (size - 1)

for i in range(size):
    for j in range(size):
        t = (i + j) / max_sum
        
        r = lerp(color1[0], color2[0], t)
        g = lerp(color1[1], color2[1], t)
        b = lerp(color1[2], color2[2], t)
        
        image[i, j] = [int(r), int(g), int(b)]

plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.show()