import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import binary_opening

image = np.load("wires3.npy")
struct = np.ones((3,1))
process = binary_opening(image, struct)

labeled_image = label(image)
labeled_process = label(process)
print(f"Original {np.max(labeled_image)}")
print(f"Processed {np.max(labeled_process)}")

for i in range(1, np.max(labeled_image) + 1):
    wire_parts = label(process * (labeled_image == i))
    print(f"Wire {i}: {np.max(wire_parts)} pieces")


struct = np.ones((3,3))
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(binary_opening(image, struct))
plt.show()
