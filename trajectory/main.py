import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def extract_centers(frame_id):
    data = np.load(f"out/h_{frame_id}.npy")

    labeled, count = ndimage.label(data)

    centers = ndimage.center_of_mass(data, labeled, range(1, count + 1))
    return np.array(centers)

first = extract_centers(0)

order = np.argsort(first[:, 0])
tracks = [[pt] for pt in first[order]]

frames_total = 100

for frame_id in range(1, frames_total):
    new_points = extract_centers(frame_id)

    if len(new_points) == 0:
        continue

    for track in tracks:
        last_point = track[-1]

        diff = new_points - last_point
        dist = np.sum(diff**2, axis=1)

        nearest = np.argmin(dist)

        track.append(new_points[nearest])

plt.figure(figsize=(9, 7))

for idx, track in enumerate(tracks):
    track = np.array(track)

    x = track[:, 1]
    y = track[:, 0]

    plt.plot(x, y, linewidth=2)

    plt.scatter(x[0], y[0])
    plt.scatter(x[-1], y[-1], marker='x')

plt.gca().invert_yaxis()
plt.title("Движение объектов")
plt.grid(True)

plt.show()
