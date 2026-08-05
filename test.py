import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np

images = []

for filename in glob.glob("plakoro/**/*.webp"):
    img = cv2.imread(filename)

    if img is None:
        print(f"Couldn't load {filename}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    images.append(gray)

images = np.array(images)


background = np.median(images, axis=0).astype(np.uint8)


plt.imshow(background, cmap="gray")
plt.axis("off")
plt.show()
