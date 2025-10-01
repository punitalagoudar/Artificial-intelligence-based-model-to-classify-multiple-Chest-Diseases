import cv2
import numpy as np

img = cv2.imread('G:/0.A.New_BE 2025/Chest/chest_code/test/PNEUMONIA/person3_virus_16.jpeg', cv2.IMREAD_GRAYSCALE)

count1=0
for val in range(124,130):
    pix1 = np.sum(img == val)
    count1=count1+pix1

count1=count1/1000
print(count1)

