import time

import cv2
import pytesseract


img = cv2.imread('sample1.jpg')
start = time.time()
print(pytesseract.image_to_string(img))
end = time.time()
print("Detection Took", end-start, " (s)")