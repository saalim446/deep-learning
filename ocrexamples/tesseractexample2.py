import cv2
import numpy as np

img = cv2.imread('image.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)

    else:
        angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


gray = get_grayscale(img)
thresh = thresholding(gray)
opening = opening(gray)
canny = canny(gray)

import matplotlib.pyplot as plt

plt.subplot(221); plt.title("gray"); plt.imshow(gray, cmap='gray');
plt.subplot(222); plt.title("threshold"); plt.imshow(thresh);
plt.subplot(223); plt.title("opening"); plt.imshow(opening);
plt.subplot(224); plt.title("canny"); plt.imshow(canny);

plt.show()

import pytesseract


img = gray
print("gray ocr")
print(pytesseract.image_to_string(img))

img = thresh
print("threshold ocr")
print(pytesseract.image_to_string(img))

img = opening
print("opening ocr")
print(pytesseract.image_to_string(img))

img = canny
print("canny ocr")
print(pytesseract.image_to_string(img))
