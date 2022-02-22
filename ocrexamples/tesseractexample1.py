import cv2
import pytesseract


img = cv2.imread('sample1.jpg')

print(pytesseract.image_to_string(img))