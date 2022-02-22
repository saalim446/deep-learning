import os

from sklearn import datasets
from sklearn.svm import SVC
from skimage.io import imread
from skimage.exposure import rescale_intensity
from skimage.transform import resize

IMAGE_DIR = 'images'
TEST_IMAGE = '5.jpg'


digits = datasets.load_digits()
features, labels = digits.data, digits.target

print(features.shape)
print(features)
print(labels)

clf = SVC(gamma = 0.001)
clf.fit(features, labels)

orgimage = imread(os.path.join(IMAGE_DIR, TEST_IMAGE))
img = resize(orgimage, (8,8))
img = rescale_intensity(img, out_range=(0, 16))


x_test = [sum(pixel)/3.0 for row in img for pixel in row]
print("The predicted digit is {}".format(clf.predict([x_test])))

import matplotlib.pyplot as plt

plt.subplot(121); plt.title("orginal image"); plt.imshow(orgimage);
plt.subplot(122); plt.title("rescaled image"); plt.imshow(img);
plt.show()
