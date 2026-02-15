URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'



import os
import urllib
import urllib.request
from zipfile import ZipFile
import cv2
import matplotlib.pyplot as plt

# if not os.path.isfile(FILE):
#     print(f'Downloading {URL} and saving as {FILE}...')
#     urllib.request.urlretrieve(URL, FILE)

# print('Unzipping images...')
# with ZipFile(FILE) as zip_images:
#     zip_images.extractall(FOLDER)




labels = os.listdir('fashion_mnist_images/train')


X = []
y = []

for label in labels:
    for file in os.listdir(os.path.join(
                    'fashion_mnist_images', 'train', label
    )):

        image = cv2.imread(os.path.join(
                    'fashion_mnist_images', 'train', label, file
                ),cv2.IMREAD_UNCHANGED)


        X.append(image)
        y.append(label)
        




plt.imshow(image, cmap='grey')
plt.show()

print('Done!')