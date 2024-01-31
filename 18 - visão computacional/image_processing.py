# Reading an Image
### Image Link: https://i.scdn.co/image/ab6761610000e5eb31f6ab67e6025de876475814

# Reading Image using Opencv
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('ab6761610000e5eb31f6ab67e6025de876475814')
## Opencv by defaults reads the image in BGR
plt.imshow(img)
rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(rgb_image)

# Converting to grayscale
gray_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2GRAY)
plt.imshow(gray_image,cmap='gray')

rgb_image.shape
gray_image.shape
# As you can see, the coloured image has 3 channels, whereas the gray image has only 1 channel, the shape of the image is (height,width,num of channels)


# Image resizing
img_gray_small = cv.resize(gray_image,(50,50))
plt.imshow(img_gray_small, cmap='gray')
# As you can see the image is pixelated now. The original image contained a lot more data about the image, it's dimension was 640x640. After resizing to 50x50 a lot of the information was lost and we are seeing a pixelated image when visualised at the same scale
img_gray_small.shape

# Cropping:
### Cropping the image is as simple as slicing the numpy array, we know the height, width of the image, we can crop it according to our choice

plt.imshow(gray_image[10:340,:], cmap='gray')
plt.imshow(gray_image[:,:300], cmap='gray')


