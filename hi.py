from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

image = Image.open("./MYPhoto.PNG")

plt.imshow(image)
plt.show()

image_fliped=image.transpose(Image.FLIP_LEFT_RIGHT)
plt.imshow(image_fliped)
plt.show()
image_rotate = image.transpose(Image.ROTATE_180)
#image_180=image.rotate(180)
plt.imshow(image_rotate)
plt.show()

image_downsize = image.resize((int(image.width/2),int(image.height/2)))
plt.imshow(image_downsize)
plt.show()

print(image.size)
print(image_downsize.size)