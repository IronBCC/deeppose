#!/usr/bin/env python

import scipy.misc
import numpy as np
import random

# Image size
width = 220
height = 220
channels = 3

number = 1000


# Create an empty image


# Draw something (http://stackoverflow.com/a/10032271/562769)
xx, yy = np.mgrid[:height, :width]
circle = (xx - 100) ** 2 + (yy - 100) ** 2

f = open('points.csv', 'w')
for i in range(number):
    x, y = random.randint(0, width-1), random.randint(0, height-1)

    img = np.zeros((height, width, channels), dtype=np.uint8)
    img[y][x][0] = 255
    img[y][x][1] = 255
    img[y][x][2] = 255

    img_name = "point"+str(i)+".png"
    scipy.misc.imsave("imgs/"+img_name, img)
    f.write(img_name + "," + str(x) + "," + str(y) + "\n")

    img_rxy = np.zeros((height, width, channels), dtype=np.uint8)
    for cy in range(img.shape[0]):
        for cx in range(img.shape[1]):
            img_rxy[cy][cx][1] = x
            img_rxy[cy][cx][2] = y
    img_rxy[y][x][0] = 255

    img_name = "point_RXY_" + str(i) + ".png"
    scipy.misc.imsave("imgs/" + img_name, img_rxy)

f.close()
# Set the RGB values
# for y in range(img.shape[0]):
#     for x in range(img.shape[1]):
#         r, g, b = circle[y][x], circle[y][x], circle[y][x]
#         img[y][x][0] = r
#         img[y][x][1] = g
#         img[y][x][2] = b

# Display the image
#scipy.misc.imshow(img)

# Save the image

