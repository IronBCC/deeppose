#!/usr/bin/env python

import scipy.misc
import numpy as np
import random
import math

# Image size
width = 220
height = 220
channels = 3

number = 10000
spot_size = 10


def circle_bright(x, y, center_x, center_y):
    dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    if dist > spot_size:
        return 0

    return 255 - max((dist / spot_size * 255.0), 0)


f = open('spots.csv', 'w')
for i in range(number):
    center_x, center_y = random.randint(0, width-1), random.randint(0, height-1)

    img = np.zeros((height, width, channels), dtype=np.uint8)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img[y][x][0] = circle_bright(x,y, center_x, center_y)
            img[y][x][1] = circle_bright(x,y, center_x, center_y)
            img[y][x][2] = circle_bright(x,y, center_x, center_y)

    img_name = "spots"+str(i)+".png"
    scipy.misc.imsave("imgs/"+img_name, img)

    f.write(img_name + "," + str(center_x) + "," + str(center_y) + "\n")

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

