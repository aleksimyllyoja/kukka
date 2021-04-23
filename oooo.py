import numpy as np
import cv2
import itertools
from math import *
from scipy import ndimage
import sys
from random import *

from helpers import *

width = 300
height = 218

def zigzag(rows=5, w=100, h=20):
    cps = []
    x0 = 80
    y0 = 50
    for r in range(rows):
        l = []
        if r%2==0:
            l.append([x0+r*h, y0])
            l.append([x0+r*h, y0+w])
        else:
            l.append([x0+r*h, y0+w])
            l.append([x0+r*h, y0])

        cps += smooth_variation(l, lambda: [(random()-0.5)*50 for i in range(randint(2, 5))], n=120)
    return cps #smooth_variation(cps, lambda: [(random()-0.5) for i in range(2)], n=100)

def oooo(xs = 40, cs = 50):
    cps = zigzag(rows=8, w=100, h=20)

    angles = np.linspace(0, 2*pi*150, len(cps))
    radiuses = [10+random()*7 for i in range(len(cps)-1)]
    radiuses.append(radiuses[0])

    circle = lambda radiuses, angles, cps: [
        circle_point(x, y, r, a) for (x, y), (r, a) in zip(cps, (zip(radiuses, angles)))
    ]

    ring = circle(radiuses, angles, cps)

    #paths.append(ring)
    #return ring
    return smooth_variation(ring, lambda: [-random()*4 for i in range(randint(1, 5))])

import matplotlib.pyplot as plt
paths = []
#paths += ring1
img = create_image()
img = plot_paths([paths], img)

#show_image(img)

fig = plt.figure(figsize=(8, 8))
columns = 1
rows = 1
def draw():
    plt.clf()
    for i in range(1, columns*rows+1):
        image = create_image()

        paths = []
        #paths.append([[5, 5], [0, 0]])
        #paths.append([[0, 5], [5, 0]])

        paths += [oooo(xs=120)]
        #plant += create_plant1((50, 150), (280, 109))
        #plant = foobar()

        image = plot_paths(paths, image)
        #show_image(image)

        f = open('o5.json', 'w')
        f.write(str(paths))

        image = cv2.transpose(image)

        fig.add_subplot(rows, columns, i)
        plt.imshow(image)
        plt.axis('off')
    fig.canvas.draw()
    plt.show()

draw()
