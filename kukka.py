import numpy as np
import cv2
import itertools
from scipy.special import comb
from math import *
from scipy import ndimage
import sys
from random import *

from helpers import *

leaf = lambda p1, p2, r, s, bvs, bn, sm: list(reversed(
    smooth_variation(rasablv(p1, p2, r, s, bn, neg(bvs)), sm)
))+ smooth_variation(rasablv(p1, p2, r, s, bn, bvs), sm)

"""
* Leaf frequency
* Better smooth variation
* Leaf level
"""
def _create_plant(
    p1, p2,
    stem_base_mod,
    stem_base_precision,
    stem_stroke_mod,
    leaf_base_mod,
    leaf_base_precision,
    leaf_stroke_mod,
    leaf_rotation,
    leaf_shear):

    length = distance(p1, p2)/3
    _stem = blv(p1, p2, stem_base_mod(length)(), n=stem_base_precision())

    stem = smooth_variation(_stem, stem_stroke_mod)

    paths = []
    paths.append(stem)

    for i, (_p1, _p2) in enumerate(zip(_stem, _stem[1:])):
        lr = -1 if randint(0, 1) == 1 else 1
        if i==len(_stem)-2: continue

        for j in range(randint(1, 2)):
            lr = -lr
            #l = (length/8)/((i+1)/10)
            l = length
            _leaf = leaf(
                _p2, circle_point(*_p2, l, line_angle(_p1, _p2)+lr*pi/2),
                r=leaf_rotation(-lr),
                s=leaf_shear(),
                bvs=leaf_base_mod(),
                bn=leaf_base_precision(),
                sm=leaf_stroke_mod,
            )

            paths.append([[5, 5], [0, 0]])
            paths.append([[0, 5], [5, 0]])
            paths.append(_leaf)

    return paths

def create_plant1(p1, p2):
    return _create_plant(p1, p2,
        stem_base_mod = lambda length: lambda: [np.random.uniform(-5, 5) for i in range(2)],
        stem_base_precision = lambda: 5,
        stem_stroke_mod = lambda: [np.random.uniform(-20, 20) for i in range(5)],

        leaf_base_mod = lambda: [np.random.uniform(-50, 50) for i in range(5)],
        leaf_base_precision = lambda: 3,
        leaf_stroke_mod = lambda: [np.random.uniform(-10, 10) for i in range(3)],

        leaf_shear = lambda: (random()-0.5)*2,
        leaf_rotation = lambda lr: np.random.uniform(lr*pi/3, lr*pi/10),
    )


import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 8))
columns = 1
rows = 1

def draw():
    plt.clf()
    for i in range(1, columns*rows+1):
        image = create_image()

        paths = []
        paths.append([[5, 5], [0, 0]])
        paths.append([[0, 5], [5, 0]])

        paths += create_plant1((50, 109), (280, 109))
        #plant += create_plant1((50, 150), (280, 109))
        #plant = foobar()

        image = plot_paths(paths, image, thickness=5, color=(120, 224, 12))
        #show_image(image)

        #f = open('kukka18.json', 'w')
        #f.write(str(paths))

        image = cv2.transpose(image)

        fig.add_subplot(rows, columns, i)
        plt.imshow(image)
        plt.axis('off')
    fig.canvas.draw()
    plt.show()

def on_press(event):
    if event.key == 'r': draw()

fig.canvas.mpl_connect('key_press_event', on_press)
draw()
