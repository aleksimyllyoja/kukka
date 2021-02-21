import numpy as np
import cv2
import itertools
from scipy.special import comb
from math import *
from scipy import ndimage
import sys
from random import *

def bezier(ps, n=100):
    def _bpns(ps, n=100):
        return np.array(
            [_bernstein(i, len(ps)-1, np.linspace(0.0, 1.0, n))
             for i in range(0, len(ps))
            ]
        )

    def _take2(ps):
        return list(map(lambda p: [p[0], p[1]], ps))

    def _bernstein(i, n, t):
        return comb(n, i)*(t**(n-i))*(1-t)**i

    return _take2(
        zip(
            np.array([p[0] for p in ps])@_bpns(ps, n),
            np.array([p[1] for p in ps])@_bpns(ps, n)
        )
    )

def distance(p1, p2):
    return sqrt(
        (p2[0]-p1[0])**2 +
        (p2[1]-p1[1])**2
    )

def scale(ps, s):
    return [[p[0]*s, p[1]*s] for p in ps]

def line_angle(p1, p2):
    return atan2(p2[1]-p1[1], p2[0]-p1[0])

def split_on(p1, p2, s):
    return [
        (p2[0]-p1[0])*s + p1[0],
        (p2[1]-p1[1])*s + p1[1]
    ]

def create_image(width=300, height=218, s=2):
    img = np.zeros((height*s, width*s, 3), np.uint8)
    img[:,:] = (255, 255, 255)
    return img

def show_image(image, rotation=90):
    cv2.imshow(
        'image',
        ndimage.rotate(image, rotation)
    )
    if cv2.waitKey(): cv2.destroyAllWindows()

def plot_paths(paths, image, color=(0,0,0), s=2, thickness=1):
    paths = map(lambda ps: scale(ps, s), paths)
    for path in paths:
        for p1, p2 in zip(path, path[1:]):
            cv2.line(image,
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                color, thickness=thickness
            )

    return image

def mark(x, y, image, s=2, size=1):
    cv2.circle(
        image,
        (int(x)*s, int(y)*s),
        size,
        (0, 0, 255),
        thickness=-1
    )

def circle_point(x, y, r, a):
    return (
        x+cos(a)*r,
        y+sin(a)*r
    )

def _circle(x, y, radiuses, angles):
    ps = []
    for r, a in zip(radiuses, angles):
        ps.append(
            circle_point(x, y, r, a)
        )
    return ps

def shaky_line(p1, p2):
    a = line_angle(p1, p2)
    splits = np.linspace(0, 1, 5)
    bps = []
    for i, s in enumerate(splits[1:-1]):
        p = split_on(p1, p2, s)
        #mark(*p1, image, size=1)
        bps.append(
            circle_point(*p, randint(0, 10), a+(-1)**i*pi/2)
        )

    #mark(*p1, image, size=2)
    #mark(*p2, image, size=2)

    bps.insert(0, p1)
    bps.append(p2)

    return bezier(bps, n=10)

image = create_image()

precision = 20
radiuses = [randint(10, 90) for x in range(precision-1)]
radiuses.append(radiuses[0])
angles = np.linspace(0, 2*pi, precision)

c1 = _circle(150, 109, radiuses, angles)
cps = list(zip(c1, c1[1:]))

c2 = [shaky_line(p1, p2) for (p1, p2) in cps]

paths = c2
image = plot_paths(paths, image)

show_image(image)
