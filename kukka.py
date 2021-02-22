import numpy as np
import cv2
import itertools
from scipy.special import comb
from math import *
from scipy import ndimage
import sys
from random import *

_take2 = lambda ps: list(map(lambda p: [p[0], p[1]], ps))

def bezier(ps, n=100):
    def _bpns(ps, n=100):
        return np.array(
            [_bernstein(i, len(ps)-1, np.linspace(0.0, 1.0, n))
             for i in range(0, len(ps))
            ]
        )

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

def show_image(image, rotation=-90):
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

def line_variations(p1, p2, vs):
    a = line_angle(p1, p2)
    splits = np.linspace(0, 1, len(vs)+2)
    bps = []
    for i, s in enumerate(splits[1:-1]):
        p = split_on(p1, p2, s)
        pn = circle_point(*p, vs[i], a+pi/2)
#        mark(*pn, image, size=1)
        bps.append(pn)

#    mark(*p1, image, size=2)
#    mark(*p2, image, size=2)

    bps.insert(0, p1)
    bps.append(p2)

    return bps

def shaky_line(p1, p2, variations, bp=10):
    bps = line_variations(p1, p2, variations)
    return bezier(bps, n=bp)

def _c0():
    precision = 4
    radiuses = [10+x*10 for x in range(int(precision/2))]
    radiuses += reversed(radiuses)

    angles = np.linspace(0, 2*pi, precision)

    c1 = _circle(150, 109, radiuses, angles)
    cps = list(zip(c1, c1[1:]))

    #variations = [randint(-50, 50) for i in range(10)]

    return [
        shaky_line(p1, p2, [randint(-50, 50) for i in range(10)], bp=10)
        for (p1, p2) in cps
    ]

flatten = lambda t: [item for sublist in t for item in sublist]

_R = lambda a: np.array(
    [[cos(a), -sin(a), 0],
     [sin(a), cos(a),  0],
     [0,      0,       1]]
)

_T = lambda x, y: np.array(
    [[1, 0, x],
     [0, 1, y],
     [0, 0, 1]]
)

_HS = lambda s: np.array(
    [[1, s, 0],
     [0, 1, 0],
     [0, 0, 1]]
)

_VS = lambda s: np.array(
    [[1, 0, 0],
     [s, 1, 0],
     [0, 0, 1]]
)

rotate_around = lambda ps, p0, a: _take2(list(map(
    lambda p: _T(*p0)@_R(a)@_T(-p0[0], -p0[1])@(p+[1]),
    ps
)))

shear_around = lambda ps, p0, s: _take2(list(map(
    lambda p: _T(*p0)@_VS(s)@_T(-p0[0], -p0[1])@(p+[1]),
    ps
)))

def leaf(p1, p2, rotation, shear_amount):
    global image

    #mark(*p1, image, size=5)

    d = distance(p1, p2)

    base_variations1 = [randint(0, 30-i*6) for i in range(5)]
    base_variations2 = list(map(lambda x: -x, base_variations1))

    sl1 = bezier(line_variations(p1, p2, base_variations1), n=6)
    sl2 = bezier(line_variations(p1, p2, base_variations2), n=6)

    sl1 = shear_around(sl1, p1, shear_amount)
    sl2 = shear_around(sl2, p1, shear_amount)

    sl1 = rotate_around(sl1, p1, rotation)
    sl2 = rotate_around(sl2, p1, rotation)

    #image = plot_paths([list(reversed(sl1))+sl2], image, color=(0, 0, 255))

    vi = int(d/7)

    vs2 = [randint(-vi, vi) for i in range(10)]
    vs3 = [randint(-vi, vi) for i in range(10)]

    cps1 = list(zip(sl1, sl1[1:]))
    cps2 = list(zip(sl2, sl2[1:]))

    l1 = [shaky_line(p1, p2, vs2, 100) for (p1, p2) in reversed(cps1)]
    l2 = [shaky_line(p1, p2, vs3, 100) for (p1, p2) in reversed(cps2)]

    return list(reversed(flatten(l2)))+flatten(l1)

def stem(p1, p2):

    base_variations = [randint(-20, 20) for i in range(5)]
    base = bezier(line_variations(p1, p2, base_variations), n=6)

    return base

#paths = _c0()
#paths = leaf((50, 109), (100, 109))

def create_plant():
    image = create_image()

    paths = []
    _stem = stem((50, 109), (280, 109))

    paths.append(_stem)
    #mark(*stem[1], image, size=5)

    for i in range(1, 6):
        paths.append(
            leaf(
                _stem[i], (_stem[i][0], _stem[i][1]-(80-i*10)),
                rotation=pi/randint(4, 6),
                shear_amount=random()
            )
        )

        paths.append(
            leaf(
                _stem[i], (_stem[i][0], _stem[i][1]+(80-i*10)),
                rotation=-pi/randint(4, 6),
                shear_amount=random()
            )
        )

    image = plot_paths(paths, image)

    return image

import matplotlib.pyplot as plt

w=2
h=2
fig=plt.figure(figsize=(8, 8))
columns = 2
rows = 2
for i in range(1, columns*rows +1):

    img = create_plant()
    img = cv2.transpose(img)

    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.axis('off')

plt.show()

#show_image(image)
