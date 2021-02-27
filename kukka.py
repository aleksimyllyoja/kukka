import numpy as np
import cv2
import itertools
from scipy.special import comb
from math import *
from scipy import ndimage
import sys
from random import *

_take2 = lambda ps: list(map(lambda p: [p[0], p[1]], ps))

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
    lambda p: _T(*p0)@_R(a)@_T(-p0[0], -p0[1])@(p+[1]), ps
)))

shear_around = lambda ps, p0, s: _take2(list(map(
    lambda p: _T(*p0)@_VS(s)@_T(-p0[0], -p0[1])@(p+[1]), ps
)))

bernstein = lambda i, n, t: comb(n, i)*(t**(n-i))*(1-t)**i

_bpns = lambda ps, n=100: np.array([
    bernstein(i, len(ps)-1, np.linspace(0.0, 1.0, n))
    for i in range(0, len(ps))
])

bezier = lambda ps, n=100: _take2(zip(
    np.array([p[0] for p in ps])@_bpns(ps, n),
    np.array([p[1] for p in ps])@_bpns(ps, n)
))

distance = lambda p1, p2: sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

scale = lambda ps, s: [[p[0]*s, p[1]*s] for p in ps]

line_angle = lambda p1, p2: atan2(p2[1]-p1[1], p2[0]-p1[0])

split_on = lambda p1, p2, s: [(p2[0]-p1[0])*s + p1[0], (p2[1]-p1[1])*s + p1[1]]

circle_point = lambda x, y, r, a: (x+cos(a)*r, y+sin(a)*r)

circle = lambda x, y, radiuses, angles: [
    circle_point(x, y, r, a) for r, a in zip(radiuses, angles)
]

_line_variations = lambda p1, p2, vs, a: [p1]+[
    circle_point(*split_on(p1, p2, s), vs[i], a+pi/2)
    for i, s in enumerate(np.linspace(0, 1, len(vs)+2)[1:-1])
]+[p2]

line_variations = lambda p1, p2, vs: _line_variations(
    p1, p2, vs, line_angle(p1, p2)
)

blv = lambda p1, p2, vs, n: bezier(line_variations(p1, p2, vs), n=n)

_rasablv = lambda p1, p2, r, s, n, vs: rotate_around(shear_around(
    blv(p1, p2, vs, n), p1, s), p1, r)

_neg = lambda xs: list(map(lambda x: -x, xs))

leaf = lambda p1, p2, r, s, bvs, bn, sm: list(reversed(
    smooth_variation(_rasablv(p1, p2, r, s, bn, _neg(bvs)), sm)
))+ smooth_variation(_rasablv(p1, p2, r, s, bn, bvs), sm)

def _create_plant(p1, p2,
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
            l = (length/8)/((i+1)/10)

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
        stem_base_mod = lambda length: lambda: [np.random.uniform(-10, 10) for i in range(2)],
        stem_base_precision = lambda: 6,
        stem_stroke_mod = lambda: [np.random.uniform(-20, 20) for i in range(5)],

        leaf_base_mod = lambda: [np.random.uniform(-50, 50) for i in range(5)],
        leaf_base_precision = lambda: 3,
        leaf_stroke_mod = lambda: [np.random.uniform(-10, 10) for i in range(3)],

        leaf_shear = lambda: (random()-0.5)*2,
        leaf_rotation = lambda lr: np.random.uniform(lr*pi/3, lr*pi/10),
    )

def smooth_variation(ps, mod):
    _ps = []
    for i, (p1, p2) in enumerate(zip(ps, ps[1:])):
        lvs = line_variations(p1, p2, mod())

        if i>0:
            lvs[1] = circle_point(
                *last_p,
                distance(last_p, last_v),
                atan2(last_p[1]-last_v[1], last_p[0]-last_v[0])
            )

        _ps += reversed(bezier(lvs, n=10))

        last_v = lvs[-2]
        last_p = lvs[-1]

    return _ps

##########################################

def create_image(width=300, height=218, s=2):
    img = np.zeros((height*s, width*s, 3), np.uint8)
    img[:,:] = (255, 255, 255)
    return img

def show_image(image, rotation=-90):
    cv2.imshow('image', ndimage.rotate(image, rotation))
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

image = create_image()

paths = []
paths.append([[5, 5], [0, 0]])
paths.append([[0, 5], [5, 0]])
paths += create_plant1((50, 109), (280, 109))
#plant += create_plant1((50, 150), (280, 109))
#plant = foobar()

image = plot_paths(paths, image)
show_image(image)

f = open('kukka18.json', 'w')
f.write(str(paths))

"""
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 8))
columns = 2
rows = 2
for i in range(1, columns*rows+1):

    image = create_image()

    plant = create_plant1((50, 109), (280, 109))
    #plant = foobar()

    image = plot_paths(plant, image)
    image = cv2.transpose(image)

    fig.add_subplot(rows, columns, i)
    plt.imshow(image)
    plt.axis('off')

plt.show()
"""
#show_image(image)
