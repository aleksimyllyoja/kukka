from helpers import *

image = create_image()

paths = []
paths.append([[5, 5], [0, 0]])
paths.append([[0, 5], [5, 0]])

paths += create_plant1((50, 109), (280, 109))
#plant += create_plant1((50, 150), (280, 109))
#plant = foobar()

image = plot_paths(paths, image, thickness=5, color=(120, 224, 12))
#show_image(image)
