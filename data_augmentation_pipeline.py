from os.path import join

from skimage import io
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import unsharp_mask
from os import listdir

def contrast_adjustment(img_path):
    image = io.imread(img_path)
    path = img_path.rsplit('/', 1)[0] + '/'
    image_name = img_path.split('/')[-1].split('.')[0]
    image_extension = img_path.split('/')[-1].split('.')[1]

    # Gamma
    gamma_corrected = exposure.adjust_gamma(image, 2)


def histogram_equalization(image_path):
    """
    Applies histogram equalization to the image
    See: https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html
    :param image_path: Image path
    :return: 
    """
    image = io.imread(image_path)

    # Equalization
    image_equalized = exposure.equalize_hist(image)

    # Adaptive Equalization
    image_adaptive_eq = exposure.equalize_adapthist(image, clip_limit=0.03)
    path = image_path.rsplit('/', 1)[0]
    image_name = image_path.rsplit('/', 1)[-1]
    just_name = image_name.split('.')[0]
    extension = image_name.split('.')[1]
    
    hist_name = just_name + "_hist." + extension
    adap_name = just_name + "_adap." + extension

    io.imsave(join(path, hist_name), image_equalized)
    io.imsave(join(path, adap_name), image_adaptive_eq)


def contrast_adjustment(image_path):
    image = io.imread(image_path)
    sharp_image = unsharp_mask(image, radius=20, amount=1)

    path = image_path.rsplit('/', 1)[0]
    image_name = image_path.rsplit('/', 1)[-1]
    just_name = image_name.split('.')[0]
    extension = image_name.split('.')[1]

    sharp_image_path = just_name + "_sharp." + extension
    io.imsave(join(path, sharp_image_path), sharp_image)


def pipeline(path):
    for image in listdir(path):
        contrast_adjustment(join(path, image))
        histogram_equalization(join(path, image))
        print(f'Image {image} completed')

pipeline("/home/alvaro/Documents/MaskForMuscle/Cropped/images")