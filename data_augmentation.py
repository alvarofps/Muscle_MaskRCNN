from skimage import io
from skimage import data
from skimage.transform import rotate, resize
from skimage.filters import unsharp_mask
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from wand.image import Image
import numpy as np
import os
import glob
from skimage.filters import gaussian
from skimage import transform
from skimage import exposure
from bbox_util import *
import imgaug as ig
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import math
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, img_as_float
from skimage import exposure


images_names = [f for f in os.listdir('/home/alvaro/Documents/MaskForMuscle/') if os.path.splitext(f)[-1] == '.png']
images = []
name_and_file = {}
only_names = [f for f in os.listdir('/home/alvaro/Documents/MaskForMuscle/') if os.path.splitext(f)[-1] == '.png']

"""
HOW IMAGES ARE SAVED AND FORMATTED
Example:

blackshark_001.png ----> Original image
blackshark_001_g.png ----> Gaussian blur
blackshark_001_c.png ----> Gamma contrast adjustment 

blackshark_001_h.png ----> Horizontal flip ----> Needs Bbox transformation
blackshark_001_v.png ----> Vertical flip ----> Needs Bbox transformation
blackshark_001_30cc.png ----> Original image with 30 degree counter clockwise rotation ----> Needs Bbox transformation
blackshark_001_h_30c.png ----> Horizontal flipped image with 30 clockwise rotation ----> Needs Bbox transformation
blackshark_001_sh.png ----> Horizontal shear ----> Needs Bbox transformation
"""


def complete_path():
    global images_names
    for i in range(0, len(images_names)):
        images_names[i] = '/home/alvaro/Documents/MaskForMuscle/'+images_names[i]


def read_images():
    global images
    global name_and_file
    for img in images_names:
        images.append(io.imread(img))

    name_and_file = dict(zip(images_names, images))


"""
This method was used for the previous dataset,
in which only hammerhead_shark data was present
"""


def rotate_images():
    for name in name_and_file:
        new_image_1 = rotate(name_and_file[name], 30, resize=False)
        new_image_2 = rotate(name_and_file[name], 45, resize=False)
        new_image_3 = rotate(name_and_file[name], 210, resize=False)
        length = len(name.split('/'))
        print()
        io.imsave('30' + name.split('/')[length - 1], new_image_1)
        io.imsave('45' + name.split('/')[length - 1], new_image_2)
        io.imsave('210' + name.split('/')[length - 1], new_image_3)


def sharpen_image(path):
    image = io.imread(path)
    image_name = path.split('/')[-1]
    print(image_name)
    print('Image type', type(image))
    result_1 = unsharp_mask(image, radius=1, amount=1)
    result_2 = unsharp_mask(image, radius=5, amount=2)
    result_3 = unsharp_mask(image, radius=20, amount=1)

    fig, axes = plt.subplots(nrows=2, ncols=2,
                             sharex=True, sharey=True, figsize=(10, 10))
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original image')
    ax[1].imshow(result_1, cmap=plt.cm.gray)
    ax[1].set_title('Enhanced image, radius=1, amount=1.0')
    ax[2].imshow(result_2, cmap=plt.cm.gray)
    ax[2].set_title('Enhanced image, radius=5, amount=2.0')
    ax[3].imshow(result_3, cmap=plt.cm.gray)
    ax[3].set_title('Enhanced image, radius=20, amount=1.0')
    print(type(result_3))
    for a in ax:
        a.axis('off')
    fig.tight_layout()
    plt.show()
    # io.imsave(path, result_3)


# THis dumb lib wand does not allow to create np.ndarrays
# thus first crop the apply sharpening
def crop_image(path, save_path):
    img = Image(filename=path)
    # except 093
    # 1400, 150
    # We pass the top left corner coors, then the desired width and height
    img.crop(left=127, top=121, width=381, height=365)
    # img.resize(1138, 640)
    img.save(filename=save_path)


"""
This methods horizontally and vertically flips images
Also rotates 30 degrees counter and clockwise
The saves them with the respective identifier
"""


def flip_rotate_images(img_path):
    image = io.imread(img_path)
    """
    With rsplit I can specify how many splits I perform. 
    >>> s = "a,b,c,d"
    >>> s.rsplit(',', 1)
    ['a,b,c', 'd']
    >>> s.rsplit(',', 2)
    ['a,b', 'c', 'd']
    I now have the path but it is missing a '/' at the end
    """
    path = img_path.rsplit('/', 1)[0] + '/'

    image_name = img_path.split('/')[-1].split('.')[0]
    image_extension = img_path.split('/')[-1].split('.')[1]

    degree_of_rotation_counterclockwise = 30
    degree_of_rotation_clockwise = -30

    """Additional sharpening if needed"""
    # sharp = unsharp_mask(image, radius=20, amount=1)

    hor_flipped = image[:, ::-1]
    io.imsave(path + image_name + '_h.'+image_extension, hor_flipped)

    vert_flipped = image[::-1, :]
    io.imsave(path + image_name + '_v.' + image_extension, vert_flipped)

    rotated_counterclockwise = rotate(image, degree_of_rotation_counterclockwise, resize=True)
    io.imsave(path + image_name + '_' + str(degree_of_rotation_counterclockwise) + 'cc.' + image_extension,
              rotated_counterclockwise)

    rotated_clockwise = rotate(hor_flipped, degree_of_rotation_clockwise, resize=True)
    io.imsave(path + image_name + '_' + str(abs(degree_of_rotation_clockwise)) + 'c.' + image_extension,
              rotated_clockwise)

    # fig, axes = plt.subplots(nrows=2, ncols=2,
    #                          sharex=True, sharey=True, figsize=(10, 10))
    # ax = axes.ravel()
    #
    # ax[0].imshow(image, cmap=plt.cm.gray)
    # ax[0].set_title('Original')
    # ax[1].imshow(vert_flipped, cmap=plt.cm.gray)
    # ax[1].set_title('Vertical flip')
    # ax[2].imshow(rotated_clockwise, cmap=plt.cm.gray)
    # ax[2].set_title('Horizontal flip rotated {} counterclockwise'.format(degree_of_rotation_counterclockwise))
    # ax[3].imshow(rotated_counterclockwise, cmap=plt.cm.gray)
    # ax[3].set_title('Rotated {} counterclockwise'.format(degree_of_rotation_counterclockwise))
    #
    # for a in ax:
    #     a.axis('on')
    # fig.tight_layout()
    # plt.show()


def gaussian_filter(img_path):
    # Read image
    image = io.imread(img_path)
    path = img_path.rsplit('/', 1)[0] + '/'

    image_name = img_path.split('/')[-1].split('.')[0]
    image_extension = img_path.split('/')[-1].split('.')[1]

    """
    It works best when given two sigma values because that is the way
    the filter omits pixels to create 'blur'
    Truncate give the radius of the kernel in terms of the two sigmas
    These combination works really well
    From: https://datacarpentry.org/image-processing/06-blurring/
    """
    max_smooth = gaussian(image, sigma=(3.14159, 13), truncate=3.5, multichannel=True)
    # io.imsave(path + image_name + '_g.' + image_extension, max_smooth)

    fig, axes = plt.subplots(nrows=1, ncols=2,
                             sharex=True, sharey=True, figsize=(10, 10))
    ax = axes.ravel()

    ax[0].imshow(max_smooth, cmap=plt.cm.gray)
    ax[0].set_title('Max Smoothing ')
    ax[1].imshow(image, cmap=plt.cm.gray)
    ax[1].set_title('Original')

    for a in ax:
        a.axis('on')
    fig.tight_layout()
    plt.show()

def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

def new_shear(img_path):
    image = io.imread(img_path) / 255
    path = img_path.rsplit('/', 1)[0] + '/'
    image_name = img_path.split('/')[-1].split('.')[0]
    image_extension = img_path.split('/')[-1].split('.')[1]


    # Create Afine transform
    afine_tf = transform.AffineTransform(shear=-0.1)

    # Apply transform to image data
    modified = transform.warp(image, inverse_map=afine_tf, order=2, preserve_range=True)
                              # ,mode='wrap')

    # Display the result
    # io.imsave(path + image_name + '_sh.' + image_extension, modified)

    plt.imshow(np.hstack([image, modified]))
    plt.show()

def histogram_equalization(image_path):
    """Histogram equalization enhances an image with low contrast.
    View: https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html

    Parameters
    ----------
    image_path: str
        The file location of the spreadsheet

    Returns
    -------
    void
        Plots different methods of contrast enhancements
    """
    image = io.imread(image_path)

    # Contrast stretching
    p2, p98 = np.percentile(image, (2, 98))
    image_rescaled = exposure.rescale_intensity(image, in_range=(p2, p98))

    # Equalization
    image_equalized = exposure.equalize_hist(image)

    # Adaptive Equalization
    image_adaptive_eq = exposure.equalize_adapthist(image, clip_limit=0.03)

    fig, axes = plt.subplots(nrows=2, ncols=2,
                             sharex=True, sharey=True, figsize=(10, 10))
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original')
    ax[1].imshow(image_rescaled, cmap=plt.cm.gray)
    ax[1].set_title('Contrast stretching')
    ax[2].imshow(image_equalized, cmap=plt.cm.gray)
    ax[2].set_title('Histogram equalization')
    ax[3].imshow(image_adaptive_eq, cmap=plt.cm.gray)
    ax[3].set_title('Adaptive equalization')

    for a in ax:
        a.axis('on')
    fig.tight_layout()
    plt.show()

    # fig = plt.figure(figsize=(8, 5))
    # axes = np.zeros((2, 4), dtype=np.object)
    # axes[0, 0] = fig.add_subplot(2, 4, 1)
    # for i in range(1, 4):
    #     axes[0, i] = fig.add_subplot(2, 4, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
    # for i in range(0, 4):
    #     axes[1, i] = fig.add_subplot(2, 4, 5 + i)
    #
    # ax_img, ax_hist, ax_cdf = plot_img_and_hist(image, axes[:, 0])
    # ax_img.set_title('Low contrast image')
    #
    # y_min, y_max = ax_hist.get_ylim()
    # ax_hist.set_ylabel('Number of pixels')
    # ax_hist.set_yticks(np.linspace(0, y_max, 5))
    #
    # ax_img, ax_hist, ax_cdf = plot_img_and_hist(image_rescaled, axes[:, 1])
    # ax_img.set_title('Contrast stretching')
    #
    # ax_img, ax_hist, ax_cdf = plot_img_and_hist(image_equalized, axes[:, 2])
    # ax_img.set_title('Histogram equalization')
    #
    # ax_img, ax_hist, ax_cdf = plot_img_and_hist(image_adaptive_eq, axes[:, 3])
    # ax_img.set_title('Adaptive equalization')
    #
    # ax_cdf.set_ylabel('Fraction of total intensity')
    # ax_cdf.set_yticks(np.linspace(0, 1, 5))
    #
    # # prevent overlap of y-axis labels
    # fig.tight_layout()
    # plt.show()





def shear(img_path):
    image = io.imread(img_path)/255
    path = img_path.rsplit('/', 1)[0] + '/'
    image_name = img_path.split('/')[-1].split('.')[0]
    image_extension = img_path.split('/')[-1].split('.')[1]

    shear_value = 0.3

    tf = transform.AffineTransform(shear=shear_value)
    vertical_shear = transform.warp(image, tf, order=1, preserve_range=True, mode='constant')

    # io.imsave(path + image_name + '_sh.' + image_extension, vertical_shear)

    plt.imshow(np.hstack([image, vertical_shear]))
    plt.show()


def contrast_adjustment(img_path):
    image = io.imread(img_path)
    path = img_path.rsplit('/', 1)[0] + '/'
    image_name = img_path.split('/')[-1].split('.')[0]
    image_extension = img_path.split('/')[-1].split('.')[1]

    # Gamma
    gamma_corrected = exposure.adjust_gamma(image, 2)
    # io.imsave(path + image_name + '_c.' + image_extension, gamma_corrected)


    plt.imshow(np.hstack([image, gamma_corrected]))
    plt.show()


def perform_data_augmentation():
    complete_path()
    for image_file in images_names:
        if image_file.endswith('.png'):
            flip_rotate_images(image_file)
            gaussian_filter(image_file)
            shear(image_file)
            contrast_adjustment(image_file)


def yolo_line_to_shape(image, xcen, ycen, w, h):

    width = image.shape[1]
    height = image.shape[0]

    xmin = max(float(xcen) - float(w) / 2, 0)
    xmax = min(float(xcen) + float(w) / 2, 1)
    ymin = max(float(ycen) - float(h) / 2, 0)
    ymax = min(float(ycen) + float(h) / 2, 1)

    xmin = int(width * xmin)
    xmax = int(width * xmax)
    ymin = int(height * ymin)
    ymax = int(height * ymax)

    list_ = [xmin, ymin, xmax, ymax]
    n_list_ = np.array(list_)

    return n_list_


# TODO: JUST DO WITH DE PAPER SPACE BLOG, WITH RESIZING
def rotate_by_blog(angle, img, bboxes):
    """
    Args:
        img (PIL Image): Image to be flipped.
    Returns:
        PIL Image: Randomly flipped image.


    """

    angle = angle
    print(angle)

    w, h = img.shape[1], img.shape[0]
    cx, cy = w // 2, h // 2

    corners = get_corners(bboxes)

    corners = np.hstack((corners, bboxes[:, 4:]))

    img = rotate_im(img, angle)

    corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)

    new_bbox = get_enclosing_box(corners)

    scale_factor_x = img.shape[1] / w

    scale_factor_y = img.shape[0] / h

    img = cv2.resize(img, (w, h))

    new_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]

    bboxes = new_bbox

    bboxes = clip_box(bboxes, [0, 0, w, h], 0.25)

    return img, bboxes



def read_file_test(txt_path, img_path):
    image = io.imread(img_path)

    print("Original image shape {}".format(image.shape))

    boxes = []
    file = open(txt_path, 'r')
    for i, line in enumerate(file):
        print(line)
        """In order to strip the string you must save the returned object, that's because strings are immutable"""
        stripped = line.strip('\n')
        just_coord = np.array(stripped.split(' ')[1::])
        np_coord = just_coord.astype(np.float)
        print(np_coord)
        boxes.insert(i, np_coord)

    list = yolo_line_to_shape(image, boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3])
    list_of_boxes = []
    list_of_boxes.insert(0, list)

    hor_flipped = image[:, ::-1]

    print("Yolo x1, y1, x2, y2 {} and its type {}".format(list, type(list)))
    # 107  49 594 281
    corners = get_corners(np.array(list_of_boxes))

    rotated_img, rotated_bboxes = rotate_by_blog(-30, hor_flipped, np.array(list_of_boxes, dtype=float))
    draw_rect(image, np.array(list_of_boxes), rotated_img, rotated_bboxes)

    print("Rotated image shape {}, rotated bboxes {}".format(rotated_img.shape, rotated_bboxes.shape))
    """
    TODO: Finally rotate with the given methods and make sure to reverse engineer yolo_line_toshape
    """



    # plt.imshow( rotated_img)
    # plt.show()


    #draw_rect(image, np.array(list_of_r_boxes))
    # plt.imshow(image)
    # for i, coord in enumerate(rotated_corners[0]):
    #     if i % 2 != 0:
    #         plt.scatter([rotated_corners[0][i - 1]], [rotated_corners[0][i]])
    # # plt.scatter([351], [15])
    # plt.show()

image_path = '/home/alvaro/Documents/MaskForMuscle/001.png'
inner_path = '/home/alvaro/Documents/MaskForMuscle/Cropped/057_cropped.png'
#
# complete_path()
# for img in images_names:
#     name = img.split('/')[-1].split('.')[0]
#
#     crop_image(img, f'/home/alvaro/Documents/MaskForMuscle/Cropped/{name}_cropped.png')



# complete_path()
# read_images()
# rotate_images()
# crop_image(image_path, inner_path)
# sharpen_image(inner_path)
# flip_rotate_images(image_path)
# gaussian_filter(inner_path)
# histogram_equalization(inner_path)
# new_shear(inner_path)
# contrast_adjustment(inner_path)
new_shear(inner_path)
# perform_data_augmentation()
# read_file_test("/home/alvaro/Pictures/rot/image_to_rot.txt", "/home/alvaro/Pictures/rot/image_to_rot.png")
