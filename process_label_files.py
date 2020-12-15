"""
The only purpose of this script is to process the label files of the masks
We only need the points that form the polygon
"""
import json
from os import listdir
from os.path import isfile, join
from random import choice
import matplotlib.pyplot as plt
from  matplotlib.image import imread
from skimage.color import rgb2gray
from skimage.color import rgb2grey
from numpy import zeros
import numpy as np
import cv2
from skimage import img_as_ubyte



def remove_image_data(path):
    """
    Removes the imageData field in the label json file
    :param path: Folder containing the labels
    :return:
    """
    for file in listdir(path):
        file_path = join(path, file)
        with open(file_path, 'r') as json_file:
            labels = json.load(json_file)

            if "imageData" in labels:
                clean_label_file = remove_key(labels, "imageData")
                with open(file_path, 'w') as modified_json_file:
                    json.dump(clean_label_file, modified_json_file)


def remove_key(d, key):
    """
    Helper function to delete the key of a dictionary but returns a shallow copy of it
    :param d: Dictionary
    :param key: Key to delete
    :return: Shallow copy of the dictionary without the given key
    """
    r = dict(d)
    del r[key]
    return r

def draw_mask_over_image(image_path, label_path):
    """
    Selects a random image and applies the mask based on
    its json label
    :param image_path: Folder where the images are
    :param label_path: Folder where the labels are
    :return:
    """
    random_image = choice(listdir(image_path))
    random_image_file = join(image_path, random_image)
    image = imread(random_image_file)
    one_channel_image = image[:, :, 1]
    print(f'One channel image type {one_channel_image.dtype}')
    print(f'Biggest pixel val {np.max(one_channel_image)}')
    # This image is float32, we need to change its values to uint8
    # First normalize values, dividing by the largest so we have values [0, ..., 1]
    one_ch_norm = one_channel_image / np.max(one_channel_image)
    print(f'Biggest pixel val after norm {np.max(one_ch_norm)}')
    # Then multiply the entire matrix by 255, to avoid loss of info in the pixels
    one_ch_norm = one_ch_norm * 255
    # Finally cast to unint8
    one_ch_norm = one_ch_norm.astype(np.uint8)
    # print(one_channel_image.shape)
    random_label = random_image.replace(".png", ".json")
    with open(join(label_path, random_label), 'r') as json_file:
        labels = json.load(json_file)
        mask_points = labels["shapes"][0]["points"]
    x = []
    y = []
    for point in mask_points:
        x.append(point[0])
        y.append(point[1])
    h, w = one_channel_image.shape
    # Get logic on this down, I think it has something to do with the amount of rows and cols
    # bless up
    background = zeros((h, w))
    print(f'Image {random_image}: Height: {h} and Width: {w}')
    print(f'Number of vertices: {len(mask_points)} of the mask')
    # plt.imshow(background)
    # plt.fill(x, y, facecolor='gray', edgecolor='gray')
    # plt.show()
    # cv2.imshow('pull_up', background)
    # First we generate a mask, a dark background

    # Then we apply the mask with color 255, is white?
    mask = cv2.fillPoly(background, np.int32([mask_points]), color=255)
    print(f'Mask shape: {mask.shape}')
    cv2.imshow('one_channel', one_ch_norm)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)

    just_name = random_image.split('.')[0]
    extension = random_image.split('.')[1]
    mask_name = just_name + '_mask.' + extension
    one_channel_name = just_name + '_one_ch.' + extension

    mask_path = join(image_path + 'sem_seg', mask_name)
    one_channel_images_path = join(image_path + 'one_channel', one_channel_name)
    cv2.imwrite(one_channel_images_path, one_ch_norm)
    cv2.imwrite(mask_path, mask)


def to_grayscale_segmentation(image_path, label_path):
    """
    Creates the semantic segmentation ground truth files of the images,
    These have _mask appended to the name file
    :param image_path: Folder where the images are
    :param label_path: Folder where the labels are
    :return:
    """
    for image_file in listdir(image_path):
        random_image_file = join(image_path, image_file)
        image = imread(random_image_file)
        one_channel_image = image[:, :, 1]
        random_label = image_file.replace(".png", ".json")

        with open(join(label_path, random_label), 'r') as json_file:
            labels = json.load(json_file)
            mask_points = labels["shapes"][0]["points"]
        x = []
        y = []
        for point in mask_points:
            x.append(point[0])
            y.append(point[1])
        h, w = one_channel_image.shape
        background = zeros((h, w))
        # Then we apply the mask with color 255, is white?
        mask = cv2.fillPoly(background, np.int32([mask_points]), color=255)
        just_name = image_file.split('.')[0]
        extension = image_file.split('.')[1]
        mask_name = just_name + '_mask.' + extension
        mask_path = join(image_path.replace('images/',"") + 'sem_seg/', mask_name)
        cv2.imwrite(mask_path, mask)


def create_one_channel_images(image_path):
    """
    Extract only one channel of the image
    New files are appended with one_channel_png
    :param image_path: Folder where the images live
    :return:
    """
    for image_file in listdir(image_path):
        random_image_file = join(image_path, image_file)
        image = imread(random_image_file)
        one_channel_image = image[:, :, 1]
        one_ch_norm = one_channel_image / np.max(one_channel_image)
        one_ch_norm = one_ch_norm * 255
        one_ch_norm = one_ch_norm.astype(np.uint8)
        just_name = image_file.split('.')[0]
        extension = image_file.split('.')[1]
        one_channel_name = just_name + '_one_ch.' + extension

        one_channel_images_path = join(image_path.replace('images/',"") + 'one_channel', one_channel_name)
        cv2.imwrite(one_channel_images_path, one_ch_norm)


# draw_mask_over_image("/home/alvaro/Documents/MaskForMuscle/Cropped/", "/home/alvaro/Documents/MaskForMuscle/Cropped/labels/")
to_grayscale_segmentation("/home/alvaro/Documents/MaskForMuscle/Cropped/images/", "/home/alvaro/Documents/MaskForMuscle/Cropped/labels/")
#create_one_channel_images("/home/alvaro/Documents/MaskForMuscle/Cropped/images/")