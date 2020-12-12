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
    random_image = choice(listdir(image_path))
    random_image_file = join(image_path, random_image)
    image = imread(random_image_file)
    one_channel_image = image[:, :, 1]
    print(type(one_channel_image))
    print(one_channel_image.shape)
    random_label = random_image.replace(".png", ".json")
    with open(join(label_path, random_label), 'r') as json_file:
        labels = json.load(json_file)
        mask_points = labels["shapes"][0]["points"]
    x = []
    y = []
    for point in mask_points:
        x.append(point[0])
        y.append(point[1])

    print(len(mask_points))
    plt.imshow(image)
    plt.fill(x, y, facecolor='white', edgecolor='white')
    plt.show()


def to_grayscale_segmentation():
    pass



draw_mask_over_image("/home/alvaro/Documents/MaskForMuscle/Cropped/", "/home/alvaro/Documents/MaskForMuscle/Cropped/labels/")