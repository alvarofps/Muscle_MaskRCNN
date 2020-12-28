import detectron2 as d2
from os import listdir
from os.path import join
from skimage.io import imread
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import random
import matplotlib.pyplot as plt


def open_data(images_path, sem_seg_path, type):
    """
    Function that returns the dicts of images and labels
    :return:
    """
    final_list = []
    FILENAME = 'file_name'
    HEIGHT = 'height'
    WIDTH = 'width'
    IMAGE_ID = 'image_id'
    SEM_SEG_FILENAME = 'sem_seg_file_name'
    for index, image in enumerate(listdir(images_path)):
        image_and_sem_seg = {}
        im = imread(join(images_path, image))
        image_and_sem_seg[FILENAME] = join(images_path, image)
        image_and_sem_seg[WIDTH] = im.shape[1]
        image_and_sem_seg[HEIGHT] = im.shape[0]
        image_and_sem_seg[IMAGE_ID] = index
        image_and_sem_seg[SEM_SEG_FILENAME] = join(sem_seg_path, image).replace('.png', '_mask.png')
        final_list.append(image_and_sem_seg)

    # shuffled_list = list(random.shuffle(final_list))
    val_index = int(len(final_list) * 0.2)

    if type == 'val':
        return final_list[0:val_index + 1]
    elif type == 'train':
        return final_list[val_index: len(final_list)]


def register_dataset():
    for type in ['train', 'val']:
        DatasetCatalog.register('muscle_' + type, lambda type=type: open_data('/home/alvaro/Documents/MaskForMuscle/Cropped/augmented/', '/home/alvaro/Documents/MaskForMuscle/Cropped/new_sem_seg/', type))
        MetadataCatalog.get('muscle_' + type).set(stuff_classes=['noise', 'inner_muscle'])

    dicts = open_data('/home/alvaro/Documents/MaskForMuscle/Cropped/augmented/', '/home/alvaro/Documents/MaskForMuscle/Cropped/new_sem_seg/', 'val')
    for d in random.sample(dicts, 3):
        img = imread(d['file_name'])
        visualizer =  Visualizer(img[:, :, ::-1], metadata = MetadataCatalog.get('muscle_val'), scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        plt.imshow(vis.get_image())
        plt.show()


# open_data('/home/alvaro/Documents/MaskForMuscle/Cropped/augmented/', '/home/alvaro/Documents/MaskForMuscle/Cropped/sem_seg/', 'train')
register_dataset()