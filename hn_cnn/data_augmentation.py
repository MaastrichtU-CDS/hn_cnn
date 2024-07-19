import random

import numpy as np
from PIL import Image
from torchvision import transforms

from hn_cnn.constants import *
from hn_cnn.utils import update_parameters

DEFAULT_AUGMENTATION = {
    HORIZONTAL_FLIP: 0.5,
    VERTICAL_FLIP: 0.5,
    ROTATE_90: 0.75,
    ROTATION: 10,
}

def rotate90(image):
    """ Randomly rotate an image by 90, 180 or 270 degrees
    """
    return Image.fromarray(
        np.rot90(np.array(image), k = random.randint(1, 3))
    )

def get_training_augmentation(augment = {}):
    augmentations = update_parameters(augment, DEFAULT_AUGMENTATION)
    print(augmentations)
    return transforms.Compose([
        #transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=augmentations[HORIZONTAL_FLIP]),
        transforms.RandomVerticalFlip(p=augmentations[VERTICAL_FLIP]),
        transforms.RandomApply([
            transforms.Lambda(rotate90),
        ], p=augmentations[ROTATE_90]),
        transforms.RandomRotation((-augmentations[ROTATION], augmentations[ROTATION])),
        transforms.ToTensor(),
        #transforms.Normalize((mean), (std)),
    ])

TRANSFORM_TO_TENSOR = transforms.Compose([
    #transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    #transforms.Normalize((mean), (std)),
])
