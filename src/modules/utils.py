import os
import cv2
import time
import logging
import skimage.io
from logging.config import fileConfig


def init_logging():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    logging_conf = os.path.join(dir_path, '..', 'logging_config.ini')
    if not os.path.exists(logging_conf):
        raise FileNotFoundError("Expected logging_config.ini in src dir")

    fileConfig(logging_conf)

class Timer:
    def __init__(self, title=None, verbose=True, print_f=logging.info):
        self.verbose = verbose
        self.title = title
        self.print_f = print_f

        if self.title is None:
            self.title = ''

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

        if self.verbose:
            self.print_f("Duration %s %d s" % (self.title, self.interval))


def init_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def bgr_to_rgb(img):
    """ To show using plt.imshow because cv2 uses BGR"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img):
    """ skimage to cv2 format """
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def load_image(img_path):
    input_img = skimage.io.imread(img_path)
    shape = input_img.shape
    if len(shape) == 2:
        input_img = skimage.color.gray2rgb(input_img)

    if len(shape) > 2 and shape[2] == 4:
        input_img = skimage.color.rgba2rgb(input_img)

    return input_img

def save_image(path, img):
    skimage.io.imsave(path, img)
    #cv2.imwrite(path, img)
