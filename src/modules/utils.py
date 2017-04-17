import os
import cv2
import skimage.io
from logging.config import fileConfig


def init_logging():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    logging_conf = os.path.join(dir_path, '..', 'logging_config.ini')
    if not os.path.exists(logging_conf):
        raise FileNotFoundError("Expected logging_config.ini in src dir")

    fileConfig(logging_conf)


def init_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def bgr_to_rgb(img):
    """ To show using plt.imshow because cv2 uses BGR"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img):
    """ skimage to cv2 format """
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def load_image(path):
    """ Util method to load image - solves warinings in skimage and gif problem with
        cv2
    """
    img = cv2.imread(path)
    if (img is None or len(img) == 0) and path.lower().endswith('gif'):
        img = rgb_to_bgr(skimage.io.imread(path))

    return img

def save_image(path, img):
    if path.lower().endswith('gif'):
        skimage.io.imsave(path, img)
    else:
        cv2.imwrite(path, img)
