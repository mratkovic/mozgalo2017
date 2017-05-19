import pickle
import numpy as np
from tqdm import  tqdm
import os
import sys
from skimage import io

# stl10
def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        return images




# path to the directory with the data
DATA_DIR = './data'

# url of the binary data
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

# path to the binary train file with image data
DATA_PATH = '/home/marko/Downloads/stl10_binary/train_X.bin'
# path to the binary train file with labels
LABEL_PATH = '/home/marko/Downloads/stl10_binary/train_y.bin'

out_dir = '../dataset/stl10/imgs/'
os.makedirs(out_dir, exist_ok=True)
labels_dict = {}


labels = read_labels(LABEL_PATH)
images = read_all_images(DATA_PATH)


for i, img in tqdm(enumerate(images)):
    img_name = "%06d.jpg" % i
    out_path = os.path.join(out_dir, img_name)
    io.imsave(out_path, img)
    labels_dict[out_path] = labels[i]

labels_path = '../dataset/stl10/labels.pickle'
with open(labels_path, 'wb') as f:
    pickle.dump(labels_dict, f)
