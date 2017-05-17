from __future__ import print_function, division

import os
import pickle
import numpy as np
import skimage as ski
import skimage.io
from tqdm import tqdm

def init_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def save_image(img, path):
    img = img.astype(np.uint8)
    ski.io.imsave(path, img)


def show_image(img):
    img = img.astype(np.uint8)
    ski.io.imshow(img)
    ski.io.show()


def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y


def unpickle(file):
    import _pickle as cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding='latin')
    fo.close()
    return dict


def get_label_names(blob):
    label_dict = unpickle(blob)['label_names']
    return np.array(label_dict)


DATA_DIR = '/home/marko/Downloads/cifar-10-batches-py/'
img_height, img_width, num_channels = 32, 32, 3
train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []
for i in range(1, 6):
    subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
    train_x = np.vstack((train_x, subset['data']))
    train_y += subset['labels']

train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
train_y = np.array(train_y, dtype=np.int32)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
test_y = np.array(subset['labels'], dtype=np.int32)

valid_size = 5000
train_x, train_y = shuffle_data(train_x, train_y)
valid_x = train_x[:valid_size, ...]
valid_y = train_y[:valid_size, ...]
train_x = train_x[valid_size:, ...]
train_y = train_y[valid_size:, ...]

dump_dir = '../dataset/cifar10/imgs/'
label_path = os.path.join('../dataset/cifar10/labels.picle')
os.makedirs(dump_dir, exist_ok=True)

class_names = get_label_names(os.path.join(DATA_DIR, 'batches.meta'))
print(class_names)
labels = {}

for i, (img, lab) in tqdm(enumerate(list(zip(train_x, train_y)))):
    path = os.path.join(dump_dir, '%05d.jpg' % i)
    save_image(img, path)
    labels[path] = lab

with open(label_path, 'wb') as f:
    pickle.dump(labels, f)