import logging
import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tqdm import tqdm

from .utils import load_image


def _create_graph(model_path):
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_ROOT = os.path.join(_CURRENT_DIR, '..', '..', 'models/')


class InceptionNet:
    def __init__(self):
        self.def_file_path = os.path.join(_MODELS_ROOT, 'inception', 'inception_def.pb')
        self.featuremap_tensor_name = 'pool_3:0'
        self.input_placeholder_name = 'DecodeJpeg/contents:0'
        self.softmax_name = 'softmax:0'
        self.label = self._label_lookup()

    def create_graph(self):
        _create_graph(self.def_file_path)


    @staticmethod
    def _load_as_jpeg_bytes(path):
        img = load_image(path)
        tmp_path = './tmp.jpg'
        cv2.imwrite(tmp_path, img)  # store as jpg
        image_data = gfile.FastGFile(tmp_path, 'rb').read()
        os.remove(tmp_path)  # remove tmp file

        return image_data

    @staticmethod
    def _label_lookup():
        file_path = os.path.join(_MODELS_ROOT, 'inception', 'labels_imagenet.txt')

        with open(file_path, 'r') as inf:
            label_lookup = eval(inf.read())
        return {int(k): v for k, v in label_lookup.items()}

    def extract_features(self, imgs_path_list):
        return self._eval_tensor(self.featuremap_tensor_name, imgs_path_list)

    def get_probas(self, imgs_path_list):
        return self._eval_tensor(self.softmax_name, imgs_path_list)

    def get_labels(self, probas):
        class_id = probas.argmax(axis=1)
        return [self.label[i] for i in class_id]

    def _eval_tensor(self, tensor_name, imgs_path_list):
        with tf.Session() as sess:
            vals = []
            endpoint = sess.graph.get_tensor_by_name(tensor_name)

            for ind, image in enumerate(tqdm(imgs_path_list)):
                if not os.path.exists(image):
                    logging.warning('File does not exist %s', image)
                    continue

                feed_dict = {self.input_placeholder_name: self._load_as_jpeg_bytes(image)}
                predictions = sess.run(endpoint, feed_dict)

                vals.append(np.squeeze(predictions))

        return np.array(vals)
