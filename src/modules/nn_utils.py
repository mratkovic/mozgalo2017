import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from glob import glob

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skimage
from skimage.transform import resize
from sklearn.decomposition import PCA
from tqdm import tqdm
import shutil
import _pickle as cPickle
import cv2
from .utils import  load_image
from .models.slim.preprocessing import inception_preprocessing
from .models.slim.nets.inception_v4 import inception_v4_arg_scope, inception_v4



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


def get_img(path):
    img = io.imread(path)
    img = resize(img, (229, 229))
    return img


def show_img(path):
    plt.figure(figsize=(16, 12))
    img = get_img(path)
    plt.imshow(img)
    plt.show()

def maxpool2d(x, k=2, stride=2, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1],
                          padding=padding)


class InceptionNetV4:
    def __init__(self):
        self.model_checkpoint_path = os.path.join(_MODELS_ROOT, 'inception_v4', 'inception_v4.ckpt')

    def create_graph(self):
        slim = tf.contrib.slim
        self.images = tf.placeholder(tf.float32, [None, 229, 229, 3])

        with slim.arg_scope(inception_v4_arg_scope()):
            self.logits, self.end_points = inception_v4(self.images, num_classes=1001, is_training=False)

        exclude = ['InceptionV4/AuxLogits']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(self.sess, self.model_checkpoint_path)

        # Creating the prediction endpoint by maxpooling all the layers
        self.pred = self.create_prediction(self.end_points)

        # inputs for preprocessing
        self.placeholder_input = tf.placeholder(tf.float32, [None, None, None])
        self.preprocessed_img = inception_preprocessing.preprocess_image(
            self.placeholder_input, 229, 229, False
        )

    # This creates a concatenated vector with the result of max pooling all the endpoints with maximum size kernel for them
    # for example an output of convolution is 12x12x1024, then max pool will be with kernel [1,12,12,1] and padding VALID
    # the resulting vector will be 1,1,1024 for each image
    def create_prediction(self, endpoints):
        values_to_concat = []
        for end in sorted(endpoints):
            if 'Pred' in end or 'Log' in end:
                continue
            all_channels_maxpool = maxpool2d(
                self.end_points[end],
                self.end_points[end].shape[1],
                padding='VALID'
            )
            values_to_concat.append(all_channels_maxpool)
        concatenated = tf.concat(values_to_concat, 3)
        return concatenated

    @staticmethod
    def _label_lookup():
        file_path = os.path.join(_MODELS_ROOT, 'inception', 'labels_imagenet.txt')

        with open(file_path, 'r') as inf:
            label_lookup = eval(inf.read())
        return {int(k): v for k, v in label_lookup.items()}

    def extract_features(self, imgs_path_list):
        fts = []
        for img_path in tqdm(imgs_path_list):
            img_fts = self._extract_featres(img_path)
            fts.append(img_fts)
        return np.array(fts)

    def _extract_featres(self, img_path):
        input_img = io.imread(img_path)
        shape = input_img.shape
        if len(shape) == 2:
            input_img = skimage.color.gray2rgb(input_img)

        if len(shape) > 2 and shape[2] == 4:
            input_img = skimage.color.rgba2rgb(input_img)
        input_img = resize(input_img, (229, 229), mode='constant')
        preprocessed_img = self.sess.run(self.preprocessed_img,
                             feed_dict={self.placeholder_input: input_img})
        res = self.sess.run(self.pred, feed_dict={self.images: [preprocessed_img]})
        return res[0].flatten()



class InceptionNetV4:
    def __init__(self):
        self.model_checkpoint_path = os.path.join(_MODELS_ROOT, 'inception_v4', 'inception_v4.ckpt')

    def create_graph(self):
        slim = tf.contrib.slim
        self.images = tf.placeholder(tf.float32, [None, 229, 229, 3])

        with slim.arg_scope(inception_v4_arg_scope()):
            self.logits, self.end_points = inception_v4(self.images, num_classes=1001, is_training=False)

        exclude = ['InceptionV4/AuxLogits']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(self.sess, self.model_checkpoint_path)

        # Creating the prediction endpoint by maxpooling all the layers
        self.pred = self.create_prediction(self.end_points)

        # inputs for preprocessing
        self.placeholder_input = tf.placeholder(tf.float32, [None, None, None])
        self.preprocessed_img = inception_preprocessing.preprocess_image(
            self.placeholder_input, 229, 229, False
        )

    # This creates a concatenated vector with the result of max pooling all the endpoints with maximum size kernel for them
    # for example an output of convolution is 12x12x1024, then max pool will be with kernel [1,12,12,1] and padding VALID
    # the resulting vector will be 1,1,1024 for each image
    def create_prediction(self, endpoints):
        values_to_concat = []
        for end in sorted(endpoints):
            if 'Pred' in end or 'Log' in end:
                continue
            all_channels_maxpool = maxpool2d(
                self.end_points[end],
                self.end_points[end].shape[1],
                padding='VALID'
            )
            values_to_concat.append(all_channels_maxpool)
        concatenated = tf.concat(values_to_concat, 3)
        return concatenated

    @staticmethod
    def _label_lookup():
        file_path = os.path.join(_MODELS_ROOT, 'inception', 'labels_imagenet.txt')

        with open(file_path, 'r') as inf:
            label_lookup = eval(inf.read())
        return {int(k): v for k, v in label_lookup.items()}

    def extract_features(self, imgs_path_list):
        fts = []
        for img_path in tqdm(imgs_path_list):
            img_fts = self._extract_featres(img_path)
            fts.append(img_fts)
        return np.array(fts)

    def _extract_featres(self, img_path):
        input_img = io.imread(img_path)
        shape = input_img.shape
        if len(shape) == 2:
            input_img = skimage.color.gray2rgb(input_img)

        if len(shape) > 2 and shape[2] == 4:
            input_img = skimage.color.rgba2rgb(input_img)
        input_img = resize(input_img, (229, 229), mode='constant')
        preprocessed_img = self.sess.run(self.preprocessed_img,
                             feed_dict={self.placeholder_input: input_img})
        res = self.sess.run(self.pred, feed_dict={self.images: [preprocessed_img]})
        return res[0].flatten()


import sys
sys.path.append('../src/deep-learning-models/')
from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions

class Resnet50:

    def __init__(self):
        pass

    def create_graph(self):
        self.model = ResNet50(weights='imagenet', include_top=False)

    def extract_features(self, imgs_path_list):
        fts = []
        for img_path in tqdm(imgs_path_list):
            img_fts = self._extract_featres(img_path)
            fts.append(img_fts)
        return np.array(fts)

    def _extract_featres(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = self.model.predict(x)
        return preds.flatten()


