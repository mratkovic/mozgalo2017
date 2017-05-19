import os
import sys
import numpy as np
import skimage
import tensorflow as tf
from skimage import io
from skimage.transform import resize
from tqdm import tqdm

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_ROOT = os.path.join(_CURRENT_DIR, '..', '..', '..', 'models/')
# resolve apsolute imports in slim
sys.path.append(os.path.join(_CURRENT_DIR, '..', 'external', 'models', 'slim'))

from src.modules.external.models.slim.nets import inception_v4
from src.modules.external.models.slim.preprocessing import inception_preprocessing

def maxpool2d(x, k=2, stride=2, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1],
                          padding=padding)

class InceptionNetV4:

    def __init__(self):
        self.model_checkpoint_path = os.path.join(_MODELS_ROOT, 'inception_v4', 'inception_v4.ckpt')
        self.name = 'inceptionv4'

    def create_graph(self):
        slim = tf.contrib.slim
        self.images = tf.placeholder(tf.float32, [None, 229, 229, 3])

        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
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

    def extract_features(self, imgs_path_list):
        fts = []
        for img_path in tqdm(imgs_path_list):
            img_fts = self._extract_featres(img_path)
            fts.append(img_fts)
        return np.array(fts)

    def _load_img(self, img_path):
        input_img = io.imread(img_path)
        shape = input_img.shape
        if len(shape) == 2:
            input_img = skimage.color.gray2rgb(input_img)

        if len(shape) > 2 and shape[2] == 4:
            input_img = skimage.color.rgba2rgb(input_img)

        return input_img

    def _extract_featres(self, img_path):
        input_img = self._load_img(img_path)
        input_img = resize(input_img, (229, 229), mode='constant')
        preprocessed_img = self.sess.run(self.preprocessed_img,
                             feed_dict={self.placeholder_input: input_img})

        res = self.sess.run(self.pred, feed_dict={self.images: [preprocessed_img]})
        return res[0].flatten()

