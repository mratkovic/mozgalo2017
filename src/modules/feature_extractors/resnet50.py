import os
import sys
from keras.preprocessing import image
import numpy as np

from tqdm import tqdm

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_ROOT = os.path.join(_CURRENT_DIR, '..', '..', 'models/')
sys.path.append(os.path.join(_CURRENT_DIR, '..', 'external', 'deep-learning-models'))

from resnet50 import ResNet50
from imagenet_utils import preprocess_input


class Resnet50:

    def __init__(self):
        self.name = 'resnet50'
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


