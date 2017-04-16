import os
import skimage
import skimage.io
import glob
import shutil
from tqdm import tqdm

import numpy as np
import pandas as pd
import logging

from .utils import load_image, save_image

class ImageDataset:
    def __init__(self, root_dir):
        if not os.path.exists(root_dir):
            full_path = os.path.abspath(root_dir)
            logging.error('No such dir %s' % full_path)
            raise FileNotFoundError('No directory %s' % full_path)

        self.root_dir = root_dir
        self.features = []
        self.imgs = []
        self.paths = []

        self._discover_paths()

    def _discover_paths(self):
        self.paths = glob.glob(os.path.join(self.root_dir, '*'))
        self.paths = [os.path.realpath(p) for p in self.paths]

    @property
    def size(self):
        return len(self.paths)

    def load_imgs(self):
        logging.debug('Loading imgs in dataset')
        self.imgs = np.array([self.get_img(i) for i in tqdm(range(self.size))])

    def get_img(self, index):
        return load_image(self.paths[index])

    def extract_features(self, model):
        self.features = model.extract_features(self.paths)

    def _features_to_df(self):
        df = pd.DataFrame(self.features)
        names = [os.path.basename(p) for p in self.paths]
        df.insert(0, 'name', names)
        df.set_index('name', inplace=True)

        return df

    def _features_from_df(self, df):
        self.features = df.as_matrix()
        self.paths = [os.path.join(self.root_dir, p) for p in df.index.values]

    def store_csv_features(self, csv_path):
        df = self._features_to_df()
        df.to_csv(csv_path)

    def load_csv_features(self, csv_path):
        self._features_from_df(pd.read_csv(csv_path))

    def store_features(self, path):
        df = self._features_to_df()
        df.to_pickle(path)

    def load_features(self, path):
        self._features_from_df(pd.read_pickle(path))

    def save_cluset_to_file(self, path, labels):
        distinct_labels = np.unique(labels)
        if os.path.exists(path):
            shutil.rmtree(path)
        for label in tqdm(distinct_labels):
            indeces = np.where(labels == label)[0]
            os.makedirs('%s/%d/' % (path, label), exist_ok=True)
            for index in indeces:
                save_image('%s/%d/%s' % (path, label, os.path.basename(self.paths[index])), self.imgs[index])
