import os
import cv2
import glob
from tqdm import  tqdm

import numpy as np
import pandas as pd
import logging


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
        return cv2.imread(self.paths[index])

    def extract_features(self, model):
        self.features = model.extract_features(self.paths)

    def store_features(self, csv_path):
        df = pd.DataFrame(self.features)
        names = [os.path.basename(p) for p in self.paths]
        df.insert(0, 'name', names)
        df.set_index('name', inplace=True)

        df.to_csv(csv_path)

    def load_features(self, csv_path):
        df = pd.read_csv(csv_path)
        df.set_index('name', inplace=True)

        self.features = df.as_matrix()
        self.paths = [os.path.join(self.root_dir, p) for p in df.index.values]
