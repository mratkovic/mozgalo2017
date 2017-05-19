import os
import skimage
import skimage.io
import pickle
import glob
import shutil
from tqdm import tqdm

import numpy as np
import pandas as pd
import logging
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, adjusted_rand_score

from .utils import load_image, save_image


# Definitions

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_CURRENT_DIR, '..', '..', '')

DATASETS_ROOT = os.path.join(_PROJECT_ROOT, 'dataset')
CLUSTER_OUT_ROOT = os.path.join(_PROJECT_ROOT,'clusters/')

full_path = lambda x: os.path.join(DATASETS_ROOT, x)

datasets = {
 'mozgalo': full_path('mozgalo_dataset'),
 'cifar10': full_path('cifar10'),
 'cats_dogs': full_path('cats_dogs'),
 'stl10': full_path('stl10')
}

dataset_imgs_root = {
    name: os.path.join(datasets[name], 'imgs') for name in datasets.keys()
}

features_file_name = {
    'inceptionv3': 'features_v3_pickle',
    'inceptionv4': 'features_v4_pickle',
    'resnet50: ': 'features_resnet50_pickle'
}


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
        self.rebuild_path_index()
       
    def rebuild_path_index(self):
        self.path_to_id = {os.path.basename(p):i for i, p in enumerate(self.paths)}

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
        self.rebuild_path_index()

    def store_csv_features(self, csv_path):
        df = self._features_to_df()
        df.to_csv(csv_path)

    def load_csv_features(self, csv_path):
        self._features_from_df(pd.read_csv(csv_path))
        self.load_imgs()

    def store_features(self, path):
        df = self._features_to_df()
        df.to_pickle(path)

    def load_features(self, path, skip_img_load=False):
        self._features_from_df(pd.read_pickle(path))

        if not skip_img_load:
            self.load_imgs()

    def save_clusters_to_file(self, path, labels, force_reload=False):
        distinct_labels = np.unique(labels)
        if os.path.exists(path):
            shutil.rmtree(path)
        for label in tqdm(distinct_labels):
            indices = np.where(labels == label)[0]
            os.makedirs('%s/%d/' % (path, label), exist_ok=True)
            for index in indices:
                try:
                    if force_reload:
                        img = self.get_img(index)
                    else:
                        img = self.imgs[index]
                    save_image('%s/%d/%s' % (path, label, os.path.basename(self.paths[index])), img)
                except:
                    logging.warning("Index error %s, paths len %s", index, len(self.path))



def get_labels_path(dataset_name):
    return os.path.join(datasets[dataset_name], 'labels.pickle')


def get_labels(dataset_name):
    labels_path = get_labels_path(dataset_name)

    with open(labels_path, 'rb') as f:
        labels_dict = pickle.load(f)
    return {os.path.basename(p): l for p, l in labels_dict.items()}


def get_n_different_classes(dataset_name):
    if os.path.exists(get_labels_path(dataset_name)):
        return len({lab for p, lab in get_labels(dataset_name).items()})
    return 'unknown'


def pickled_features_path(dataset_name, features_extraction_model):
    file_name = features_file_name[features_extraction_model.name]
    return os.path.join(datasets[dataset_name], file_name)

def get_scores(y_true, y_pred, average='macro', pos_label=None):
    p = precision_score(y_true, y_pred, average=average, pos_label=pos_label)
    r = recall_score(y_true, y_pred, average=average, pos_label=pos_label)
    f1 = f1_score(y_true, y_pred, average=average, pos_label=pos_label)

    return [p, r, f1]


def get_dominant_labels(root, labels_dict):
    files = os.listdir(root)
    labels = []
    for file in files:
        labels.append(labels_dict[file])

    c = Counter(labels)
    dominant_label = c.most_common(1)[0][0]
    return files, dominant_label


def dump_wrong_images(wrong_paths, clusters_path, dataset_name):
    out_dir = os.path.join(os.path.dirname(clusters_path), os.path.basename(clusters_path) + '_wrong')
    os.makedirs(out_dir, exist_ok=True)

    for p in wrong_paths:
        in_path = os.path.join(dataset_imgs_root[dataset_name], p)
        out_path = os.path.join(out_dir, p)
        shutil.copy2(src=in_path, dst=out_path)


def calculate_stats(y, y_pred, dataset_name):
    a = accuracy_score(y, y_pred)
    macro_scores = get_scores(y, y_pred, 'macro')
    ari = adjusted_rand_score(y, y_pred)

    columns = ['dataset', 'accuracy', 'precision', 'recall', 'F1', 'ARI']
    data = [dataset_name, a] + macro_scores + [ari]

    return pd.DataFrame([data], columns=columns)


def compare_cluster_to_labels(clusters_path, dataset_name):
    dirs = os.listdir(clusters_path)
    labels_dict = get_labels(dataset_name)

    paths = []
    y_pred = []
    for d in dirs:
        path = os.path.join(clusters_path, d)
        files, label = get_dominant_labels(path, labels_dict)
        paths.extend(files)
        y_pred.extend([label] * len(files))

    y = np.array([labels_dict[p] for p in paths])
    y_pred = np.array(y_pred)
    paths = np.array(paths)
    wrong_paths = paths[y != y_pred]
    dump_wrong_images(wrong_paths, clusters_path, dataset_name)

    return calculate_stats(y, y_pred, dataset_name)
