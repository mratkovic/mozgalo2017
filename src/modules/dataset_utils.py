import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

from src.modules.utils import *

def show_grid(imgs):
    n = len(imgs)
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(1.0 * n / rows))

    fig, ax = plt.subplots(rows, cols, figsize=(16, 12))
    ax = np.array(ax).flatten()

    for i, img in enumerate(imgs):
        ax[i].imshow(img)

    fig.tight_layout()
    plt.show()


def show_random_sample(root, n_images=25):
    files = np.array([os.path.join(root, p) for p in os.listdir(root)])
    n = len(files)
    subset = np.random.choice(n, n_images, replace=False)
    sample = files[subset]
    imgs = [io.imread(p) for p in sample]
    show_grid(imgs)


def find_similary_from_dataset(img_path, dataset, get_features_f, k=10, similarity=cosine_similarity):
    features = get_features_f(img_path)
    db_features = dataset.features
    dists = similarity([features], db_features)[0]
    closest_indices = dists.argsort()[-k:][::-1]

    return dataset.imgs[closest_indices]


def plot_closest_results(img_path, get_features_f, dataset, k=12):
    closest = find_similary_from_dataset(img_path, dataset, get_features_f, k)
    plt.figure(figsize=(20, 15))

    rows = int(np.ceil(np.sqrt(k)))
    cols = int(np.ceil(k/rows))

    test_img = io.imread(img_path)
    plt.figure(figsize=(16, 8))
    plt.imshow(test_img)
    plt.title("Query image - %s" % img_path)
    plt.show()

    fig, ax = plt.subplots(rows, cols, figsize=(16,12))
    ax = np.array(ax).flatten()

    for i, img in enumerate(closest):
        ax[i].imshow(img)

    fig.tight_layout()
    plt.show()