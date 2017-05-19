import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from skimage.transform import resize
from matplotlib import offsetbox

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
    
def plot_embedding(X, y, images, title=None, figsize=(20,20), img_size=(65 ,65)):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)     
    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    colormap = plt.cm.gist_ncar
    colors = [colormap(i) for i in np.linspace(0, 1,len(set(y)))]
    for i in range(X.shape[0]):
        plt.scatter(X[i, 0], X[i, 1], color=colors[y[i]])
    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(images.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 6e-3:
                ## don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            img = resize(images[i], img_size)
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(img, cmap=plt.cm.gray_r), X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
