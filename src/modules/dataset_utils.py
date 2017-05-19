import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
import seaborn as sns
import pandas as pd
import glob
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from itertools import combinations_with_replacement

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


def pca_summary(pca, standardised_data, out=True):
    names = [str(i) for i in range(1, len(pca.explained_variance_ratio_) + 1)]
    a = list(np.std(pca.transform(standardised_data), axis=0))
    b = list(pca.explained_variance_ratio_)
    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_) + 1)]
    columns = pd.Index(["sdev", "varproportion", "cumultiveproportion"])
    summary = pd.DataFrame(list(zip(a, b, c)), index=names, columns=columns)
    return summary


def screeplot(pca, standardised_values):
    y = np.std(pca.transform(standardised_values), axis=0) ** 2
    x = np.arange(len(y)) + 1
    x = x[:50]
    y = y[:50]

    print(x.shape, y.shape)
    plt.plot(x[:500], y[:500], "o-")
    plt.xticks(x[:500], ["Comp." + str(i) for i in x[:500]], rotation=60)
    plt.ylabel("Variance")
    plt.show()


def get_features(dir_path, dataset, features):
    lista = list(map(lambda x: dataset.path_to_id[os.path.basename(x)], glob.glob('%s/*' % dir_path)))
    return features[lista]


def compute_metrics(root_path_of_clusters, features, metrics, verbose=True):
    # ako je ime slike 0001 tada se na indexu 0 u sorted_index nalazi index slike u dataset.paths i dataset.imgs
    clusters = sorted(glob.glob('%s/*' % root_path_of_clusters))
    diag = []
    upper_triang = []
    for d1, d2 in combinations_with_replacement(clusters, 2):
        out = metrics(get_features(d1, features), get_features(d2, features))
        if verbose:
            print("Cluster %s-%s" % (os.path.basename(d1), os.path.basename(d2)))
            print("\t min:  %f" % out.min())
            print("\t max:  %f" % out.max())
            print("\t mean: %f" % out.mean())
        if int(os.path.basename(d1)) == int(os.path.basename(d2)):
            diag.append(out.mean())

        elif int(os.path.basename(d1)) < int(os.path.basename(d2)):
            upper_triang.append(out.mean())
    diag_mean = np.array(diag).mean()
    upper_triag_mean = np.array(upper_triang).mean()
    if verbose:
        print("Mean of metrics within clusters:  %f" % diag_mean)
        print("Mean of metrics between clusters: %f" % upper_triag_mean)
    return diag_mean, upper_triag_mean