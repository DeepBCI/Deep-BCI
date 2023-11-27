import numpy as np
import math
from sklearn.cluster import KMeans


def clustering(signal, k = 20, t = 10, step = 3):
    """
    Implements a clustering algorithm designed to extract the most prevalent clusters of subsequences from a given signal.


    signal: input signal
    k: number of clusters
    t: length of subsequences
    step: distance between subsequences
    """

    x = signal

    subsequences = np.zeros((math.floor((len(x) - t) / step), t))
    for i in range(np.shape(subsequences)[0]):
        subsequences[i] = x[step * i:(step * i + t)]

    closest_indexes = []
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(subsequences)
    centers = kmeans.cluster_centers_

    for center_index, center in enumerate(centers):
        # Calculate Euclidean distances between the cluster center and all data points
        distances = np.linalg.norm(subsequences - center, axis=1)

        # Find the index of the closest data point to the current cluster center
        closest_index = np.argmin(distances)
        closest_indexes.append(closest_index)

    print('Clustering of is done')
    return closest_indexes



def subsequence_augmentation(signal, closest_indexes, l = 11, t = 10, step = 3, noise_amount = 15.5):
    """
    Provides noise augmentation while preserving the part of closest subsequences to the cluster centers obtained from clustering.


    signal: input signal
    l: number of clusters to choose
    t: length of subsequences
    step: distance between subsequences
    """

    x = signal[0]
    chosen_indexes = np.random.choice(closest_indexes, l, replace=False)

    shapes = np.zeros(len(x))

    for i in chosen_indexes:
        shapes[(step*i):((step*i) + t)] = np.ones(t)
    y = (np.multiply(add_noise(x, noise_amount), (-1) * shapes + 1) + np.multiply(x, shapes)).unsqueeze(0)

    return y


def add_noise(signal, noise_level):
    """
    Adds noise to the data.

    signal: input signal
    noise_level: level of noise
    """

    x = signal
    noise = np.random.normal(1, noise_level, np.shape(signal)[0])
    y = x + noise
    return y
