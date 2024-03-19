# -*- coding: utf-8 -*-
"""
Package: ClassifAi
File: Tools.py
Version: Feburary 2024
Read README.md for usage instructions
"""
import numpy as np
import pandas as pd
import random

def generate_uniform_dataset(p, n, binf=-1, bsup=1):
    """ Generate a uniformly distributed dataset.

    Args:
        p (int): Number of dimensions of the description.
        n (int): Number of examples for each class.
        binf (float, optional): Lower bound of the uniform distribution. Default is -1.
        bsup (float, optional): Upper bound of the uniform distribution. Default is 1.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
            - The first array contains the generated dataset with dimensions (n*p, p).
            - The second array contains the labels for the generated dataset.

    Assumptions:
        - n is even.

    The values generated uniformly are within the interval [binf, bsup].
    """
    if not isinstance(n, int):
        raise ValueError("Provide n that is an integer")
    return np.random.uniform(low=binf, high=bsup, size=(n*p, p)), np.asarray([-1 for i in range(0,n)] + [+1 for i in range(0,p*n - n)]) 


def generate_gaussian_dataset(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ Generate a dataset with points following Gaussian distributions.

    Args:
        positive_center (array-like): The mean (center) of the Gaussian distribution for positive class.
        positive_sigma (array-like): The covariance matrix of the Gaussian distribution for positive class.
        negative_center (array-like): The mean (center) of the Gaussian distribution for negative class.
        negative_sigma (array-like): The covariance matrix of the Gaussian distribution for negative class.
        nb_points (int): The number of points to generate for each class.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
            - The first array (data_desc) contains the descriptions of the generated dataset.
            - The second array (data_labels) contains the corresponding labels for the dataset.
    """
    dataset_negative = np.random.multivariate_normal(negative_center, negative_sigma, nb_points)
    dataset_positive = np.random.multivariate_normal(positive_center, positive_sigma, nb_points)
    dataset = np.concatenate((dataset_negative, dataset_positive))
    label = np.array([-1 for i in range(nb_points)] + [1 for i in range(nb_points)])
    return (dataset, label)

def generate_train_test(desc_set, label_set, n_pos, n_neg):
    """ Generate a training and testing dataset split.

    Args:
        desc_set (ndarray): Array containing the descriptions.
        label_set (ndarray): Array containing the corresponding labels.
        n_pos (int): Number of examples with label +1 to include in the training set.
        n_neg (int): Number of examples with label -1 to include in the training set.

    Returns:
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]: A tuple containing two tuples:
            - The first tuple contains the training dataset: descriptions and labels.
            - The second tuple contains the testing dataset: descriptions and labels.

    Assumptions:
        - desc_set and label_set have the same number of rows.
        - n_pos and n_neg, as well as their sum, are less than the total number of examples in desc_set.
    """
    if len(desc_set) != len(label_set):
            raise ValueError("Sets are not in equal length!")
    if n_pos + n_neg > len(desc_set):
        raise ValueError("n_pos and n_neg, as well as their sum, should be less than or equal to the total number of examples in desc_set.")

    length_pos = len(label_set[label_set == 1])
    length_neg = len(label_set[label_set == -1])

    indexes_pos = random.sample([i for i in range(0, length_pos)], n_pos)
    indexes_neg = random.sample([i for i in range(length_pos, length_pos + length_neg)], n_neg)

    desc_train = np.take(desc_set, indexes_pos + indexes_neg, axis=0)
    label_train = np.take(label_set, indexes_pos + indexes_neg, axis=0)

    indexes_test_pos = [i for i in range(0, length_pos) if i not in indexes_pos]
    indexes_test_neg = [i for i in range(length_pos, length_pos + length_neg) if i not in indexes_neg]

    desc_test = np.concatenate((np.take(desc_set, indexes_test_pos, axis=0), np.take(desc_set, indexes_test_neg, axis=0)))
    label_test = np.concatenate((np.take(label_set, indexes_test_pos, axis=0), np.take(label_set, indexes_test_neg, axis=0)))

    return (desc_train, label_train), (desc_test, label_test)


def crossval(X, Y, n_iterations, iteration):
    """ Perform cross-validation by splitting the data into training and testing sets.

    Args:
        X (ndarray): Input features.
        Y (ndarray): Corresponding labels.
        n_iterations (int): Number of iterations for cross-validation.
        iteration (int): Current iteration index.

    Returns:
        tuple: A tuple containing four arrays:
            - X_train: Features for the training set.
            - Y_train: Labels for the training set.
            - X_test: Features for the testing set.
            - Y_test: Labels for the testing set.

    Raises:
        ValueError: If the iteration index exceeds the total number of iterations or if the length of X is not equal to the length of Y.
    """
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length.")

    if iteration >= n_iterations:
        raise ValueError("The iteration index should be less than the total number of iterations.")

    div = len(X) // n_iterations
    start_i = iteration * div
    end_i = (iteration + 1) * div

    X_test = X[start_i:end_i]
    Y_test = Y[start_i:end_i]
    X_train = np.concatenate([X[:start_i], X[end_i:]])
    Y_train = np.concatenate([Y[:start_i], Y[end_i:]])

    return X_train, Y_train, X_test, Y_test


import numpy as np

def crossval_strat(X, Y, n_iterations, iteration):
    """
    Separate the data into training and testing sets for cross-validation, preserving class distribution.

    Args:
        X (ndarray): Input data.
        Y (ndarray): Output labels.
        n_iterations (int): Total number of test sets.
        iteration (int): Current iteration index.

    Returns:
        tuple: A tuple containing four arrays:
            - X_train: Training data.
            - Y_train: Training labels.
            - X_test: Testing data.
            - Y_test: Testing labels.
    """
    classes = np.unique(Y)
    class_indices = {c: np.where(Y == c)[0] for c in classes}

    n_samples_per_class_test = {c: len(indices) // n_iterations for c, indices in class_indices.items()}

    start_idx = {}
    end_idx = {}
    for c, indices in class_indices.items():
        start_idx[c] = iteration * n_samples_per_class_test[c]
        end_idx[c] = (iteration + 1) * n_samples_per_class_test[c]

    X_test = np.concatenate([X[class_indices[c]][start_idx[c]:end_idx[c]] for c in classes])
    Y_test = np.concatenate([Y[class_indices[c]][start_idx[c]:end_idx[c]] for c in classes])

    Xapp = []
    Yapp = []

    for c, indices in class_indices.items():
        if end_idx[c] <= len(indices):
            Xapp.append(X[indices[:start_idx[c]]])
            Xapp.append(X[indices[end_idx[c]:]])
            Yapp.append(Y[indices[:start_idx[c]]])
            Yapp.append(Y[indices[end_idx[c]:]])

    Xapp = np.concatenate(Xapp)
    Yapp = np.concatenate(Yapp)

    return Xapp, Yapp, X_test, Y_test
