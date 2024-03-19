# -*- coding: utf-8 -*-
"""
Package: ClassifAi
File: Tools.py
Version: Feburary 2024
Read README.md for usage instructions
"""
import numpy as np
import pandas as pd

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
