# -*- coding: utf-8 -*-
"""
Package: ClassifAi
File: Visuals.py
Version: Feburary 2024
Read README.md for usage instructions
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline  


def correlation(x, y):
    """
    Calculate the Pearson correlation coefficient between two sets of data.
    Arguments:
    x (array-like): The first set of data.
    y (array-like): The second set of data. Must have the same length as x.
    """
    if len(x) != len(y):
        raise ValueError("Sets are not in equal length!")   
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    cov_xy = np.sum((x - mean_x) * (y - mean_y))
    std_x = np.sqrt(np.sum((x - mean_x)**2))
    std_y = np.sqrt(np.sum((y - mean_y)**2))

    correlation = cov_xy / (std_x * std_y)

    plt.figure(figsize=(12,8))
    plt.scatter(x, y, alpha=1)

    plt.xlabel('First df')
    plt.ylabel('Second df')

    plt.show()

    return correlation


def plot2DSet(desc, labels):
    """ Display a 2D dataset.

    Args:
        desc (ndarray): The descriptions of the dataset.
        labels (ndarray): The labels of the dataset.

    The function should use the color 'red' for the class -1 and 'blue' for the +1 class.
    """
    desc_neg = desc[labels == -1]
    desc_pos = desc[labels == 1]
    
    plt.figure(figsize=(12,8))
    plt.scatter(desc_neg[:,0], desc_neg[:,1], marker='o', color="red") # 'o' red for class -1
    plt.scatter(desc_pos[:,0], desc_pos[:,1], marker='x', color="blue") # 'x' blue for class +1
    plt.show()
