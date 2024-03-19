# ClassifAI

## Introduction

ClassifAi is a Python library designed to facilitate the understanding and implementation of various machine learning classifiers.
It provides tools for dataset generation, kernel transformations, visualization of datasets, and plotting decision boundaries.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Usage](#usage)
5. [Docs](#docs)

## Features

Dataset Generation: Easily generate synthetic datasets with specified characteristics, such as uniform or Gaussian distributions.\
Classifier Training: Train various machine learning classifiers\
Cross-Validation: Perform cross-validation to evaluate classifier performance and validate models.\
Visualization: Visualize datasets and classifier decisions with intuitive plotting functions.

## Technologies Used

- **Python**: The core programming language for developing the library.
- **NumPy**: Utilized for numerical operations and array manipulations.
- **Matplotlib**: Used for data visualization and plotting functionalities.

## Usage

To use ClassifAI make sure you cloned the repository. Now in a directoy where you cloned it, follow these steps:

1. **Installation**: Install the library using pip:
   ```bash
   pip install ./ClassifAI
   ```

2. **Import Modules**: To use functions, import the required modules and classes from the ClassifAI package:
   ```python
   from ClassifAI.Tools import generate_uniform_dataset
   from ClassifAI.Visuals import plot2DSet
   ```
3. **Examples**:
   ```python
   from ClassifAi.Kernel import KernelBias
   kernel = KernelBias()
   V = np.array([[1, 2], [3, 4], [5, 6]])
   transformed_data = kernel.transform(V)
   ```

   ```python
   import numpy as np
   from ClassifAi.Classifier import MultiClassClassifier
   from ClassifAi.Tools import crossval_strat
   from sklearn.datasets import load_digits

   digits = load_digits()
   X = digits.data
   y = digits.target

   n_iterations = 5  # Number of cross-validation iterations
   for i in range(n_iterations):
     X_train, y_train, X_test, y_test = crossval_strat(X, y, n_iterations, i)
  
     multi_classifier = MultiClassClassifier()
     multi_classifier.train(X_train, y_train)
  
     accuracy = multi_classifier.accuracy(X_test, y_test)
     print(f"Iteration {i + 1} - Accuracy: {accuracy}")
   ```

   ```python
   from ClassifAi.Visuals import plot2DSet, plot_decision_boundary
   plot2DSet(X_uniform, y_uniform)
   from sklearn.svm import SVC
   classifier = SVC(kernel='linear')
   classifier.fit(X_uniform, y_uniform)
   plot_decision_boundary(X_uniform, y_uniform, classifier)
   ```

   ## Docs
   Full documentation can be found in DOCS.md

   
