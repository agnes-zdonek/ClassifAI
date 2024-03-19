## ClassifAi Modules

## Table of Contents

1. [Classifier](#classifier)
2. [Kernel](#kernel)
3. [Visuals](#visuals)
4. [Tools](#tools)



---

### Classifier
Provides implementations for various classifiers such as KNN, Perceptron, and MultiClassClassifier along with their methods for training, scoring, and prediction.

- **Abstract class to represent a classifier**.
- **Methods**:
  - `train(desc_set, label_set)`: Trains the model on the given dataset.
  - `score(x)`: Returns the prediction score for x.
  - `predict(x)`: Returns the prediction for x (-1 or +1).
  - `accuracy(desc_set, label_set)`: Calculates the accuracy of the system on a given dataset.



### KNNClassifier

- **Class to represent a K Nearest Neighbors classifier**.
- **Methods**:
  - `train(desc_set, label_set)`: Trains the model on the given dataset.
  - `score(x)`: Returns the proportion of +1 among the k nearest neighbors of x.
  - `predict(x)`: Returns the prediction for x (-1 or +1).


### RandomLinearClassifier

- **Class to represent a random linear classifier**.
- **Methods**:
  - `score(x)`: Returns the prediction score for x.
  - `predict(x)`: Returns the prediction for x (-1 or +1).



### PerceptronClassifier

- **Rosenblatt Perceptron**.
- **Methods**:
  - `train(desc_set, label_set, nb_max=100, threshold=0.001)`: Iterative learning of the perceptron on the given dataset.
  - `score(x)`: Returns the prediction score for x.
  - `predict(x)`: Returns the prediction for x (-1 or +1).



### BiaisedPerceptronClassifier

- **Rosenblatt Perceptron with bias**.
- **Methods**:
  - `train_step(desc_set, label_set)`: Performs a single iteration over all examples of the dataset.
  - `score(x)`: Returns the prediction score for x.
  - `predict(x)`: Returns the prediction for x (-1 or +1).



### MultiClassClassifier

- **Multi-class classifier**.
- **Methods**:
  - `train(desc_set, label_set)`: Trains the model on the given dataset.
  - `score(x)`: Returns the prediction score for x.
  - `predict(x)`: Returns the prediction for x.



### ClassifierPerceptronKernel

- **Kernelized Rosenblatt Perceptron**.
- **Methods**:
  - `train_step(desc_set, label_set)`: Perform a single iteration over all examples in the dataset.
  - `score(x)`: Return the prediction score for x.



### ClassifierADALINE

- **ADALINE classifier**.
- **Methods**:
  - `train(desc_set, label_set)`: Train the model on the given dataset.
  - `score(x)`: Return the prediction score for x.
  - `predict(x)`: Return the prediction for x.

---


### Kernel
Contains abstract classes and functions for defining kernel functions, including methods for transformation and dimensionality reduction.

- **Abstract class to represent kernel functions**.
- **Methods**:
  - `get_input_dim()`: Return the dimension of the input space.
  - `get_output_dim()`: Return the dimension of the output space.
  - `transform(V)`: Transform V into the new representation space.



### KernelBias

- **Class for a simple kernel: 2D -> 3D transformation**.
- **Methods**:
  - `transform(V)`: Transform 2D ndarray into 3D ndarray.

---

### Visuals:
Offers functions for visualizing data and classifier decision boundaries, facilitating better understanding and interpretation of the classification process.


#### correlation(x, y)

- **Calculate the Pearson correlation coefficient between two sets of data**.
- **Arguments**:
  - `x (array-like)`: The first set of data.
  - `y (array-like)`: The second set of data. Must have the same length as x.
- **Returns**: The Pearson correlation coefficient.
- **Plots**: Scatter plot of the two sets of data.

#### plot2DSet(desc, labels)

- **Display a 2D dataset**.
- **Arguments**:
  - `desc (ndarray)`: The descriptions of the dataset.
  - `labels (ndarray)`: The labels of the dataset.
- **Plots**: Scatter plot of the dataset, with 'red' color for class -1 and 'blue' color for class +1.

#### plot_decision_boundary(desc_set, label_set, classifier, step=30)

- **Plot the decision boundary associated with the classifier**.
- **Arguments**:
  - `desc_set (ndarray)`: The input dataset descriptions.
  - `label_set (ndarray)`: The corresponding labels of the dataset.
  - `classifier (Classifier)`: The trained classifier to visualize.
  - `step (int, optional)`: The resolution of the plot. Higher values provide a more precise boundary. Default is 30.
- **Plots**: Contour plot of the decision boundary, with 'pink' color for class -1 and 'lightskyblue' color for class +1.

---

### Tools:
Includes a collection of utility functions for dataset generation, dataset splitting, cross-validation, and performance analysis, enhancing the efficiency of machine learning workflows.

#### generate_uniform_dataset(p, n, binf=-1, bsup=1)

- **Generate a uniformly distributed dataset**.
- **Parameters**:
  - `p (int)`: Number of dimensions of the description.
  - `n (int)`: Number of examples for each class.
  - `binf (float, optional)`: Lower bound of the uniform distribution. Default is -1.
  - `bsup (float, optional)`: Upper bound of the uniform distribution. Default is 1.
- **Returns**: A tuple containing two arrays: the generated dataset and the labels.

#### generate_gaussian_dataset(positive_center, positive_sigma, negative_center, negative_sigma, nb_points)

- **Generate a dataset with points following Gaussian distributions**.
- **Parameters**:
  - `positive_center (array-like)`: Mean (center) of the Gaussian distribution for positive class.
  - `positive_sigma (array-like)`: Covariance matrix of the Gaussian distribution for positive class.
  - `negative_center (array-like)`: Mean (center) of the Gaussian distribution for negative class.
  - `negative_sigma (array-like)`: Covariance matrix of the Gaussian distribution for negative class.
  - `nb_points (int)`: Number of points to generate for each class.
- **Returns**: A tuple containing two arrays: the descriptions of the generated dataset and the corresponding labels.

#### generate_train_test(desc_set, label_set, n_pos, n_neg)

- **Generate a training and testing dataset split**.
- **Parameters**:
  - `desc_set (ndarray)`: Array containing the descriptions.
  - `label_set (ndarray)`: Array containing the corresponding labels.
  - `n_pos (int)`: Number of examples with label +1 to include in the training set.
  - `n_neg (int)`: Number of examples with label -1 to include in the training set.
- **Returns**: Two tuples containing the training and testing datasets: descriptions and labels.

#### crossval(X, Y, n_iterations, iteration)

- **Perform cross-validation by splitting the data into training and testing sets**.
- **Parameters**:
  - `X (ndarray)`: Input features.
  - `Y (ndarray)`: Corresponding labels.
  - `n_iterations (int)`: Number of iterations for cross-validation.
  - `iteration (int)`: Current iteration index.
- **Returns**: Four arrays representing the training and testing datasets.

#### crossval_strat(X, Y, n_iterations, iteration)

- **Separate the data into training and testing sets for cross-validation, preserving class distribution**.
- **Parameters**:
  - `X (ndarray)`: Input data.
  - `Y (ndarray)`: Output labels.
  - `n_iterations (int)`: Total number of test sets.
  - `iteration (int)`: Current iteration index.
- **Returns**: Four arrays representing the training and testing datasets.

#### analyze_perfs(L)

- **Calculate the mean and standard deviation of a list of real numbers**.
- **Parameters**:
  - `L (list)`: A non-empty list of real numbers.
- **Returns**: The mean and standard deviation.

#### cross_validation(C, DS, nb_iter)

- **Perform cross-validation**.
- **Parameters**:
  - `C (Classifier)`: The classifier.
  - `DS (tuple)`: A tuple containing the dataset (X, Y).
  - `nb_iter (int)`: Number of iterations for cross-validation.
- **Returns**: A tuple containing the performance scores for each iteration, the mean performance score, and the standard deviation of performance scores.

---


