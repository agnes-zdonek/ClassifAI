# -*- coding: utf-8 -*-
"""
Package: ClassifAi
File: Classifiers.py
Version: Feburary 2024
Read README.md for usage instructions
"""
import numpy as np
import copy

class Classifier:
    """ Abstract class to represent a classifier
        Note: This class should not be instantiated directly.
    """
    
    def __init__(self, input_dimension):
        """ Constructor for Classifier
            Argument:
                - input_dimension (int): dimension of example descriptions
            Assumption: input_dimension > 0
        """
        if input_dimension > 0:
            raise ValueError
        self.input_dimension = input_dimension
        
    def train(self, desc_set, label_set):
        """ Trains the model on the given dataset
            desc_set: ndarray with descriptions
            label_set: ndarray with corresponding labels
            Assumption: desc_set and label_set have the same number of rows
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self, x):
        """ Returns the prediction score for x
            x: a description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ Returns the prediction for x (-1 or +1)
            x: a description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Calculates the accuracy of the system on a given dataset
            desc_set: ndarray with descriptions
            label_set: ndarray with corresponding labels
            Assumption: desc_set and label_set have the same number of rows
        """
        if len(desc_set) != len(label_set):
            raise ValueError("Sets are not in equal length!")
        correct_predictions = sum(1 for x, y in zip(desc_set, label_set) if self.predict(x) == y)
        return correct_predictions / len(desc_set)
    
class KNNClassifier(Classifier):
    """ Class to represent a K Nearest Neighbors classifier.
    """

    def __init__(self, input_dimension, k):
        """ Constructor for Classifier
            Argument:
                - input_dimension (int): input dimension of examples
                - k (int): number of neighbors to consider
            Assumption: input_dimension > 0
        """
        super().__init__(input_dimension)
        self.k = k
        self.dataset = None
        
    def score(self, x):
        """ Returns the proportion of +1 among the k nearest neighbors of x (real value)
            x: a description: an ndarray
        """
        X_train, Y_train = self.dataset
        distances = np.linalg.norm(X_train - x, axis=1)
        sorted_indices = np.argsort(distances)
        k_neighbors = Y_train[sorted_indices[:self.k]]

        proportion_positive = np.sum(k_neighbors == 1) / self.k
        score = 2 * (proportion_positive - 0.5)

        return score
    
    def predict(self, x):
        """ Returns the prediction for x (-1 or +1)
            x: a description: an ndarray
        """
        score = self.score(x)
        return 1 if score > 0 else -1

    def train(self, desc_set, label_set):
        """ Trains the model on the given dataset
            desc_set: ndarray with descriptions
            label_set: ndarray with corresponding labels
            Assumption: desc_set and label_set have the same number of rows
        """       
        if len(desc_set) != len(label_set):
            raise ValueError("Sets are not in equal length!") 
        self.dataset = (desc_set, label_set)
        
class RandomLinearClassifier(Classifier):
    """ Class to represent a random linear classifier
        This class inherits from the Classifier class
    """
    
    def __init__(self, input_dimension):
        """ Constructor for Classifier
            Argument:
                - input_dimension (int): dimension of example descriptions
            Assumption: input_dimension > 0
        """
        super().__init__(input_dimension)
        
        vect = np.random.uniform(-1, 1, input_dimension)
        self.w = vect / np.linalg.norm(vect)
        
          
    def train(self, desc_set, label_set):
        """ Trains the model on the given dataset
            desc_set: ndarray with descriptions
            label_set: ndarray with corresponding labels
            Assumption: desc_set and label_set have the same number of rows
        """        
        print("No training for this classifier!")
    
    def score(self, x):
        """ Returns the prediction score for x (real value)
            x: a description
        """
        return np.dot(x, self.w)
    
    def predict(self, x):
        """ Returns the prediction for x (-1 or +1)
            x: a description
        """
        score = self.score(x)
        return 1 if score > 0 else -1
    
class PerceptronClassifier(Classifier):
    """ Rosenblatt Perceptron
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True ):
        """ Constructor for Classifier
            Argument:
                - input_dimension (int): dimension of example descriptions (>0)
                - learning_rate (default 0.01): epsilon
                - init is the mode of w initialization:
                    - if True (default): initialization to 0 of w
                    - if False: initialization by random drawing of small values
        """
        super().__init__(input_dimension)
        self.learning_rate = learning_rate
        if init:
            self.w = np.zeros(input_dimension)
        else:
            v = np.random.uniform(0, 1, input_dimension)
            v = (2*v -1) * 0.001
            self.w = v.copy()
        self.allw = [self.w.copy()]
        
    def train_step(self, desc_set, label_set):
        """ Performs a single iteration over all examples of the dataset
            given by randomly taking the examples.
            Arguments:
                - desc_set: ndarray with descriptions
                - label_set: ndarray with corresponding labels
        """
        if len(desc_set) != len(label_set):
            raise ValueError("Sets are not in equal length!")       
        indices = np.arange(len(desc_set))
        np.random.shuffle(indices)
        for i in indices:
            x_i = desc_set[i]
            y_i = label_set[i]
            y_pred = np.sign(np.dot(x_i, self.w))
            if y_pred != y_i:
                self.w += self.learning_rate * y_i * x_i
                self.allw.append(self.w.copy())
     
    def train(self, desc_set, label_set, nb_max=100, threshold=0.001):
        """ Iterative learning of the perceptron on the given dataset.
            Arguments:
                - desc_set: ndarray with descriptions
                - label_set: ndarray with corresponding labels
                - nb_max (default: 100): maximum number of iterations
                - threshold (default: 0.001): convergence threshold
            Returns: a list of norm difference values
        """
        if len(desc_set) != len(label_set):
            raise ValueError("Sets are not in equal length!")            
        norm_differences = []
        for i in range(nb_max):
            old_w = self.w.copy()
            self.train_step(desc_set, label_set)
            norm_difference = np.linalg.norm(np.abs(old_w - self.w))
            norm_differences.append(norm_difference)
            if norm_difference < threshold: 
                break
        return norm_differences
    
    def score(self, x):
        """ Returns the prediction score for x (real value)
            x: a description
        """
        return np.dot(x, self.w)
    
    def predict(self, x):
        """ Returns the prediction for x (-1 or +1)
            x: a description
        """
        return np.sign(self.score(x))
    
    def get_allw(self):
        return self.allw

class BiaisedPerceptronClassifier(PerceptronClassifier):
    """ Rosenblatt Perceptron with bias
        Variant of the basic perceptron
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructor for Classifier
            Argument:
                - input_dimension (int): dimension of example descriptions (>0)
                - learning_rate (default 0.01): epsilon
                - init is the mode of w initialization:
                    - if True (default): initialization to 0 of w
                    - if False: initialization by random drawing of small values
        """
        super().__init__(input_dimension, learning_rate, init)
        
    def train_step(self, desc_set, label_set):
        """ Performs a single iteration over all examples of the dataset
            given by randomly taking the examples.
            Arguments:
                - desc_set: ndarray with descriptions
                - label_set: ndarray with corresponding labels
        """
        if len(desc_set) != len(label_set):
            raise ValueError("Sets are not in equal length!")      
        indices = np.arange(len(desc_set))
        np.random.shuffle(indices)     
        for i in indices:
            error = self.score(desc_set[i]) * label_set[i]
            if error < 1:
                self.w = self.w + self.learning_rate * (label_set[i] - self.score(desc_set[i])) * desc_set[i]
                self.allw.append(self.w.copy())    
                
class MultiClassClassifier(Classifier):
    """ Multi-class classifier
    """
    def __init__(self, binary_classifier):
        """ Constructor for Classifier
            Argument:
                - binary_classifier: positive/negative binary classifier
        """
        self.binary_classifier = binary_classifier
        self.classifiers = []
        
    def train(self, desc_set, label_set):
        """ Trains the model on the given dataset
            Iterates over the data randomly
            desc_set: ndarray with descriptions
            label_set: ndarray with corresponding labels
            Assumption: desc_set and label_set have the same number of rows
        """
        if len(desc_set) != len(label_set):
            raise ValueError("Sets are not in equal length!")    
        unique_classes = np.unique(label_set)
        
        for c in unique_classes:
            cl = copy.deepcopy(self.binary_classifier)
            y_tmp = np.where(label_set == c, 1, -1)
            cl.train(desc_set, y_tmp)
            self.classifiers.append(cl)
    
    def score(self,x):
        """ Returns the prediction score for x (real value)
            x: a description
        """
        scores = [cl.score(x) for cl in self.classifiers]
        return scores
    
    def predict(self, x):
        """ Returns the prediction for x (argmax of scores)
            x: a description
        """
        scores = self.score(x)
        return np.argmax(scores)
    
    def accuracy(self, desc_set, label_set):
        if len(desc_set) != len(label_set):
            raise ValueError("Sets are not in equal length!")
        yhat = np.array([self.predict(x) for x in desc_set])
        return np.where(label_set == yhat, 1., 0.).mean()


class ClassifierPerceptronKernel(PerceptronClassifier):
    """ Kernelized Rosenblatt Perceptron.
    """
    def __init__(self, input_dimension, learning_rate, noyau, init=0):
        """ Constructor for ClassifierPerceptronKernel.

        Args:
            input_dimension (int): Dimension of the input space (original space).
            learning_rate (float): Learning rate (epsilon).
            noyau (Kernel): Kernel to use.
            init (int, optional): Initialization mode of w:
                - if 0 (default): w initialized to 0,
                - if 1: w initialized by randomly drawing small values.

        Raises:
            ValueError: If the input_dimension is not positive or the learning_rate is not positive.

        """
        if input_dimension <= 0:
            raise ValueError("Input dimension must be positive.")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        
        self.__init__(input_dimension, learning_rate)
        self.noyau = noyau
        if init == 0:
            self.w = np.zeros(self.noyau.get_output_dim())
        else:
            self.w = np.random.uniform(0, 1, self.noyau.get_output_dim())
            lst = []
            for i in self.w:
                lst.append((2 * i - 1) * 0.001)
            self.w = np.array(lst) 
        
    def train_step(self, desc_set, label_set):
        """ Perform a single iteration over all examples in the dataset,
            taking examples randomly.

        Args:
            desc_set (ndarray): Array containing descriptions.
            label_set (ndarray): Array containing corresponding labels.

        """
        if len(desc_set) != len(label_set):
            raise ValueError("The number of descriptions must be equal to the number of labels.")
        
        num_examples = desc_set.shape[0]
        indices = np.arange(num_examples)
        np.random.shuffle(indices)
        for idx in indices:
            x = desc_set[idx]
            y = label_set[idx]
            x_k = self.noyau.transform(x)
            if y * np.dot(self.w, x_k) <= 0:
                self.w += self.learning_rate * y * x_k
     
    def score(self, x):
        """ Return the prediction score for x.

        Args:
            x (ndarray): A description in the original space.

        Returns:
            float: The prediction score.

        Raises:
            ValueError: If the input dimension of x does not match the expected input dimension.
        """
        if len(x) != self.input_dimension:
            raise ValueError("Input dimension of x does not match the expected input dimension.")
        
        x_k = self.noyau.transform(x)
        return np.dot(self.w, x_k)

import numpy as np

class ClassifierADALINE(Classifier):
    """ ADALINE classifier.
    """
    def __init__(self, input_dimension, learning_rate, history=False, niter_max=1000):
        """ Constructor for ClassifierADALINE.

        Args:
            input_dimension (int): Dimension of the input space (description).
            learning_rate (float): Learning rate (epsilon).
            history (bool, optional): Whether to store the weights during training.
            niter_max (int, optional): Maximum number of iterations.

        Raises:
            ValueError: If the input_dimension is not positive or the learning_rate is not positive.

        """
        if input_dimension <= 0:
            raise ValueError("Input dimension must be positive.")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")

        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.history = history
        self.niter_max = niter_max
        self.allw = []
        self.w = (2*(np.random.rand(self.input_dimension)) - 1) * 0.001
        
    def train(self, desc_set, label_set):
        """ Train the model on the given dataset.

        Args:
            desc_set (ndarray): Array containing descriptions.
            label_set (ndarray): Array containing corresponding labels.

        Raises:
            ValueError: If the number of descriptions does not match the number of labels.

        """
        if len(desc_set) != len(label_set):
            raise ValueError("The number of descriptions must be equal to the number of labels.")
        
        n_samples = desc_set.shape[0]
        iteration = 0
        while iteration < self.niter_max:
            i = np.random.randint(0, n_samples)
            xi = desc_set[i]
            yi = label_set[i]
            gradient = xi * (np.dot(xi, self.w) - yi)
            self.w -= self.learning_rate * gradient
            if self.history:
                self.allw.append(self.w.copy())
            iteration += 1
    
    def score(self, x):
        """ Return the prediction score for x.

        Args:
            x (ndarray): A description.

        Returns:
            float: The prediction score.

        Raises:
            ValueError: If the input dimension of x does not match the expected input dimension.

        """
        if len(x) != self.input_dimension:
            raise ValueError("Input dimension of x does not match the expected input dimension.")
        
        return np.dot(x, self.w)
    
    def predict(self, x):
        """ Return the prediction for x.

        Args:
            x (ndarray): A description.

        Returns:
            int: The predicted label (+1 or -1).

        """
        return 1 if self.score(x) >= 0 else -1
