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
