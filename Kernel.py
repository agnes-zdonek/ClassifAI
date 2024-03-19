# -*- coding: utf-8 -*-
"""
Package: ClassifAi
File: Kernel.py
Version: Feburary 2024
Read README.md for usage instructions
"""
import numpy as np

class Kernel():
    """ Abstract class to represent kernel functions.
    """
    def __init__(self, dim_in, dim_out):
        """ Constructor for Kernel.

        Args:
            dim_in (int): Dimension of the input space (kernel input).
            dim_out (int): Dimension of the output space (kernel output).
        """
        if dim_in < 0 or dim_out < 0: 
            raise ValueError
        
        if not isinstance(dim_out, int):
            raise ValueError("Provide dim_out that is an integer")
        
        if not isinstance(dim_in, int):
            raise ValueError("Provide dim_in that is an integer")
        
        self.input_dim = dim_in
        self.output_dim = dim_out
        
    def get_input_dim(self):
        """ Return the dimension of the input space.
        """
        return self.input_dim

    def get_output_dim(self):
        """ Return the dimension of the output space.
        """
        return self.output_dim
    
    def transform(self, V):
        """ Transform V into the new representation space.

        Args:
            V (ndarray): Input data to be transformed.

        Returns:
            ndarray: Transformed data in the new representation space.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Please Implement this method")


class KernelBias(Kernel):
    """ Class for a simple kernel: 2D -> 3D transformation.
    """
    def __init__(self):
        """ Constructor for KernelBias.
            No arguments, dimensions are fixed.
        """
        super().__init__(2, 3)
        
    def transform(self, V):
        """ Transform 2D ndarray into 3D ndarray.
        
        Args:
            V (ndarray): 2D array to be transformed.
            
        Returns:
            ndarray: 3D array with an additional dimension added.
        """
        if V.ndim == 1:  # Check if it's a vector or a matrix
            W = np.array([V])  # Convert to a matrix
            V_proj = np.append(W, np.ones((len(W), 1)), axis=1)
            V_proj = V_proj[0]  # Return something of the same dimension
        else:
            V_proj = np.append(V, np.ones((len(V), 1)), axis=1)
            
        return V_proj

