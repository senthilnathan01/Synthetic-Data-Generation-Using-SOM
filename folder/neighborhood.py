import numpy as np
import inspect
import sys

small = .000000000001


class NeighborhoodFactory(object):

    @staticmethod
    def build(neighborhood_func):
        """
               Factory method to create a neighborhood function instance.

               Args:
                   neighborhood_func (str): The name of the desired neighborhood function.

               Returns:
                   object: An instance of the specified neighborhood function.

               Raises:
                   Exception: If the specified neighborhood function is unsupported.
        """
        neighborhood_classes = [GaussianNeighborhood, BubbleNeighborhood]
        for neighborhood_class in neighborhood_classes:
            if neighborhood_func == neighborhood_class.name:
                return neighborhood_class()
        raise Exception(f"Unsupported neighborhood function '{neighborhood_func}'")


class GaussianNeighborhood(object):

    name = 'gaussian'

    @staticmethod
    def calculate(distance_matrix, radius, dim):  # This returns h_ci(t) as in SOM algorithm

        return np.exp(-1.0*(distance_matrix**2)/(2.0*radius**2)).reshape(dim, dim)    # as per MATLAB; 28 Aug 2022

    def __call__(self, *args, **kwargs):
        return self.calculate(*args)


class BubbleNeighborhood(object):

    name = 'bubble'

    @staticmethod
    def calculate(distance_matrix, radius, dim):
        def l(a, b):
            c = np.zeros(b.shape)
            c[a-b >= 0] = 1
            return c

        return l(radius,
                 np.sqrt(distance_matrix.flatten())).reshape(dim, dim) + small

    def __call__(self, *args, **kwargs):
        return self.calculate(*args)
