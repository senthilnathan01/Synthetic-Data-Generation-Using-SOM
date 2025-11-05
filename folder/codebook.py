import numpy as np
import scipy as sp
import matplotlib
from .decorators import timeit

class InvalidNodeIndexError(Exception):
    pass


class InvalidMapsizeError(Exception):
    pass


def generate_hex_lattice(n_rows, n_columns):
    """
     Generate coordinates for a hexagonal lattice.
     Args:
        n_rows (int): The number of rows in the lattice.
        n_columns (int): The number of columns in the lattice.

      Returns:
        numpy.ndarray: An array of (x, y) coordinates representing the hexagonal lattice.
    """
    x_coord = []
    y_coord = []
    for i in range(n_columns):
        for j in range(n_rows):
            x = i + (0.5 if j % 2 != 0 else 0)
            y = j * np.sqrt(0.75)
            x_coord.append(x)
            y_coord.append(y)
    coordinates = np.column_stack([x_coord, y_coord])
    return coordinates


class Codebook(object):

    def __init__(self, mapsize, lattice='hexa'):
        """
            Initializes a Codebook instance.

            Args:
                mapsize (list or int): A list of two integers or a single integer to specify the map size.
                lattice (str): The lattice type, either 'hexa' (hexagonal) or 'rect' (rectangular).

            Raises:
                InvalidMapsizeError: If the mapsize format is invalid.
        """
        self.lattice = lattice

        if 2 == len(mapsize):
            _size = [1, np.max(mapsize)] if 1 == np.min(mapsize) else mapsize

        elif 1 == len(mapsize):
            _size = [1, mapsize[0]]
            print('input was considered as the numbers of nodes')
            print(('map size is [{dlen},{dlen}]'.format(dlen=int(mapsize[0]/2))))
        else:
            raise InvalidMapsizeError(
                "Mapsize is expected to be a 2 element list or a single int")

        self.mapsize = _size
        self.nnodes = mapsize[0]*mapsize[1]
        self.matrix = np.asarray(self.mapsize)
        self.initialized = False

        if lattice == "hexa":
            n_rows, n_columns = mapsize
            coordinates = generate_hex_lattice(n_rows, n_columns)
            self.coordinates=coordinates   # This is coordinates vector: (nnodes x 2)
            # self.lattice_distances: distance between all coordinates with each other
            self.lattice_distances = (sp.spatial.distance_matrix(coordinates, coordinates)   
                                      .reshape(n_rows * n_columns, n_rows, n_columns))

    @timeit()
    def random_initialization(self, data):  # Only for cSOM
        """
        :param data: data to use for the initialization
        :returns: initialized matrix with same dimension as input data
        """
        mn = np.tile(np.min(data, axis=0), (self.nnodes, 1))
        mx = np.tile(np.max(data, axis=0), (self.nnodes, 1))
        self.matrix = mn + (mx-mn)*(np.random.rand(self.nnodes, data.shape[1]))
        self.initialized = True

    @timeit()    
    def pca_linear_initialization(self, data):  # Implemented as per MATLAB; See check_sompy.m; Implemented on 28 Aug 2022

        D = data.copy()      # data = Normalized data
        dim = D.shape[1]
        msize = self.mapsize
        nnodes = self.nnodes
        mdim = len(msize)    # Number of principal components (2 in our case)

        me = np.mean(D, axis=0)   # mean vector; each index contains mean of that column of D
        D = D - me                # Broadcasting will occur here; D is normalized/centered about the mean here

        #### Covariance matrix and eigen value/vector calculation
        ## A: Covariance matrix
        ## V: eigen-vector matrix; each column is eigen vector; size: (dim x mdim)
        ## eigval: eigen-values vector; size: (1 x mdim)
        # np.cov(x, rowvar=False); row is considered as observations and columns as variables in matrix x
        A = np.cov(D, rowvar=False)  # Same as MATLAB A(i,j) loop calculation
        S,V = np.linalg.eig(A)  # S is eigenvalues(array), V is eigenvectors(matrix; each column is eigen vector correspondingly)
        ind = np.argsort(S)[::-1]           # ind = indices of eigenvalues; sorted in descending order hence the [::-1]
        V = V[:,ind]                        # V = eigenvector matrix sorted along columns in order of ind
        eigval = S[ind]                     # eigval = Sorted according to ind
        V = V[:,:mdim]                      # V is now matrix eigen-vectors of only highest two eigen values
        eigval = eigval[:mdim]              # eigval is highest two eigen-values
        norm_V = np.linalg.norm(V, axis=0)  # norm of vector V
        V = (V/norm_V)*np.sqrt(eigval)      # V = normalized to unit length; and scaled to sqrt(eigenval)

        #### Codebook Initialization
        initialized_codebook = np.tile(me,(nnodes,1))  # Each row is the mean vector me; size: nnodes x dim
        Coords = self.coordinates.copy()               # Coordinates in topological space; size: nnodes x mdim
        Coords = np.flip(Coords, axis=1)               # Swap the columns
        ma = np.max(Coords, axis=0)
        mi = np.min(Coords, axis=0)

        # Scaling of Coordinates (As per MATLAB modifiedsom_lininit1.m)
        Coords = ((Coords-mi)/(ma-mi))*2.6
        Coords = (Coords - 1.3)*2

        # initialize codebook matrix: multiple of both eigen vectors are added to mean vector (i.e. each row)
        # here the multiple is coordinates of som topology which were scaled centrally about origin i.e. Coords
        for n in range(nnodes):
            for d in range(mdim):
                initialized_codebook[n,:] = initialized_codebook[n,:] + Coords[n,d]*V[:,d] 

        self.matrix = np.around(initialized_codebook, decimals=6)
        self.initialized = True

    def grid_dist(self, node_ind):
        """
        Calculates grid distance based on the lattice type.

        :param node_ind: number between 0 and number of nodes-1. Depending on
                         the map size, starting from top left
        :returns: matrix representing the distance matrix
        """

        if self.lattice == 'hexa':
            return self._hexa_dist(node_ind)

    def _hexa_dist(self, node_ind):
        return self.lattice_distances[node_ind]
