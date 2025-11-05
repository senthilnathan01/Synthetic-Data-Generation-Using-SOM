# Modified Version: Dr. Palaniappan Ramu (palramu@iitm.ac.in)
#                   Associate Professor, Department of Engineering Design, Indian Institute of Technology Madras(IIT Madras),
#                   Chennai, India
#                   Website: https://ed.iitm.ac.in/~palramu/team.html


import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist


class aux_fun():
    """
        Class for auxiliary functions, implemented similarly to MATLAB.
        Date: 12 Sept 22
    """

    # 'self' is not utilised in any methods here.
    def __init__(self):
        self.used = True

    # In all methods here, 'som' is an object of SOM class from som_isom or som_csom file

    def som_bmus(self, som, data, distance_metric='euclidean'):
        """
        Find the Best Matching Units (BMUs) in a Self-Organizing Map for given data.

        Args:
            som: A trained SOM object
            data (ndarray): Data samples, where each row is a sample.
            distance_metric (str, optional): The distance metric to use for BMU calculation.

        Returns:
            ndarray: An array of BMU indices for each data point.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a NumPy ndarray.")

        if data.ndim == 1:
            data = data.reshape(1, len(data))

        dist_array = cdist(data, som.codebook.matrix, distance_metric)
        arg_min = np.argmin(dist_array, axis=1)
        return arg_min
    
    def som_bmus2(self, som, data):
        bmu = som.find_bmu(data)
        size=som.codebook.matrix.shape
        all_values = np.arange(size[0])
        unique_values, counts = np.unique(bmu[0], return_counts=True)
        all_counts = np.zeros_like(all_values)
        all_counts[unique_values.astype(int)] = counts
        hits= all_counts

        return hits

    def quantization_error(self, som, data, distance_metric='euclidean'):
        """
        Calculate the quantization error of a Self-Organizing Map (SOM) with respect to given data.

        Args:
            som (sompy.sompy.SOM): A trained Self-Organizing Map.
            data (ndarray): Data samples, where each row is a sample.
            distance_metric (str, optional): The distance metric to use for BMU calculation.

        Returns:
            float: The quantization error.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a NumPy ndarray.")

        if data.ndim == 1:
            data = data.reshape(1, len(data))

        dist_array = cdist(data, som.codebook.matrix, distance_metric)
        min_dist = np.min(dist_array, axis=1)
        qe = np.mean(min_dist)

        return qe

    def topographic_error(self, som, distance_metric='euclidean'):
        """
           Calculate the topographic error of a Self-Organizing Map (SOM).
           In simple terms it assesses how well neighboring data points in the input space are also neighbors in the SOM grid.

           Args:
               som : A trained SOM object

           Returns:
               float: The topographic error.
           """
        # bmu, i.e. nearest 1st node of each data. bmu1 is a vector or size (1 x datalength)
        bmu1 = self.som_bmus(som, som._data)

        dist_array = cdist(som._data, som.codebook.matrix, distance_metric)  # for cSOM bmu
        # 2nd nearest node of each data. bmu2 is a vector or size (1 x datalength)
        # # sorts each row of dist_array in ascending order
        bmu2 = np.argsort(dist_array, axis=1)[:, 1]
        # 'ne1' represents the neighbors of each SOM unit
        ne1 = self.som_unit_neighs(som)

        # TGE is calculated as the mean of comparisons between BMU and second-best matching units
        # to determine if they are not immediate neighbors (not equal to 1
        tge = np.mean(ne1[bmu1, bmu2] != 1)
        return tge

    def som_unit_neighs(self, som):  # Can be also used just before sm.train
        """
           Calculate the 1-neighborhood matrix for a Self-Organizing Map (SOM).

           Args:
               som (sompy.sompy.SOM): A trained Self-Organizing Map.

           Returns:
               ndarray: The 1-neighborhood matrix where each row represents a SOM unit,
                        and a 1 in the matrix indicates units that are in the 1-neighborhood radius.
           """

        # Ne1 Neighborhood sparse matrix; e.g. row 1 will have 1 at 2 and 4 for 3x3 map
        # Retrieve the distance matrix from the SOM
        dist_mat = som._distance_matrix  # Similar to som_unit_dist matrix in MATLAB
        nnodes = som.codebook.nnodes

        # Initialize the neighborhood matrix
        ne1 = csr_matrix((nnodes, nnodes)).toarray()
        for i in range(nnodes):
            # Find the indices of nodes within the 1-neighborhood of unit i
            ind = np.where(np.logical_and(dist_mat[i, :] > 0, dist_mat[i, :] < 1.01))
            # Mark these nodes as part of the 1-neighborhood of unit i
            ne1[i, ind] = 1

        # Use this line to find indices which are in 1-Neighborhood radius of Node 0
        # np.where(som_unit_neighs(sm)[0,:]==1)
        return ne1

    def som_divide(self, som, data, inds):

        """
            Divide data samples into lists based on their Best Matching Units (BMUs) in a Self-Organizing Map (SOM).

            Args:
                som : Trained SOM object
                data (ndarray): Data samples where each row is a sample.
                inds (int or ndarray): int or ndarray of map index number

            Returns:
                list of ndarray: A list of arrays containing data indices mapped to the specified node(s).
            """
        list_id = []
        # Ensure that 'inds' is a list of indices
        if isinstance(inds, int):
            inds = [inds]

        # Calculate BMUs for the data
        bmus = self.som_bmus(som, data)
        for node_index in inds:
            # Create a boolean mask indicating which data samples map to the specified node
            # returns True at positions in sm._bmu[0] where inds[i] match with them
            bool_id = (bmus == node_index)
            # Find the indices of data samples that match the node
            matching_indices = np.argwhere(bool_id).T
            # Unpack and append the list of matching indices to the result
            list_id.append(*matching_indices)
        return list_id

    def som_divide(self, som, data, inds):  # returns list of ndarray
        # Will work even if data is not training data but test data i.e. other than som._data
        # data(ndarray); data samples; each row is a sample, each column is a dim [x1,x2,..,xn,fx]
        # data(ndarray): Should be variance/range normalized
        # inds: ndarray of map index number
        list_id = []
        if type(inds) == int:
            # bool_id = np.isin(som._bmu[0],inds)
            bmus = self.som_bmus(som, data)
            bool_id = np.isin(bmus, inds)  # returns True at positions in sm._bmu[0] where inds match with them
            ids_i = np.argwhere(bool_id).T  # returns indices at 'True' position
            list_id = list(*ids_i)  # list_id: list; each position i in list is data indices mapped to node inds
        else:
            for i in range(inds.size):
                # bool_id = np.isin(som._bmu[0],inds)
                bmus = self.som_bmus(som, data)
                bool_id = np.isin(bmus,
                                  inds[i])  # returns True at positions in sm._bmu[0] where inds[i] match with them
                ids_i = np.argwhere(bool_id).T  # returns indices at True
                # list_id: list of ndarray; each position i in list is ndarray of data indices mapped to node inds[i]
                list_id.append(*ids_i)
        return list_id

    def som_hits(self, som, data):
        """
        Calculate the number of hits for each map unit in a Self-Organizing Map (SOM).

        Args:
            som:  A trained SOm object
            data (ndarray): Data samples, where each row is a sample.

        Returns:
            ndarray: An array containing the number of hits for each map unit.
        """

        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a NumPy ndarray.")

        map_inds = np.arange(som.codebook.nnodes)  # Get all map indices of the trained map
        list_id = self.som_divide(som, data, map_inds)  # Get a list of ndarray of data indices mapped to each map node

        # Calculate the number of hits for each map unit using list comprehension
        hits = np.array([len(mapping) for mapping in list_id])
        return hits



    def som_umat(self, som, comp='all', cond='median'):
        """
           Calculate the U-matrix (Unified Distance Matrix) for a Self-Organizing Map (SOM).

           Args:
               som: A trained SOM object.
               comp (str or list, optional): Components to consider in distance calculation. 'all' or a list of indices.
               cond (str, optional): The condition for calculating U-matrix values ('mean', 'median', or 'mode').

           Returns:
               ndarray: The U-matrix for the SOM.
           """

        y, x = som.codebook.mapsize
        t = True
        # These if-else statements are for handling component while calculating distance for U-matrix, e.g. DD matrix
        if comp == 'all':
            codebook = som.codebook.matrix.copy()
            M = np.array([codebook[:, i].reshape((x, y)).T for i in range(codebook.shape[1])])
        else:
            # comp = [0,1] or list of indexes
            if isinstance(comp, list):
                codebook = som.codebook.matrix[:, comp].copy()
                M = np.array([codebook[:, i].reshape((x, y)).T for i in range(codebook.shape[1])])
            else:
                t = False
                codebook = som.codebook.matrix[:, comp].copy()
                M = codebook.reshape(y, x)

        # dim = codebook.shape[1]
        # M = codebook.copy()
        ux = 2 * x - 1
        uy = 2 * y - 1
        # M = np.array([M[:,i].reshape((x,y)).T for i in range(M.shape[1])])
        U = np.zeros((uy, ux))

        ## Loop for node distance values
        if t:  # if type(comp) is a list of indices
            for j in range(y):
                for i in range(x):
                    if i < x - 1:
                        dx = (M[:, j, i] - M[:, j, i + 1]) ** 2
                        U[2 * j, 2 * i + 1] = np.sqrt(sum(dx))
                    if j < y - 1:
                        dy = (M[:, j, i] - M[:, j + 1, i]) ** 2
                        U[2 * j + 1, 2 * i] = np.sqrt(sum(dy))

                        if j % 2 == 1 and i < x - 1:
                            dz = (M[:, j, i] - M[:, j + 1, i + 1]) ** 2
                            U[2 * j + 1, 2 * i + 1] = np.sqrt(sum(dz))
                        elif j % 2 == 0 and i > 0:
                            dz = (M[:, j, i] - M[:, j + 1, i - 1]) ** 2
                            U[2 * j + 1, 2 * i - 1] = np.sqrt(sum(dz))
        else:  # if comp is not a list but a number
            for j in range(y):
                for i in range(x):
                    if i < x - 1:
                        dx = (M[j, i] - M[j, i + 1]) ** 2
                        U[
                            2 * j, 2 * i + 1] = dx  # np.sqrt(sum(dx)); This commented part gave an error:'numpy.float64' object is not iterable
                    if j < y - 1:
                        dy = (M[j, i] - M[j + 1, i]) ** 2
                        U[2 * j + 1, 2 * i] = dy  # np.sqrt(sum(dy))

                        if j % 2 == 1 and i < x - 1:
                            dz = (M[j, i] - M[j + 1, i + 1]) ** 2
                            U[2 * j + 1, 2 * i + 1] = dz  # np.sqrt(sum(dz))
                        elif j % 2 == 0 and i > 0:
                            dz = (M[j, i] - M[j + 1, i - 1]) ** 2
                            U[2 * j + 1, 2 * i - 1] = dz  # np.sqrt(sum(dz))

        # Loop for node values
        a = []
        for j in range(0, uy, 2):
            for i in range(0, ux, 2):

                if i == 0 and j == 0:  # Top left corner (0,0)
                    a = [U[j, i + 1], U[j + 1, i]]
                elif j == 0 and i > 0 and i < ux - 1:  # Upper edge (0,2; 0,4; 0,6)
                    a = [U[j, i - 1], U[j, i + 1], U[j + 1, i - 1], U[j + 1, i]]
                elif j == 0 and i == ux - 1:  # Top right corner (0,8)
                    a = [U[j, i - 1], U[j + 1, i - 1], U[j + 1, i]]
                elif i == 0 and j > 0 and j < uy - 1:  # Left edge (2,0; 4,0; 6,0)
                    a = [U[j, i + 1]]
                    if j % 4 == 0:
                        a = [*a, U[j - 1, i], U[j + 1, i]]
                    else:
                        a = [*a, U[j - 1, i], U[j - 1, i + 1], U[j + 1, i], U[j + 1, i + 1]]
                elif i > 0 and j > 0 and i < ux - 1 and j < uy - 1:  # Middle part of the map
                    a = [U[j, i - 1], U[j, i + 1]]
                    if j % 4 == 0:
                        a = [*a, U[j - 1, i - 1], U[j - 1, i], U[j + 1, i - 1], U[j + 1, i]]
                    else:
                        a = [*a, U[j - 1, i], U[j - 1, i + 1], U[j + 1, i], U[j + 1, i + 1]]
                elif i == 0 and j == uy - 1:  # Bottom left corner (10,0)
                    if j % 4 == 0:
                        a = [U[j, i + 1], U[j - 1, i]]
                    else:
                        a = [U[j, i + 1], U[j - 1, i], U[j - 1, i + 1]]
                elif j == uy - 1 and i > 0 and i < ux - 1:  # Lower edge
                    a = [U[j, i - 1], U[j, i + 1]]
                    if j % 4 == 0:
                        a = [*a, U[j - 1, i - 1], U[j - 1, i]]
                    else:
                        a = [*a, U[j - 1, i], U[j - 1, i + 1]]
                elif i == ux - 1 and j == uy - 1:  # Bottom right corner (10,8)
                    if j % 4 == 0:
                        a = [U[j, i - 1], U[j - 1, i], U[j - 1, i - 1]]
                    else:
                        a = [U[j, i - 1], U[j - 1, i]]
                elif i == ux - 1 and j > 0 and j < uy - 1:  # Right edge
                    a = [U[j, i - 1]]
                    if j % 4 == 0:
                        a = [*a, U[j - 1, i], U[j - 1, i - 1], U[j + 1, i], U[j + 1, i - 1]]
                    else:
                        a = [*a, U[j - 1, i], U[j + 1, i]]
                else:
                    a = 0

                if cond == 'mean':
                    U[j, i] = eval(f'np.mean({a})')
                elif cond == 'mode':
                    from scipy import stats
                    U[j, i] = eval(f'stats.mode({a})')
                else:
                    U[j, i] = eval(f'np.median({a})')

        return U  # ndarray matrix of 11x9; 1st node is at (0,0),(0,2)...(10,6),(10,8)
