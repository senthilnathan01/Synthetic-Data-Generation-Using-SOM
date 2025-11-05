# -*- coding: utf-8 -*-

# Author: Vahid Moosavi (sevamoo@gmail.com)
#         Chair For Computer Aided Architectural Design, ETH  Zurich
#         Future Cities Lab
#         www.vahidmoosavi.com

# Contributor: Sebastian Packmann (sebastian.packmann@gmail.com)

# Modified Version: Dr. Palaniappan Ramu (palramu@iitm.ac.in)
#                   Associate Professor, Department of Engineering Design, Indian Institute of Technology Madras(IIT Madras),
#                   Chennai, India
#                   Website: https://ed.iitm.ac.in/~palramu/team.html


import tempfile
import os
import logging

import numpy as np

from time import time
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist    # Implemented on 28 Aug 2022 (required for find_bmu() method) on iSOM
from joblib import load, dump

from .decorators import timeit
from .codebook import Codebook
from .neighborhood import NeighborhoodFactory
from .normalization import NormalizerFactory


class ComponentNamesError(Exception):
    pass


class LabelsError(Exception):
    pass


class SOMFactory(object):

    @staticmethod
    def build(data,
              mapsize=None,
              mask=None,                  # default mask: None
              mapshape='planar',          # default mapshape: planar
              lattice='hexa',             # default lattice shape: hexa
              normalization='range',      # default normalization: range
              initialization='pca',       # default initialization: pca
              neighborhood='gaussian',    # Only gaussian possible to keep (lets see this 16 Sept)
              training='batch',
              name='sompy_csom',
              component_names=None):
        """
        :param data: data to be clustered, represented as a matrix of n rows,
            as inputs and m cols as input features
        :param neighborhood: neighborhood object calculator.  Options are:
            - gaussian
            - bubble
            - manhattan (not implemented yet)
            - cut_gaussian (not implemented yet)
            - epanechicov (not implemented yet)

        :param normalization: normalizer object calculator. Options are:
            - var

        :param mapsize: tuple/list defining the dimensions of the som.
            If single number is provided is considered as the number of nodes.
        :param mask: mask
        :param mapshape: shape of the som. Options are:
            - planar
            - toroid (not implemented yet)
            - cylinder (not implemented yet)

        :param lattice: type of lattice. Options are:
            - rect
            - hexa

        :param initialization: method to be used for initialization of the som.
            Options are:
            - pca
            - random

        :param name: name used to identify the som
        :param training: Training mode (seq, batch)
        """
        if normalization:
            normalizer = NormalizerFactory.build(normalization)
        else:
            normalizer = None
        neighborhood_calculator = NeighborhoodFactory.build(neighborhood)
        return SOM(data, neighborhood_calculator, normalizer, mapsize, mask,
                   mapshape, lattice, initialization, training, name, component_names)


class SOM(object):

    def __init__(self,
                 data,
                 neighborhood,
                 normalizer=None,
                 mapsize=None,
                 mask=None,
                 mapshape='planar',
                 lattice='hexa',
                 initialization='pca',
                 training='batch',
                 name='sompy_csom',
                 component_names=None):
        """
        Self Organizing Map

        :param data: data to be clustered, represented as a matrix of n rows,
            as inputs and m cols as input features
        :param neighborhood: neighborhood object calculator.
        :param normalizer: normalizer object calculator.
        :param mapsize: tuple/list defining the dimensions of the som. If
            single number is provided is considered as the number of nodes.
        :param mask: mask
        :param mapshape: shape of the som.
        :param lattice: type of lattice.
        :param initialization: method to be used for initialization of the som.
        :param name: name used to identify the som
        :param training: Training mode (seq, batch)
        """
        self._data = normalizer.normalize(data) if normalizer else data
        self._normalizer = normalizer
        self._dim = data.shape[1]
        self._dlen = data.shape[0]
        self._dlabel = None
        self._bmu = None
        self.lattice=lattice       # lattice type i.e. hexa
        self.name = name           # here it is sompy_isom 
        self.data_raw = data       # dataset as supplied by the user
        self.neighborhood = neighborhood
        self.mapshape = mapshape
        self.initialization = initialization        # here it is pca or random
        self.mask = mask or np.ones([1, self._dim])
        mapsize = self.calculate_map_size(lattice) if not mapsize else mapsize  # default mapsize is calculated as in MATLAB, if not given
        self.codebook = Codebook(mapsize, lattice)
        self.training = training     # batch training
        self._component_names = self.build_component_names() if component_names is None else component_names
        self._distance_matrix = self.calculate_map_dist()    # distance_matrix shape: nnodes x nnodes
        self._initialized_matrix = None 

    @property
    def component_names(self):
        return self._component_names

    @component_names.setter
    def component_names(self, compnames):
        if self._dim == len(compnames):
            self._component_names = np.asarray(compnames)[np.newaxis, :]
        else:
            raise ComponentNamesError('Component names should have the same '
                                      'size as the data dimension/features')

    def build_component_names(self):
        cc = ['Variable-' + str(i+1) for i in range(0, self._dim)]
        return cc

    @property
    def data_labels(self):
        return self._dlabel

    @data_labels.setter
    def data_labels(self, labels):
        """
        Set labels of the training data, it should be in the format of a list
        of strings
        """
        if labels.shape == (1, self._dlen):
            label = labels.T
        elif labels.shape == (self._dlen, 1):
            label = labels
        elif labels.shape == (self._dlen,):
            label = labels[:, np.newaxis]
        else:
            raise LabelsError('wrong label format')

        self._dlabel = label

    def build_data_labels(self):
        cc = ['dlabel-' + str(i) for i in range(0, self._dlen)]
        return np.asarray(cc)[:, np.newaxis]

    def calculate_map_dist(self):
        """
        Calculates the grid distance, which will be used during the training
        steps. It supports only planar grids for the moment
        """
        nnodes = self.codebook.nnodes

        distance_matrix = np.zeros((nnodes, nnodes))
        for i in range(nnodes):
            distance_matrix[i] = self.codebook.grid_dist(i).reshape(1, nnodes)
        return distance_matrix
    
    @timeit()
    def som_randinit(self):
        self.codebook.random_initialization(self._data)
        self.initialized_matrix = self.codebook.matrix.copy()
    
    @timeit()
    def som_lininit(self):    # Implemented on 30 Aug 2022

        self.codebook.pca_linear_initialization(self._data)
        self.initialized_matrix = self.codebook.matrix.copy()  # Added on 17 Aug 2022 on iSOM
        
    @timeit()
    def set_initialized_matrix(self, matrix):     # matrix: Matrix to be initialized is sent as input
        # Implemented on 30 Aug 2022 for MOO on iSOM
        self.initialized_matrix = matrix.copy() 
        self.codebook.matrix = matrix.copy()
        
        
    def set_codebook_matrix(self, matrix):     # matrix: Matrix to be initialized is sent as input; added on 9 sept for umatrix
        # Implemented on 30 Aug 2022 for MOO
        self.codebook.matrix = matrix.copy()

    @timeit()
    def train(self,
              n_job=1,                         # This will not change
              shared_memory=False,             # This will not change
              verbose='info',                  # Keep this default as: None
              train_rough_len=None,
              train_rough_radiusin=None,
              train_rough_radiusfin=None,
              train_finetune_len=None,
              train_finetune_radiusin=None,
              train_finetune_radiusfin=None,
              train_len_factor=1,
              maxtrainlen=np.Inf):
        """
        Trains the som

        :param n_job: number of jobs to use to parallelize the traning
        :param shared_memory: flag to active shared memory
        :param verbose: verbosity, could be 'debug', 'info' or None
        :param train_len_factor: Factor that multiply default training lenghts (similar to "training" parameter in the matlab version). (lbugnon)
        """
        logging.root.setLevel(
            getattr(logging, verbose.upper()) if verbose else logging.ERROR)

        logging.info(" Training...")
        logging.debug((
            "--------------------------------------------------------------\n"
            " details: \n"
            "      > data len is {data_len} and data dimension is {data_dim}\n"
            "      > map size is {mpsz0},{mpsz1}\n"
            "      > array size in log10 scale is {array_size}\n"
            "      > number of jobs in parallel: {n_job}\n"
            " -------------------------------------------------------------\n")
            .format(data_len=self._dlen,
                    data_dim=self._dim,
                    mpsz0=self.codebook.mapsize[0],
                    mpsz1=self.codebook.mapsize[1],
                    array_size=np.log10(
                        self._dlen * self.codebook.nnodes * self._dim),
                    n_job=n_job))
        

        self.rough_train(njob=n_job, shared_memory=shared_memory, trainlen=train_rough_len,
                         radiusin=train_rough_radiusin, radiusfin=train_rough_radiusfin,trainlen_factor=train_len_factor,maxtrainlen=maxtrainlen)
        self.finetune_train(njob=n_job, shared_memory=shared_memory, trainlen=train_finetune_len,
                            radiusin=train_finetune_radiusin, radiusfin=train_finetune_radiusfin,trainlen_factor=train_len_factor,maxtrainlen=maxtrainlen)
        logging.debug(
            " --------------------------------------------------------------")
        logging.info(" Final quantization error: %f" % np.mean(self._bmu[1]))

    def _calculate_ms_and_mpd(self):  # This function calculates the parameters ms and mpd to find default radius values(Same as MATLAB)
        mn = np.min(self.codebook.mapsize)
        max_s = max(self.codebook.mapsize[0], self.codebook.mapsize[1])

        if mn == 1:
            mpd = float(self.codebook.nnodes*10)/float(self._dlen)
        else:
            mpd = float(self.codebook.nnodes)/float(self._dlen)
        ms = max_s/2.0 if mn == 1 else max_s

        return ms, mpd

    def rough_train(self, njob=1, shared_memory=False, trainlen=None, radiusin=None, radiusfin=None,trainlen_factor=1,maxtrainlen=np.Inf):
        logging.info(" Rough training...")

        ms, mpd = self._calculate_ms_and_mpd()
        trainlen = min(int(np.ceil(30*mpd)),maxtrainlen) if not trainlen else trainlen
        trainlen=int(trainlen*trainlen_factor)

        if self.initialization == 'random':
            radiusin = max(1, np.ceil(ms/3.)) if not radiusin else radiusin
            radiusfin = max(1, radiusin/6.) if not radiusfin else radiusfin

        elif self.initialization == 'pca':
            radiusin = max(1, np.ceil(ms/8.)) if not radiusin else radiusin
            radiusfin = max(1, radiusin/4.) if not radiusfin else radiusfin

        self._batchtrain(trainlen, radiusin, radiusfin, njob, shared_memory)

    def finetune_train(self, njob=1, shared_memory=False, trainlen=None, radiusin=None, radiusfin=None,trainlen_factor=1,maxtrainlen=np.Inf):
        logging.info(" Finetune training...")

        ms, mpd = self._calculate_ms_and_mpd()

        if self.initialization == 'random':
            trainlen = min(int(np.ceil(50*mpd)),maxtrainlen) if not trainlen else trainlen
            radiusin = max(1, ms/12.)  if not radiusin else radiusin # from radius fin in rough training
            radiusfin = max(1, radiusin/25.) if not radiusfin else radiusfin

        elif self.initialization == 'pca':
            trainlen = min(int(np.ceil(40*mpd)),maxtrainlen) if not trainlen else trainlen
            radiusin = max(1, np.ceil(ms/8.)/4) if not radiusin else radiusin
            radiusfin = 1 if not radiusfin else radiusfin # max(1, ms/128)

        trainlen=int(trainlen_factor*trainlen)


        self._batchtrain(trainlen, radiusin, radiusfin, njob, shared_memory)

    def _batchtrain(self, trainlen, radiusin, radiusfin, njob=1,
                    shared_memory=False):
        radius = np.linspace(radiusin, radiusfin, trainlen)   # This is neighboordhood radius i.e. sigma(t) in formula of SOM algorithm

        if shared_memory:   # Not required in our case
            data = self._data
            data_folder = tempfile.mkdtemp()
            data_name = os.path.join(data_folder, 'data')
            dump(data, data_name)
            data = load(data_name, mmap_mode='r')
 
        else:        # This is executed in our required in our case
            data = self._data

        bmu = None

        logging.info(" radius_ini: %f , radius_final: %f, trainlen: %d\n" %
                     (radiusin, radiusfin, trainlen))

        for i in range(trainlen):
            t1 = time()
            # neighborhood: This is hci(t) as per SOM algorithm
            neighborhood = self.neighborhood.calculate(
                self._distance_matrix, radius[i], self.codebook.nnodes)
            bmu = self.find_bmu(data)                                       # Implemented on 28 Aug 2022 iSOM
            self.codebook.matrix = self.update_codebook_voronoi(data, bmu, neighborhood)
            qerror = (i + 1, round(time() - t1, 3), np.mean(bmu[1]))        # Implemented on 28 Aug 2022 iSOM

            logging.info(
                " epoch: %d ---> elapsed time:  %f, quantization error: %f\n" %
                qerror)
            if np.any(np.isnan(qerror)):
                logging.info("nan quantization error, exit train\n")


        bmu = self.find_bmu(data)                          # Implemented on 28 Aug 2022 (Update bmu after latest codebook update) on iSOM
        self._bmu = bmu


    @timeit(logging.DEBUG)
    def find_bmu(self,normalized_data):      # Implemented on 28 Aug 2022 on iSOM

        # normalized_data = ndarray format
        # codebook_matrix = ndarray format

        # dist_matrix size = (dlen x nnodes) 
        # distmatrix[0][4] = distance of data point index:0 from node index:4 (includes taking square root)
        
        codebook_matrix = self.codebook.matrix

        dist_matrix = cdist(normalized_data, codebook_matrix, 'euclidean') 

        # min_dist = distance of data point closest to the node
        # min_dist, arg_min : size(1xdlen)
        # min_dist[3] = distance of data point:0 from node: argmin[3]
        min_dist = np.apply_along_axis(min, axis=1, arr=dist_matrix)     
        arg_min = np.apply_along_axis(np.argmin, axis=1, arr=dist_matrix)

        bmu = np.vstack((arg_min, min_dist))   # stacks array arg_min and min_dist on top of each other

        # Note: here arg_min values are stored as float in bmu 1st row. Hence need to convert to int before using them.

        return bmu

    @timeit(logging.DEBUG)
    def update_codebook_voronoi(self, training_data, bmu, neighborhood):
        """
        Updates the weights of each node in the codebook that belongs to the
        bmu's neighborhood.

        First finds the Voronoi set of each node. It needs to calculate a
        smaller matrix.
        Super fast comparing to classic batch training algorithm, it is based
        on the implemented algorithm in som toolbox for Matlab by Helsinky
        University.

        :param training_data: input matrix with input vectors as rows and
            vector features as cols
        :param bmu: best matching unit for each input data. Has shape of
            (2, dlen) where first row has bmu indexes
        :param neighborhood: matrix representing the neighborhood of each bmu

        :returns: An updated codebook that incorporates the learnings from the
            input data
        """
        row = bmu[0].astype(int)
        col = np.arange(self._dlen)
        val = np.tile(1, self._dlen)
        P = csr_matrix((val, (row, col)), shape=(self.codebook.nnodes,
                       self._dlen))
        S = P.dot(training_data)

        # neighborhood has nnodes*nnodes and S has nnodes*dim
        # ---> Numerator has nnodes*dim
        nom = neighborhood.T.dot(S)
        nV = P.sum(axis=1).reshape(1, self.codebook.nnodes)
        denom = nV.dot(neighborhood.T).reshape(self.codebook.nnodes, 1)
        new_codebook = np.divide(nom, denom)

        return np.around(new_codebook, decimals=6)

    def calculate_map_size(self, lattice):   # Calculates default mapsize if not provided
        """
        Calculates the optimal map size given a dataset using eigenvalues and eigenvectors. Matlab ported
        :lattice: 'rect' or 'hex'
        :return: map sizes
        """
        D = self._data.copy()
        dlen = D.shape[0]
        dim = D.shape[1]
        munits = np.ceil(5 * (dlen ** 0.5))
        A = np.ndarray(shape=[dim, dim]) + np.Inf

        for i in range(dim):
            D[:, i] = D[:, i] - np.mean(D[np.isfinite(D[:, i]), i])

        for i in range(dim):
            for j in range(dim):
                c = D[:, i] * D[:, j]
                c = c[np.isfinite(c)]
                A[i, j] = sum(c) / len(c)
                A[j, i] = A[i, j]

        VS = np.linalg.eig(A)
        eigval = sorted(np.linalg.eig(A)[0])
        if eigval[-1] == 0 or eigval[-2] * munits < eigval[-1]:
            ratio = 1
        else:
            ratio = np.sqrt(eigval[-1] / eigval[-2])

        if lattice == "rect":
            #size1 = min(munits, round(np.sqrt(munits / ratio)))
            size1 = min(munits, round(np.sqrt(munits / ratio*np.sqrt(0.75))))
        else:
            size1 = min(munits, round(np.sqrt(munits / ratio*np.sqrt(0.75))))

        size2 = round(munits / size1)

        return [int(size2), int(size1)]


