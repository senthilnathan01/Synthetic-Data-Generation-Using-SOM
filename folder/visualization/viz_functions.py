#!/usr/bin/env python
# coding: utf-8
import math

# Modified Version: Dr. Palaniappan Ramu (palramu@iitm.ac.in)
#                   Associate Professor, Department of Engineering Design, Indian Institute of Technology Madras(IIT Madras),
#                   Chennai, India
#                   Website: https://ed.iitm.ac.in/~palramu/team.html

import numpy as np
import matplotlib
from bokeh.plotting import show
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
from .comp_planes import som_cplanes
from .hitmaps import som_hitmap
from .umatrix import som_umatrix_map

class Visualization_func():
    
    def __init__(self,SOM):  # SOM: Here SOM is an instance of class SOM.
        
        self.dim = SOM._dim                          # Number of dimensions = features+response
        self.mapsize = SOM.codebook.mapsize          # SOM Map size
        self.nnodes = np.prod(SOM.codebook.mapsize)  # Number of nodes in SOM map
        self.hex_size = 1  if SOM.lattice=='hexa' else None   # Will be used if lattice type = 'hexa'
        self.codebook = SOM._normalizer.denormalize_by(SOM.data_raw, SOM.codebook.matrix)  # denormalized SOM.codebook.matrix
        self.component_names = SOM._component_names
        self.cplane_objs = []
        self.cplane_datasrc = None
        self.umatrix = None
        self.hit_objs = None
        self.hitmap_datasrc = None
        self.umat_objs = None
        self.umap_datasrc = None
        self.canvas_columns = 5 if self.dim <= 25 else math.ceil(math.sqrt(self.dim))
    
    # Only self is needed as an argument since it has attributes: mapsize and hexsize.
    # Implemented 28 Aug 2022; orientation coordinates kept as MATLAB SOM i.e. top-left then zig-zag downwards
    def hex_coord(self):                   
        
        map_size = self.mapsize
        hex_size = self.hex_size
        h = hex_size                       # 2*hex_size is the total height of hexagon. hex_size is radius of the hexagon.
        w = np.sqrt(3)*h
        r,c = map_size[0],map_size[1]
        nnodes = map_size[0]*map_size[1]
        x = []                                                        # Here x,y are center of hexagons 
        y = []
        for i in range(r):
            if i%2 == 0:
                x.append([*np.linspace(0, (c-1)*w, c)])               # Eg. (0, 3sqrt(3), c)
            else:
                x.append([*np.linspace(0.5*w, (0.5*w+(c-1)*w), c)])   # Eg. (0+0.5 , 3sqrt(3)+0.5, c)

        for j in range(r):
                y.append(*[-j*1.5*h*np.ones(c)])  

        x_coord = np.array(x).T.reshape(nnodes,)   # Converting x and y matrix to array
        y_coord = np.array(y).T.reshape(nnodes,)   

        return x_coord , y_coord
    
    # Function to map numerical data to hex code for bokeh color bar; minima,maxima,nnodes not required.
    # num_to_hex_order or num_to_hex_sort
    def num_to_hex_order(self,minima,maxima,nnodes,mapper_1): # No need for self, but to maintain consistency lets keep it.
    
        lst_sort = np.linspace(minima,maxima,nnodes) # creating equally spaced data between min and max of each variable des_var.

        lst_rgb = []
        for v in lst_sort:
            lst_rgb.append(mapper_1.to_rgba(v))   # coverting numeric data desvar to rgba color code using mapper_1

        lst_hex = []
        for v in lst_rgb:
            lst_hex.append(matplotlib.colors.to_hex(v)) # coverting rgba code of desvar to hex color code of des_var

        return lst_hex   # returns hex code in sorted order


    # Function to map numerical data to hex code for each feature data.
    def num_to_hex(self,des_var,mapper_1): # No need for self, but to maintain consistency lets keep it.
        lst_rgb = []
        for v in des_var:
            lst_rgb.append(mapper_1.to_rgba(v))   # coverting numeric data desvar to rgba color code using mapper_1
    
        lst_hex = []
        for v in lst_rgb:
            lst_hex.append(matplotlib.colors.to_hex(v)) # coverting rgba code of desvar to hex color code of des_var
        
        return lst_hex  # returns hex code for des_var feature vector
    
    
    def tooltip_response(self,data):  # keeps labels also as input for main code. Here labels saved in self.
        a=[]
        for k in data.keys():
            a.append(k)   # creates list of keys
        # E.g.: labels = ['x1','x2','x3','x4','x5','x6','x7','x8','f(X)']
        labels = self.component_names
        ttips = []
        for i,k in enumerate(a):
            if i==2:
                ttips.append(("Node","@{"+k + "}"))
            elif i>2 and i%2!=0:
                ttips.append((labels[int((i-3)*0.5)],"@{"+k + "}"))
        return ttips # E.g. [('Node', '@Node'), ('x1', '@Valuex1'), ('x2', '@Valuex2'),... ('f(X)', '@Valuef')]
    
    
    def hover_label(self,ndim,i):  # To create hover labels for plots. 
        str_title = self.component_names[i]
        Val_label = 'Value'+self.component_names[i]
        Color_label = 'Color'+self.component_names[i]       
        return str_title, Val_label, Color_label
    
    
    def aspect_ratio(self,mapsize):  # Sets the right aspect ratio for hex plots

        if mapsize[0]>mapsize[1]:
            ar = (1.5*mapsize[0])/(np.sqrt(self.canvas_columns)*mapsize[1])   #This is the one
        else:
            ar = (1.5*mapsize[0])/(np.sqrt(self.canvas_columns)*mapsize[1])
            ar = 1/ar
        # ar = 1

        return ar
    
    def norm_plot_width_height(self):
        
        orig_w = self.mapsize[1]
        orig_h = self.mapsize[0]
        norm_w = (self.mapsize[1]/(max(orig_w,orig_h)))*100
        norm_h = (self.mapsize[0]/(max(orig_w,orig_h)))*100
        norm_w = int(norm_w)
        norm_h = int(norm_h)
        
        # norm_h = int(np.ceil(np.sqrt(self.dim)))
        # norm_w = int(np.ceil(self.dim/norm_h))
        
        return norm_w,norm_h
    
    def norm_umatplot_width_height(self):   # For Umatrix
        
        orig_w = 2*self.mapsize[1]-1
        orig_h = 2*self.mapsize[0]-1
        norm_w = (orig_w/(max(orig_w,orig_h)))*100
        norm_h = (orig_h/(max(orig_w,orig_h)))*100     
        norm_w = int(norm_w)
        norm_h = int(norm_h)   
        
        return norm_w,norm_h
        
    
    def hit_label(self, ndim, i):
        str_title = self.component_names[i]
        Color_label = 'Color'+self.component_names[i]       
        return str_title, Color_label

    def get_som_cplanes(self):
        return som_cplanes(self)  # som_cplanes() returns figure objects list for each dim


    def plot_cplanes(self, fig_obj):    # cplane_objs is output from som_cplanes(vis)
        # fig_obj = som_cplanes(self)
        self.update_cplane_objs(fig_obj)
        # Call function for plot_width and plot_height(they are normalized)
        w,h = self.norm_plot_width_height()
        total_canvas_width = 1440
        w = int(total_canvas_width / (3 * self.canvas_columns))

        # gridplot() merges all child plot tools into a single parent grid toolbar
#         grid_plots = gridplot(self.cplane_objs, ncols=3, width=w*3, height=h*3, merge_tools=True,
#                      toolbar_options=dict(logo=None)) # toolbar_options for disabling bokeh logo  #
        grid_plots = gridplot(self.cplane_objs, ncols=self.canvas_columns, width=w*3, height=w*3, merge_tools=True)  # With bokeh toolbar
        # output_notebook()
        # show(grid_plots) # default: width=300, height=300 # plot_width=mapsize[1]*50, plot_height=mapsize[0]*50

        return grid_plots  # Needed if implementing on web

    def get_som_hitmap(self, hits, comp, clr):
        return som_hitmap(self,hits,comp,clr)  # hit_obj is a list of all figure objects

    def plot_hitmap(self, hit_obj, comp):  # default hit is red; # Latest updated on 13 Sept
        # hit_obj = som_hitmap(self, hits, comp, clr)
        self.update_hit_objs(hit_obj)
        w,h = self.norm_plot_width_height()
        total_canvas_width = 1440
        w = int(total_canvas_width / (3 * self.canvas_columns))
        if type(comp)==list:
            # gridplot argument takes list as input and indexing via comp happens via ndarray; coz indexing of list does not happen
            #             hit_plots = gridplot(list(np.array(self.hit_objs)[comp]), ncols=3, width=w*3, height=h*3, merge_tools=True,
            #                      toolbar_options=dict(logo=None)) # toolbar_options for disabling bokeh logo
            hit_plots = gridplot(list(np.array(self.hit_objs)[comp]), ncols=self.canvas_columns, width=w * 3,
                                 height=h * 3, merge_tools=True)  # With bokeh toolbar
        elif comp=='all':
            #             hit_plots = gridplot(self.hit_objs, ncols=3, width=w*3, height=h*3, merge_tools=True,
            #                      toolbar_options=dict(logo=None)) # toolbar_options for disabling bokeh logo
            hit_plots = gridplot(self.hit_objs, ncols=self.canvas_columns, width=w * 3, height=h * 3,
                                 merge_tools=True)  # With bokeh toolbar
        else:
            hit_plots = self.hit_objs[comp]
        return hit_plots
        # output_notebook()
        # show(hit_plots) # default: width=300, height=300 # plot_width=mapsize[1]*50, plot_height=mapsize[0]*50

    def plot_umat(self, umat):
        umat_obj = som_umatrix_map(self,umat)
        self.update_umat_objs(umat_obj)
        w,h = self.norm_umatplot_width_height()
#         output_notebook()
#         show(umat_obj) # default: width=300, height=300 # plot_width=mapsize[1]*50, plot_height=mapsize[0]*50
        return umat_obj
        
   
    def update_cplane_objs(self,fig_obj):  # fig_obj are returned from som_cplanes 
        
        # Will update this attribute once we plot cplanes
        self.cplane_objs = fig_obj
        
    def update_hit_objs(self,hit_objs):
        
        self.hit_objs = hit_objs
        
    def update_umat_objs(self,umat_obj):
        
        self.umat_objs = umat_obj
        
        