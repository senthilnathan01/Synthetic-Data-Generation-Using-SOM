#!/usr/bin/env python
# coding: utf-8


# Modified Version: Dr. Palaniappan Ramu (palramu@iitm.ac.in)
#                   Associate Professor, Department of Engineering Design, Indian Institute of Technology Madras(IIT Madras),
#                   Chennai, India
#                   Website: https://ed.iitm.ac.in/~palramu/team.html

import numpy as np
import matplotlib
import matplotlib.cm as cm
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import ColorBar,LinearColorMapper,HoverTool
from bokeh.util.hex import cartesian_to_axial  # for converting x,y to q,r using hex_tile()

def hex_coord(mapsize, hex_size):       # Separately made for Umatrix due to mapsize difference        
    map_size = mapsize
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

def som_umatrix_map(vis, umat):
    
    hex_size = vis.hex_size
    umapsize = [2*vis.mapsize[0] - 1,2*vis.mapsize[1] - 1]  # Mapsize for U-matrix i.e. uy,ux
    nnodes = umapsize[0]*umapsize[1]
    fig_obj = []                      # List of all figure objects generated in each iteration.
    
    des_var = umat.ravel(order='F')
    minima = min(des_var)
    maxima = max(des_var)

    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima)  # norm = Normalized des_var on (0,1)
    mapper_1 = cm.ScalarMappable(norm=norm, cmap=cm.jet) # Mapping of normalized des_var with colorcode type: virdis,jet etc.
    # cm.RdYlGn, cm.viridis, cm.jet            
    lst_hex = vis.num_to_hex(des_var,mapper_1) # returns list of colorcodes for des_var

    # x_coord,y_coord = Hexagonal tile position in cartesian system
    # q_coord,r_coord = Hexagonal tile position in axial system
    # data,src_des_var = Data for Hexagonal tile coordinates, data x1, Color code of x1(i.e. lst1)
    x_coord,y_coord = hex_coord(umapsize, hex_size)
    q_coord,r_coord = cartesian_to_axial(x=x_coord, y=y_coord, size=hex_size, orientation='pointytop')

    # c = colorcode for 'nnodes' hex tiles of type: jet
    c = vis.num_to_hex_order(minima,maxima,nnodes,mapper_1)   # 'c' contains palette in hex form for the des_var data.
    # mapper_2 = Color mapping from colorcode c to interval of des_var.
    mapper_2 = LinearColorMapper(palette=c, low=minima, high=maxima)

    data = {'q':q_coord, 'r':r_coord, 'Node':np.arange(nnodes), 'Value':des_var, 'Color':lst_hex}

    src_data = ColumnDataSource(data=data)  # Bokeh source object used in plotting hex tiles, hover, color etc.
    ar = vis.aspect_ratio(umapsize)            # Finds aspect ratio for plots.
    # Creating a bokeh figure     
    p = figure(title='U-Matrix', tools="" , aspect_ratio=ar)
    # p.toolbar_location='right'
    p.title.align = "center"                       
    p.axis.visible = False
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.outline_line_color = None
    hover = HoverTool(tooltips = [("Value","@Value")], toggleable=False)  #tooltips = [("Node","@Node"),("Value","@Value")]
    p.add_tools(hover)  # Adding hovertool to the component planes plot
    p.toolbar.logo = None

    # Colorbar using colorcode for 'nnodes' tiles mapper_2
    # Colorbar for hitmaps
    cbar = ColorBar(color_mapper = mapper_2, major_label_text_font_size='14px')
    p.add_layout(cbar,'right')

    p.hex_tile(q='q', r='r', color='Color', source=src_data, size=hex_size, line_color='black')
    vis.umap_datasrc = data  # Save column data source as attribute 

    return p