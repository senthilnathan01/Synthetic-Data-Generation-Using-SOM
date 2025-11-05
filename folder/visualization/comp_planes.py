#!/usr/bin/env python
# coding: utf-8

# In[ ]:

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

def som_cplanes(vis):
    
    ndim = vis.dim                    # Number of dimensions = features+response
    codebook_vectors = vis.codebook   # codebook_vectors = n_array as per old code 
    hex_size = vis.hex_size
    mapsize = vis.mapsize
    nnodes = vis.nnodes
    fig_obj = []                      # List of all figure objects generated in each iteration.
    
    for i in range(ndim):
        des_var = codebook_vectors[:,i]
        minima = min(des_var)
        maxima = max(des_var)

        norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima)  # norm = Normalized des_var on (0,1)
        mapper_1 = cm.ScalarMappable(norm=norm, cmap=cm.jet) # Mapping of normalized des_var with colorcode type: virdis,jet etc.
        # cm.RdYlGn, cm.viridis, cm.jet            
        lst_hex = vis.num_to_hex(des_var,mapper_1) # returns list of colorcodes for des_var

        # x_coord,y_coord = Hexagonal tile position in cartesian system
        # q_coord,r_coord = Hexagonal tile position in axial system
        # data,src_des_var = Data for Hexagonal tile coordinates, data x1, Color code of x1(i.e. lst1)
        x_coord,y_coord = vis.hex_coord()
        q_coord,r_coord = cartesian_to_axial(x=x_coord, y=y_coord, size=hex_size, orientation='pointytop')

        # c = colorcode for 'nnodes' hex tiles of type: jet
        c = vis.num_to_hex_order(minima,maxima,nnodes,mapper_1)   # 'c' contains palette in hex form for the des_var data.
        # mapper_2 = Color mapping from colorcode c to interval of des_var.
        mapper_2 = LinearColorMapper(palette=c, low=minima, high=maxima)

        # To create hover labels for plots.
        str_title, Val_label, Color_label = vis.hover_label(ndim,i)

        # This if-else statement is defined so that sourcedata remains same for linking of plots    
        if i==0:
            data = {'q':q_coord, 'r':r_coord, 'Node':np.arange(nnodes), Val_label:des_var, Color_label:lst_hex}
        else:
            data[Val_label] = des_var
            data[Color_label] = lst_hex

        src_des_var = ColumnDataSource(data=data)  # Bokeh source object used in plotting hex tiles, hover, color etc.
        # ar = vis.aspect_ratio(mapsize)            # Finds aspect ratio for plots.
        # ar = 'auto'
        ar=1.3
        # Creating a bokeh figure     
        p = figure(title=str_title, tools="", aspect_ratio=ar)
        # p.toolbar_location='right'
        p.title.align = "center"                       
        p.axis.visible = False
        p.xgrid.visible = False
        p.ygrid.visible = False
        p.outline_line_color = None
        #p.title.text_font_size = f'{5*min(mapsize)}px'

        if i==ndim-1:
            hover = HoverTool(tooltips = vis.tooltip_response(data), toggleable=False)
        else:
            hover = HoverTool(tooltips = [("Node","@Node"),("Value","@{"+Val_label + "}") ], toggleable=False) #formatters = {"@{Val_label}":"printf"
        p.add_tools(hover)  # Adding hovertool to the component planes plot
        p.toolbar.logo = None

        # Colorbar using colorcode for 'nnodes' tiles mapper_2
        cbar = ColorBar(color_mapper = mapper_2, major_label_text_font_size='14px')  #, major_label_text_font_size=f'{2*min(mapsize)}px'
        p.add_layout(cbar,'right')

        fig_obj.append(p)  # Collecting list of all figure objects, i.e. component plane of each feature.
        
        if i==ndim-1:
            for j in range(ndim):
                str_title = vis.component_names[j]
                colorname = 'Color'+str_title
                # creating hexagonal tiles at (q_coord,r_coord) with colorcode as per colorname in src_des_var
                fig_obj[j].hex_tile(q='q', r='r', color=colorname, source=src_des_var, size=hex_size, line_color=None) #black
                
            vis.cplane_datasrc = data  # Save column data source as attribute

    output_fig = fig_obj.pop()
    fig_obj.insert(0, output_fig)
                
    return fig_obj
            

