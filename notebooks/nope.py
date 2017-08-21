# This Python file uses the following encoding: utf-8
#------------------------------------------------------------------------------
# Name:        nope.py
# Purpose:     Norwegian Pollutant Export model (NOPE)
#
# Author:      James Sample
#
# Created:     19/08/2017
# Copyright:   (c) James Sample and NIVA   
#------------------------------------------------------------------------------
""" Code for a simple export-coefficient-based based model for simulating 
    pollutant loads from Norwegian rivers. Developed as part of the RID
    programme.
    
        https://github.com/JamesSample/rid
"""

def run_nope(in_csv):
    """ Run the model with the specified input file.
    
    Args:
        in_csv: Raw str. Path to inout file
        
    Returns:
        NetworkX graph object with results added as node 
        attributes
    """
    import pandas as pd
    import numpy as np
    import networkx as nx

    # Read input file
    df = pd.read_csv(in_csv)

    # Build graph
    g = nx.DiGraph()

    # Add nodes
    for idx, row in df.iterrows():
        nd = row['regine']
        g.add_node(nd, local=row.to_dict(), accum={})

    # Add edges
    for idx, row in df.iterrows():
        fr_nd = row['regine']
        to_nd = row['regine_ned']
        g.add_edge(fr_nd, to_nd)

    # Accumulate
    g = accumulate_loads(g)
    
    return g

def accumulate_loads(g):
    """ Core function for TEOTIL-like network accumulation over a
        hydrological network.
    
    Args
        g Pre-built NetworkX graph. Must be a directed tree/forest
          and each nodes must have properties 'li' (internal load)
          and 'wt' (transmission factor)
          
    Returns
        Graph. g is modifed by adding the property 'Oi' to each node.
        This is the total amount of nutrient flowing out of the node.  
    """
    import networkx as nx
    from collections import defaultdict
    
    # Check directed tree
    assert nx.is_tree(g), 'g is not a valid tree.'
    assert nx.is_directed_acyclic_graph(g), 'g is not a valid DAG.'   

    # Param specific cols
    par_cols = ['aqu_%s_tonnes', 'ind_%s_tonnes', 'ren_%s_tonnes', 
                'spr_%s_tonnes', 'all_point_%s_tonnes', 'nat_diff_%s_tonnes',
                'anth_diff_%s_tonnes', 'all_sources_%s_tonnes']

    # Process nodes in topo order from headwaters down
    for nd in nx.topological_sort(g)[:-1]:
        # Get catchments directly upstream
        preds = g.predecessors(nd)
        
        if len(preds) > 0:
            # Accumulate total input from upstream
            # Counters default to 0
            a_up = 0
            q_up = 0
            tot_dict = defaultdict(int)
            
            # Loop over upstream catchments            
            for pred in preds:
                a_up += g.node[pred]['accum']['upstr_area_km2']
                q_up += g.node[pred]['accum']['q_m3/s']
                
                # Loop over quantities of interest
                for name in par_cols:
                    for par in ['n', 'p']:
                        tot_dict[name % par] += g.node[pred]['accum'][name % par]              
            
            # Assign outputs
            # Area and flow
            g.node[nd]['accum']['upstr_area_km2'] = a_up + g.node[nd]['local']['a_reg_km2']
            g.node[nd]['accum']['q_m3/s'] = q_up + g.node[nd]['local']['q_reg_m3/s']
            
            # Calculate output. Oi = ti(Li + Ii)
            for name in par_cols:
                for par in ['n', 'p']:            
                    g.node[nd]['accum'][name % par] = ((g.node[nd]['local'][name % par] +
                                                         tot_dict[name % par]) *
                                                        g.node[nd]['local']['trans_%s' % par])
                    
        else:
            # Area and flow
            g.node[nd]['accum']['upstr_area_km2'] = g.node[nd]['local']['a_reg_km2']
            g.node[nd]['accum']['q_m3/s'] = g.node[nd]['local']['q_reg_m3/s']        
            
            # No upstream inputs. Oi = ti * Li
            for name in par_cols:
                for par in ['n', 'p']:
                    g.node[nd]['accum'][name % par] = (g.node[nd]['local'][name % par] *
                                                       g.node[nd]['local']['trans_%s' % par])
    
    return g

def plot_network(g, catch_id, direct='down', stat='accum', 
                 quant='upstr_area_km2'):
    """ Create schematic diagram upstream or downstream of specified node.
    
    Args:
        g         NetworkX graph object returned by nope.run_nope()
        catch_id: Str. Regine ID of interest
        direct:   Str. 'up' or 'down'. Direction to trace network
        stat:     Str. 'local' or 'accum'. Type of results to display
        quant:    Str. Any of the returned result types
        
    Returns:
        NetworkX graph. Can be displayed using draw(g2, show='ipynb')        
    """
    import networkx as nx
    
    # Parse direction
    if direct == 'down':
        # Get sub-tree
        g2 = nx.dfs_tree(g, catch_id)
        
        # Update labels with 'quant'
        for nd in nx.topological_sort(g2)[:-1]:
            g2.node[nd]['label'] = '%s\n(%.2f)' % (nd, g.node[nd][stat][quant])  
            
    elif direct == 'up':
        # Get sub-tree
        g2 = nx.dfs_tree(g.reverse(), catch_id)

        # Update labels with 'quant'
        for nd in nx.topological_sort(g2):
            g2.node[nd]['label'] = '%s\n(%.2f)' % (nd, g.node[nd][stat][quant])  
            
    else:
        raise ValueError('"direct" must be "up" or "down".')     
    
    return g2

def make_map(g, stat='accum', quant='q_m3/s', sqrt=False,
              cmap='coolwarm', n_cats=10, plot_path=None):
    """ Display a map of the regine catchments, coloured according 
    to the quantity specified.
    
    Args:
        g          NetworkX graph object returned by nope.run_nope()
        stat:      Str. 'local' or 'accum'. Type of results to display
        quant:     Str. Any of the returned result types
        sqrt:      Bool. Whether to square-root transform 'quant' before 
                   plotting 
        cmap:      Str. Valid matplotlib colourmap
        n_cats     Int. Number of categories for colour scale
        plot_path: Raw Str. Optional. Path to which plot will be saved
        
    Returns:
        Matplotlib figure object 
    """
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from mpl_toolkits.basemap import Basemap
    import networkx as nx

    # Extract data of interest from graph
    reg_list = []
    par_list = []

    for nd in nx.topological_sort(g)[:-1]:
        reg_list.append(g.node[nd]['local']['regine'])
        par_list.append(g.node[nd][stat][quant])

    # Build df
    df = pd.DataFrame(data={quant:par_list}, index=reg_list)

    # Transform if necessary
    if sqrt:
        df[quant] = df[quant]**0.5

    # Build colour scheme
    values = df[quant]
    cm = plt.get_cmap(cmap)
    scheme = [cm(float(i) / n_cats) for i in range(n_cats+1)]
    bins = np.linspace(values.min(), values.max(), n_cats+1)
    df['bin'] = np.digitize(values, bins) - 1

    # Setup map
    fig = plt.figure(figsize=(12, 16))
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)

    # Title
    tit = quant.split('_')
    name = ' '.join(tit[:-1]).capitalize()
    unit = tit[-1]
    tit = '%s (%s)' % (name, unit)

    if sqrt:
        tit = 'log[%s]' % tit

    ax.set_title(tit, fontsize=20)

    # Use Albers Equal Area projection
    m = Basemap(projection='aea',
                width=1500000,
                height=2000000,
                resolution='i',
                lat_1=53,          # 1st standard parallel
                lat_2=73,          # 2st standard parallel
                lon_0=15,lat_0=63, # Central point
                ellps='WGS84')     # http://matplotlib.org/basemap/api/basemap_api.html

    # Add map components
    m.fillcontinents (color='darkkhaki', lake_color='darkkhaki', zorder=0)
    m.drawcountries(linewidth=1)
    m.drawcoastlines(linewidth=1)
    m.drawmapboundary(fill_color='paleturquoise')

    # Read shapefile. Must be in WGS84
    reg_shp = (r'C:\Data\James_Work\Staff\Oeyvind_K\Elveovervakingsprogrammet'
               r'\Data\gis\shapefiles\reg_minste_f_wgs84')

    m.readshapefile(reg_shp, 'regine', drawbounds=False)

    # Loop over features
    for info, shape in zip(m.regine_info, m.regine):
        reg = info['VASSDRAGNR']
        if reg not in df.index:
            color = '#dddddd' # If regine not found, colour grey
        else:
            color = scheme[int(df.ix[reg]['bin'])]      

        # Make patches
        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches)
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        ax.add_collection(pc)

    # Cover up Antarctica so legend can be placed over it.
    ax.axhspan(0, 2E5, facecolor='w', edgecolor='k', zorder=2)

    # Draw color legend.
    ax_legend = fig.add_axes([0.15, 0.16, 0.7, 0.03], zorder=3)
    cmap2 = mpl.colors.ListedColormap(scheme)
    cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap2, ticks=bins, 
                                   boundaries=bins, orientation='horizontal')
    cb.ax.set_xticklabels([str(round(i, 0)) for i in bins])

    # Save
    if plot_path:
        plt.savefig(plot_path, dpi=300)

    return fig