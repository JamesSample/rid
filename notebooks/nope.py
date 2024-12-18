# This Python file uses the following encoding: utf-8
# ------------------------------------------------------------------------------
# Name:        nope.py
# Purpose:     Norwegian Pollutant Export model (NOPE)
#
# Author:      James Sample
#
# Created:     19/08/2017
# Copyright:   (c) James Sample and NIVA
# ------------------------------------------------------------------------------
""" Code for a simple export-coefficient-based based model for simulating 
    pollutant loads from Norwegian rivers. Developed as part of the RID
    programme.
    
        https://github.com/JamesSample/rid
"""


def run_nope(data, par_list):
    """ Run the model with the specified inputs.
    
    Args:
        data:     Raw str or dataframe returned by nope.make_rid_input_file
        par_list: List of parameters in the input file
        
    Returns:
        NetworkX graph object with results added as node 
        attributes
    """
    import pandas as pd
    import numpy as np
    import networkx as nx

    # Parse input
    if isinstance(data, pd.DataFrame):
        df = data

    elif isinstance(data, str):
        df = pd.read_csv(data)

    else:
        raise ValueError('"data" must be either a "raw" string or a Pandas dataframe.')

    # Build graph
    g = nx.DiGraph()

    # Add nodes
    for idx, row in df.iterrows():
        nd = row["regine"]
        g.add_node(nd, local=row.to_dict(), accum={})

    # Add edges
    for idx, row in df.iterrows():
        fr_nd = row["regine"]
        to_nd = row["regine_ned"]
        g.add_edge(fr_nd, to_nd)

    # Accumulate
    g = accumulate_loads(g, par_list)

    return g


def build_calib_network(data, calib_node_set):
    """ Build an unattributed network for the calibration catchments.
    
    Args:
        data:           Dataframe or raw str to model input file. Must include
                        'regine' and 'regine_ned' columns describing network 
                        links.
        calib_node_set: Set of catchment IDs for which calibration data are 
                        available.
        
    Returns:
        (g, nd_list). Tuple. g is a NetworkX graph object for the sub-network
        upstream of catchments in calib_node_set. nd_list is a topologically
        sorted list of nodes.
    """
    import pandas as pd
    import numpy as np
    import networkx as nx

    # Parse input
    if isinstance(data, pd.DataFrame):
        df = data

    elif isinstance(data, str):
        df = pd.read_csv(data)

    else:
        raise ValueError('"data" must be either a "raw" string or a Pandas dataframe.')

    # Build graph
    g = nx.DiGraph()

    # Add nodes
    for idx, row in df.iterrows():
        nd = row["regine"]
        g.add_node(nd, local={}, accum={})

    # Add edges
    for idx, row in df.iterrows():
        fr_nd = row["regine"]
        to_nd = row["regine_ned"]
        g.add_edge(fr_nd, to_nd)

    # Check directed tree
    assert nx.is_tree(g), "g is not a valid tree."
    assert nx.is_directed_acyclic_graph(g), "g is not a valid DAG."

    # Get nodes upstream of each site with data
    nd_set = set()
    for nd in calib_node_set:
        nds = nx.dfs_tree(g.reverse(), nd).nodes()
        nd_set.update(nds)

    # Get subgraph
    g = g.subgraph(nd_set).copy()

    # Get topo node list
    nd_list = list(nx.topological_sort(g))

    return (g, nd_list)


def accumulate_loads(g, par_list):
    """ Core function for TEOTIL-like network accumulation over a
        hydrological network.
    
    Args
        g         Pre-built NetworkX graph. Must be a directed tree/forest
                  and each nodes must have properties 'local' (internal load)
                  and 'output' (empty dict).
        par_list: List of parameters in the input file
          
    Returns
        Graph. g is modifed by adding the property 'Oi' to each node.
        This is the total amount of nutrient flowing out of the node.  
    """
    import networkx as nx
    from collections import defaultdict

    # Convert par_list to lower case
    par_list = [i.lower() for i in par_list]

    # Check directed tree
    assert nx.is_tree(g), "g is not a valid tree."
    assert nx.is_directed_acyclic_graph(g), "g is not a valid DAG."

    # Param specific cols
    #    par_cols = ['aqu_%s_tonnes', 'ind_%s_tonnes', 'ren_%s_tonnes',
    #                'spr_%s_tonnes', 'all_point_%s_tonnes', 'nat_diff_%s_tonnes',
    #                'anth_diff_%s_tonnes', 'all_sources_%s_tonnes']

    # Changed 21/11/2018. See e-mail from John Rune received 20/11/2018 at 16.15
    # Now include 'urban' and 'agri_diff' as separate categories
    par_cols = [
        "aqu_%s_tonnes",
        "ind_%s_tonnes",
        "ren_%s_tonnes",
        "spr_%s_tonnes",
        "agri_pt_%s_tonnes",
        "all_point_%s_tonnes",
        "urban_%s_tonnes",
        "agri_diff_%s_tonnes",
        "nat_diff_%s_tonnes",
        "anth_diff_%s_tonnes",
        "all_sources_%s_tonnes",
    ]

    # Process nodes in topo order from headwaters down
    for nd in list(nx.topological_sort(g))[:-1]:
        # Get catchments directly upstream
        preds = list(g.predecessors(nd))

        if len(preds) > 0:
            # Accumulate total input from upstream
            # Counters default to 0
            a_up = 0
            q_up = 0
            tot_dict = defaultdict(int)

            # Loop over upstream catchments
            for pred in preds:
                a_up += g.nodes[pred]["accum"]["upstr_area_km2"]
                q_up += g.nodes[pred]["accum"]["q_m3/s"]

                # Loop over quantities of interest
                for name in par_cols:
                    for par in par_list:
                        tot_dict[name % par] += g.nodes[pred]["accum"][name % par]

            # Assign outputs
            # Area and flow
            g.nodes[nd]["accum"]["upstr_area_km2"] = (
                a_up + g.nodes[nd]["local"]["a_reg_km2"]
            )
            g.nodes[nd]["accum"]["q_m3/s"] = q_up + g.nodes[nd]["local"]["q_reg_m3/s"]

            # Calculate output. Oi = ti(Li + Ii)
            for name in par_cols:
                for par in par_list:
                    g.nodes[nd]["accum"][name % par] = (
                        g.nodes[nd]["local"][name % par] + tot_dict[name % par]
                    ) * g.nodes[nd]["local"]["trans_%s" % par]

        else:
            # Area and flow
            g.nodes[nd]["accum"]["upstr_area_km2"] = g.nodes[nd]["local"]["a_reg_km2"]
            g.nodes[nd]["accum"]["q_m3/s"] = g.nodes[nd]["local"]["q_reg_m3/s"]

            # No upstream inputs. Oi = ti * Li
            for name in par_cols:
                for par in par_list:
                    g.nodes[nd]["accum"][name % par] = (
                        g.nodes[nd]["local"][name % par]
                        * g.nodes[nd]["local"]["trans_%s" % par]
                    )

    return g


def update_and_accumulate(g, nd_list, year, data_dict, cal_pars, par_list, reg_set):
    """ Update network with properties for 'year' and accumulate 
        downstream.
    
    Args
        g:         Pre-built NetworkX graph. Must be a directed tree/forest
                   and each nodes must have properties 'local' (internal load)
                   and 'output' (empty dict).
        nd_list:   List. Topologically sorted list of nodes of g
        year:      Int. Year of interest
        data_dict: Dict. data_dict['node', year]['quantity]
        cal_pars:  Dict. Calibration parameters
        par_list:  List of parameters in the input file
        reg_set:  List of regine ID of interest
          
    Returns
        Dict containing accumulated loads for IDs in reg_set.
    """
    import networkx as nx
    import pandas as pd
    from collections import defaultdict

    # Container for results
    out_dict = {"regine": [], "q_m3/s": []}
    for par in par_list:
        out_dict["%s_tonnes" % par] = []

    # Process nodes in topo order from headwaters down
    for nd in nd_list:
        # Update local properties
        # 1. Flow
        g.nodes[nd]["local"]["q_reg_m3/s"] = data_dict[(nd, year)]["q_reg_m3/s"]

        # 2. Water chem and transmission
        for par in par_list:
            g.nodes[nd]["local"]["trans_%s" % par] = (
                data_dict[(nd, year)]["trans_%s" % par] * cal_pars["b_r_%s" % par]
            )

            g.nodes[nd]["local"]["%s_tonnes" % par] = (
                data_dict[(nd, year)]["all_point_%s_tonnes" % par]
                * cal_pars["b_p_%s" % par]
            ) + (
                data_dict[(nd, year)]["all_diff_%s_tonnes" % par]
                * cal_pars["b_d_%s" % par]
            )

        # Accumulate
        # Get catchments directly upstream
        preds = list(g.predecessors(nd))

        if len(preds) > 0:
            # Accumulate total input from upstream
            # Counters default to 0
            q_up = 0
            tot_dict = defaultdict(int)

            # Loop over upstream catchments
            for pred in preds:
                q_up += g.nodes[pred]["accum"]["q_m3/s"]

                # Loop over quantities of interest
                for par in par_list:
                    tot_dict["%s_tonnes" % par] += g.nodes[pred]["accum"][
                        "%s_tonnes" % par
                    ]

            # Assign outputs
            # Flow
            g.nodes[nd]["accum"]["q_m3/s"] = q_up + g.nodes[nd]["local"]["q_reg_m3/s"]

            # Calculate output. Oi = ti(Li + Ii)
            for par in par_list:
                g.nodes[nd]["accum"]["%s_tonnes" % par] = (
                    g.nodes[nd]["local"]["%s_tonnes" % par]
                    + tot_dict["%s_tonnes" % par]
                ) * g.nodes[nd]["local"]["trans_%s" % par]

        else:
            # Flow
            g.nodes[nd]["accum"]["q_m3/s"] = g.nodes[nd]["local"]["q_reg_m3/s"]

            # No upstream inputs. Oi = ti * Li
            for par in par_list:
                g.nodes[nd]["accum"]["%s_tonnes" % par] = (
                    g.nodes[nd]["local"]["%s_tonnes" % par]
                    * g.nodes[nd]["local"]["trans_%s" % par]
                )

    # Add results to dict
    for nd in reg_set:
        out_dict["regine"].append(nd)
        out_dict["q_m3/s"].append(g.nodes[nd]["accum"]["q_m3/s"])
        for par in par_list:
            out_dict["%s_tonnes" % par].append(g.nodes[nd]["accum"]["%s_tonnes" % par])

    # Build df
    df = pd.DataFrame(out_dict)

    # Add year
    df["year"] = year

    return df


def run_model_multi_year(
    g, nd_list, st_yr, end_yr, in_data, par_list, reg_set, cal_pars=None
):
    """ Run model for specified years.
    
    Args
        g:         Pre-built NetworkX graph. Must be a directed tree/forest
                   and each nodes must have properties 'local' (internal load)
                   and 'output' (empty dict).
        nd_list:   List. Topologically sorted list of nodes of g
        st_yr:     Int. Start year of interest
        end_yr:    Int. End year of interest
        in_data    Dict. All data required to run model
        cal_pars:  Dict. Calibration parameters
        par_list:  List of parameters in the input file
        reg_set:   List of regine IDs of interest
          
    Returns
        Dataframe of annual accumulated loads for IDs in reg_set.
    """
    import pandas as pd

    # Build cal_par dict if necessary
    if cal_pars is None:
        cal_pars = {}
        for par in par_list:
            for coef in ["b_r", "b_p", "b_d"]:
                # Set defaults to 1
                cal_pars["%s_%s" % (coef, par)] = 1

    # Container for output
    df_list = []

    # Loop over years
    for year in range(st_yr, end_yr + 1):
        df = update_and_accumulate(
            g, nd_list, year, in_data, cal_pars, par_list, reg_set
        )

        df_list.append(df)

    # Combine
    df = pd.concat(df_list, axis=0)

    # Reorder
    cols = ["regine", "year", "q_m3/s"] + ["%s_tonnes" % i for i in par_list]
    df = df[cols]

    return df


def plot_network(g, catch_id, direct="down", stat="accum", quant="upstr_area_km2"):
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
    import graphviz

    # Parse direction
    if direct == "down":
        # Get sub-tree
        g2 = nx.dfs_tree(g, catch_id)

        # Update labels with 'quant'
        for nd in list(nx.topological_sort(g2))[:-1]:
            g2.nodes[nd]["label"] = "%s\n(%.2f)" % (nd, g.nodes[nd][stat][quant])

    elif direct == "up":
        # Get sub-tree
        g2 = nx.dfs_tree(g.reverse(), catch_id).reverse()

        # Update labels with 'quant'
        for nd in list(nx.topological_sort(g2)):
            g2.nodes[nd]["label"] = "%s\n(%.2f)" % (nd, g.nodes[nd][stat][quant])

    else:
        raise ValueError('"direct" must be "up" or "down".')

    # Draw
    res = nx.nx_agraph.to_agraph(g2)
    res.layout("dot")

    return graphviz.Source(res.to_string())


def make_map(
    g,
    stat="accum",
    quant="q_m3/s",
    sqrt=False,
    cmap="coolwarm",
    n_cats=10,
    plot_path=None,
):
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

    for nd in list(nx.topological_sort(g))[:-1]:
        reg_list.append(g.nodes[nd]["local"]["regine"])
        par_list.append(g.nodes[nd][stat][quant])

    # Build df
    df = pd.DataFrame(data={quant: par_list}, index=reg_list)

    # Transform if necessary
    if sqrt:
        df[quant] = df[quant] ** 0.5

    # Build colour scheme
    values = df[quant]
    cm = plt.get_cmap(cmap)
    scheme = [cm(float(i) / n_cats) for i in range(n_cats + 1)]
    bins = np.linspace(values.min(), values.max(), n_cats + 1)
    df["bin"] = np.digitize(values, bins) - 1

    # Setup map
    fig = plt.figure(figsize=(12, 16))
    ax = fig.add_subplot(111, facecolor="w", frame_on=False)

    # Title
    tit = quant.split("_")
    name = " ".join(tit[:-1]).capitalize()
    unit = tit[-1]
    tit = "%s (%s)" % (name, unit)

    if sqrt:
        tit = "log[%s]" % tit

    ax.set_title(tit, fontsize=20)

    # Use Albers Equal Area projection
    m = Basemap(
        projection="aea",
        width=1500000,
        height=2000000,
        resolution="i",
        lat_1=53,  # 1st standard parallel
        lat_2=73,  # 2st standard parallel
        lon_0=15,
        lat_0=63,  # Central point
        ellps="WGS84",
    )  # http://matplotlib.org/basemap/api/basemap_api.html

    # Add map components
    m.fillcontinents(color="darkkhaki", lake_color="paleturquoise", zorder=0)
    m.drawcountries(linewidth=1)
    m.drawcoastlines(linewidth=1)
    m.drawmapboundary(fill_color="paleturquoise")

    # Read shapefile. Must be in WGS84
    reg_shp = (
        r"C:\Data\James_Work\Staff\Oeyvind_K\Elveovervakingsprogrammet"
        r"\Data\gis\shapefiles\reg_minste_f_wgs84"
    )

    m.readshapefile(reg_shp, "regine", drawbounds=False)

    # Loop over features
    for info, shape in zip(m.regine_info, m.regine):
        reg = info["VASSDRAGNR"]
        if reg not in df.index:
            color = "#dddddd"  # If regine not found, colour grey
        else:
            color = scheme[int(df.ix[reg]["bin"])]

        # Make patches
        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches)
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        ax.add_collection(pc)

    # Cover up Antarctica so legend can be placed over it.
    ax.axhspan(0, 2e5, facecolor="w", edgecolor="k", zorder=2)

    # Draw color legend.
    ax_legend = fig.add_axes([0.15, 0.16, 0.7, 0.03], zorder=3)
    cmap2 = mpl.colors.ListedColormap(scheme)
    cb = mpl.colorbar.ColorbarBase(
        ax_legend, cmap=cmap2, ticks=bins, boundaries=bins, orientation="horizontal"
    )
    cb.ax.set_xticklabels([str(round(i, 0)) for i in bins])

    # Save
    if plot_path:
        plt.savefig(plot_path, dpi=300)

    return fig


def model_to_dataframe(g, out_path=None):
    """ Convert a NOPE graph to a Pandas dataframe. If a path
        is supplied, the DF will be written to CSV format.
    
    Args:
        g          NetworkX graph object returned by nope.run_nope()
        plot_path: Raw Str. Optional. CSV path to which df will 
                   be saved
        
    Returns:
        Dataframe 
    """
    import networkx as nx
    import pandas as pd
    from collections import defaultdict

    # Container for data
    out_dict = defaultdict(list)

    # Loop over data
    for nd in list(nx.topological_sort(g))[:-1]:
        for stat in ["local", "accum"]:
            for key in g.nodes[nd][stat]:
                out_dict["%s_%s" % (stat, key)].append(g.nodes[nd][stat][key])

    # Convert to df
    df = pd.DataFrame(out_dict)

    # Reorder cols
    key_cols = ["local_regine", "local_regine_ned"]
    cols = [i for i in df.columns if not i in key_cols]
    cols.sort()
    df = df[key_cols + cols]
    cols = list(df.columns)
    cols[:2] = ["regine", "regine_ned"]
    df.columns = cols

    # Write output
    if out_path:
        df.to_csv(out_path, index=False, encoding="utf-8")

    return df


def get_annual_spredt_data(year, engine, par_list=["Tot-N", "Tot-P"]):
    """ Get annual spredt data from RESA2.
    
    Args:
        year:     Int. Year of interest
        par_list: List. Parameters defined in 
                  RESA2.RID_PUNKTKILDER_OUTPAR_DEF
        engine:   SQL-Alchemy 'engine' object already connected 
                  to RESA2
    
    Returns:
        Dataframe
    """
    import pandas as pd

    # Get data, converting units to tonnes
    sql = (
        "SELECT a.komm_no as komnr, "
        "       c.name, "
        "       (a.value*b.factor) AS value "
        "FROM resa2.rid_kilder_spredt_values a, "
        "resa2.rid_punktkilder_inp_outp b, "
        "resa2.rid_punktkilder_outpar_def c "
        "WHERE a.inp_par_id = b.in_pid "
        "AND b.out_pid = c.out_pid "
        "AND ar = %s" % year
    )

    spr_df = pd.read_sql(sql, engine)

    # Only continue if data
    if len(spr_df) == 0:
        print("No spredt data for %s." % year)

        return None

    else:
        # Pivot
        spr_df = spr_df.pivot(index="komnr", columns="name", values="value")

        # Tidy
        spr_df = spr_df[par_list]
        cols = ["spr_%s_tonnes" % i.lower() for i in spr_df.columns]
        spr_df.columns = cols
        spr_df.columns.name = ""
        spr_df.reset_index(inplace=True)
        spr_df.dropna(subset=["komnr",], inplace=True)
        spr_df.dropna(subset=cols, how="all", inplace=True)
        spr_df["komnr"] = spr_df["komnr"].astype(int)

        return spr_df


def get_annual_aquaculture_data(year, engine, par_list=["Tot-N", "Tot-P"]):
    """ Get annual aquaculture data from RESA2.
    
    Args:
        year:     Int. Year of interest
        par_list: List. Parameters defined in 
                  RESA2.RID_PUNKTKILDER_OUTPAR_DEF
        engine:   SQL-Alchemy 'engine' object already connected 
                  to RESA2
    
    Returns:
        Dataframe
    """
    import pandas as pd

    # Get data, converting units to tonnes
    sql = (
        "SELECT regine, name, SUM(value) AS value FROM ( "
        "  SELECT b.regine, "
        "         c.name, "
        "         (a.value*d.factor) AS value "
        "  FROM resa2.rid_kilder_aqkult_values a, "
        "  resa2.rid_kilder_aquakultur b, "
        "  resa2.rid_punktkilder_outpar_def c, "
        "  resa2.rid_punktkilder_inp_outp d "
        "  WHERE a.anlegg_nr = b.nr "
        "  AND a.inp_par_id = d.in_pid "
        "  AND c.out_pid = d.out_pid "
        "  AND ar = %s) "
        "GROUP BY regine, name" % year
    )

    aqu_df = pd.read_sql(sql, engine)

    # Only continue if data
    if len(aqu_df) == 0:
        print("No aquaculture data for %s." % year)

        return None

    else:
        # Pivot
        aqu_df = aqu_df.pivot(index="regine", columns="name", values="value")

        # Tidy
        aqu_df = aqu_df[par_list]
        cols = ["aqu_%s_tonnes" % i.lower() for i in aqu_df.columns]
        aqu_df.columns = cols
        aqu_df.columns.name = ""
        aqu_df.reset_index(inplace=True)
        aqu_df.dropna(subset=["regine",], inplace=True)
        aqu_df.dropna(subset=cols, how="all", inplace=True)

        return aqu_df


def get_annual_renseanlegg_data(year, engine, par_list=["Tot-N", "Tot-P"]):
    """ Get annual renseanlegg data from RESA2.
    
    Args:
        year:     Int. Year of interest
        par_list: List. Parameters defined in 
                  RESA2.RID_PUNKTKILDER_OUTPAR_DEF
        engine:   SQL-Alchemy 'engine' object already connected 
                  to RESA2
    
    Returns:
        Dataframe
    """
    import pandas as pd

    sql = (
        "SELECT regine, name, SUM(value) AS value FROM ( "
        "  SELECT b.regine, "
        "         b.type, "
        "         c.name, "
        "         (a.value*d.factor) AS value "
        "  FROM resa2.rid_punktkilder_inpar_values a, "
        "  resa2.rid_punktkilder b, "
        "  resa2.rid_punktkilder_outpar_def c, "
        "  resa2.rid_punktkilder_inp_outp d "
        "  WHERE a.anlegg_nr = b.anlegg_nr "
        "  AND a.inp_par_id = d.in_pid "
        "  AND c.out_pid = d.out_pid "
        "  AND year = %s) "
        "WHERE type = 'RENSEANLEGG' "
        "GROUP BY regine, type, name" % year
    )

    ren_df = pd.read_sql(sql, engine)

    # Only continue if data
    if len(ren_df) == 0:
        print("No renseanlegg data for %s." % year)

        return None

    else:
        # Pivot
        ren_df = ren_df.pivot(index="regine", columns="name", values="value")

        # Tidy
        ren_df = ren_df[par_list]
        cols = ["ren_%s_tonnes" % i.lower() for i in ren_df.columns]
        ren_df.columns = cols
        ren_df.columns.name = ""
        ren_df.reset_index(inplace=True)
        ren_df.dropna(subset=["regine",], inplace=True)
        ren_df.dropna(subset=cols, how="all", inplace=True)

        return ren_df


def get_annual_industry_data(year, engine, par_list=["Tot-N", "Tot-P"]):
    """ Get annual industry data from RESA2.
    
    Args:
        year:     Int. Year of interest
        par_list: List. Parameters defined in 
                  RESA2.RID_PUNKTKILDER_OUTPAR_DEF
        engine:   SQL-Alchemy 'engine' object already connected 
                  to RESA2
    
    Returns:
        Dataframe
    """
    import pandas as pd

    sql = (
        "SELECT regine, name, SUM(value) AS value FROM ( "
        "  SELECT b.regine, "
        "         b.type, "
        "         c.name, "
        "         (a.value*d.factor) AS value "
        "  FROM resa2.rid_punktkilder_inpar_values a, "
        "  resa2.rid_punktkilder b, "
        "  resa2.rid_punktkilder_outpar_def c, "
        "  resa2.rid_punktkilder_inp_outp d "
        "  WHERE a.anlegg_nr = b.anlegg_nr "
        "  AND a.inp_par_id = d.in_pid "
        "  AND c.out_pid = d.out_pid "
        "  AND year = %s) "
        "WHERE type = 'INDUSTRI' "
        "GROUP BY regine, type, name" % year
    )

    ind_df = pd.read_sql(sql, engine)

    # Only continue if data
    if len(ind_df) == 0:
        print("No industry data for %s." % year)

        return None

    else:
        # Pivot
        ind_df = ind_df.pivot(index="regine", columns="name", values="value")

        # Tidy
        ind_df = ind_df[par_list]
        cols = ["ind_%s_tonnes" % i.lower() for i in ind_df.columns]
        ind_df.columns = cols
        ind_df.columns.name = ""
        ind_df.reset_index(inplace=True)
        ind_df.dropna(subset=["regine",], inplace=True)
        ind_df.dropna(subset=cols, how="all", inplace=True)

        return ind_df


def get_annual_vassdrag_mean_flows(year, engine):
    """ Get annual flow data for main NVE vassdrags based on 
        RESA2.DISCHARGE_VALUES.
    
    Args:
        year:     Int. Year of interest
        engine:   SQL-Alchemy 'engine' object already connected to 
                  RESA2
    
    Returns:
        Dataframe
    """
    import pandas as pd

    # Get NVE stn IDs
    sql = (
        "SELECT dis_station_id, TO_NUMBER(nve_serienummer) as Vassdrag "
        "FROM resa2.discharge_stations "
        "WHERE dis_station_name LIKE 'NVE Modellert%'"
    )

    nve_stn_df = pd.read_sql_query(sql, engine)
    nve_stn_df.index = nve_stn_df["dis_station_id"]
    del nve_stn_df["dis_station_id"]

    # Get avg. annual values for NVE stns
    sql = (
        "SELECT dis_station_id, AVG(xvalue) AS q_yr "
        "FROM resa2.discharge_values "
        "WHERE dis_station_id in ( "
        "  SELECT dis_station_id "
        "  FROM resa2.discharge_stations "
        "  WHERE dis_station_name LIKE 'NVE Modellert%%') "
        "AND TO_CHAR(xdate, 'YYYY') = %s "
        "GROUP BY dis_station_id "
        "ORDER BY dis_station_id" % year
    )

    an_avg_df = pd.read_sql_query(sql, engine)
    an_avg_df.index = an_avg_df["dis_station_id"]
    del an_avg_df["dis_station_id"]

    # Combine
    q_df = pd.concat([nve_stn_df, an_avg_df], axis=1)

    # Tidy
    q_df.reset_index(inplace=True, drop=True)
    q_df.sort_values(by="vassdrag", ascending=True, inplace=True)
    q_df.columns = ["vassom", "q_yr_m3/s"]

    return q_df


def get_annual_agricultural_coefficients(year, engine):
    """ Get annual agricultural inputs from Bioforsk and 
        convert to land use coefficients.
    
    Args:
        year:     Int. Year of interest
        engine:   SQL-Alchemy 'engine' object already connected to 
                  RESA2
    
    Returns:
        Dataframe
    """
    import pandas as pd
    import os

    # Read LU areas (same values used every year)
    in_csv = r"../../../NOPE/NOPE_Core_Input_Data/fysone_land_areas.csv"
    lu_areas = pd.read_csv(in_csv, sep=";", encoding="windows-1252")

    # Read Bioforsk data
    sql = "SELECT * FROM RESA2.RID_AGRI_INPUTS " "WHERE year = %s" % year
    lu_lds = pd.read_sql(sql, engine)
    del lu_lds["year"]

    # Check have data
    if len(lu_lds) == 0:
        print("No agricultural land use coefficients for %s." % year)

    # Join
    lu_df = pd.merge(lu_lds, lu_areas, how="outer", on="omrade")

    # Calculate required columns
    # N
    lu_df["agri_diff_tot-n_kg/km2"] = lu_df["n_diff_kg"] / lu_df["a_fy_agri_km2"]
    lu_df["agri_point_tot-n_kg/km2"] = (
        lu_df["n_point_kg"] / lu_df["a_fy_agri_km2"]
    )  # Orig a_fy_eng_km2??
    lu_df["agri_back_tot-n_kg/km2"] = lu_df["n_back_kg"] / lu_df["a_fy_agri_km2"]

    # P
    lu_df["agri_diff_tot-p_kg/km2"] = lu_df["p_diff_kg"] / lu_df["a_fy_agri_km2"]
    lu_df["agri_point_tot-p_kg/km2"] = (
        lu_df["p_point_kg"] / lu_df["a_fy_agri_km2"]
    )  # Orig a_fy_eng_km2??
    lu_df["agri_back_tot-p_kg/km2"] = lu_df["p_back_kg"] / lu_df["a_fy_agri_km2"]

    # Get cols of interest
    cols = [
        "fylke_sone",
        "fysone_name",
        "agri_diff_tot-n_kg/km2",
        "agri_point_tot-n_kg/km2",
        "agri_back_tot-n_kg/km2",
        "agri_diff_tot-p_kg/km2",
        "agri_point_tot-p_kg/km2",
        "agri_back_tot-p_kg/km2",
        "a_fy_agri_km2",
        "a_fy_eng_km2",
    ]

    lu_df = lu_df[cols]

    return lu_df


def make_rid_input_file(year, engine, nope_fold, out_csv, par_list=["Tot-N", "Tot-P"]):
    """ Builds a NOPE input file for the RID programme for the 
        specified year. All the required data must be complete in 
        RESA2.
    
    Args:
        year:      Int. Year of interest
        par_list:  List. Parameters defined in 
                   RESA2.RID_PUNKTKILDER_OUTPAR_DEF
        out_csv:   Path for output CSV file
        nope_fold: Path to folder containing core NOPE data files
        engine:    SQL-Alchemy 'engine' object already connected 
                   to RESA2
    
    Returns:
        Dataframe. The CSV is written to the specified path.
    """
    import pandas as pd
    import numpy as np
    import os

    # Read data from RESA2
    spr_df = get_annual_spredt_data(year, engine, par_list=par_list)
    aqu_df = get_annual_aquaculture_data(year, engine, par_list=par_list)
    ren_df = get_annual_renseanlegg_data(year, engine, par_list=par_list)
    ind_df = get_annual_industry_data(year, engine, par_list=par_list)
    agri_df = get_annual_agricultural_coefficients(year, engine)
    q_df = get_annual_vassdrag_mean_flows(year, engine)

    # Read core NOPE inputs
    # 1. Regine network
    # Changes to kommuner boundaries in 2017 require different files for
    # different years
    if year < 2017:
        csv_path = os.path.join(nope_fold, "regine_pre_2017.csv")
        reg_df = pd.read_csv(csv_path, index_col=0, sep=";")
    elif year == 2017:
        csv_path = os.path.join(nope_fold, "regine_2017.csv")
        reg_df = pd.read_csv(csv_path, index_col=0, sep=";")
    else:
        csv_path = os.path.join(nope_fold, "regine_2018_onwards.csv")
        reg_df = pd.read_csv(csv_path, index_col=0, sep=";")

    # 2. Retention factors
    csv_path = os.path.join(nope_fold, "retention.csv")
    ret_df = pd.read_csv(csv_path, sep=";")

    # 3. Land cover
    csv_path = os.path.join(nope_fold, "land_cover.csv")
    lc_df = pd.read_csv(csv_path, index_col=0, sep=";")

    # 4. Lake areas
    csv_path = os.path.join(nope_fold, "lake_areas.csv")
    la_df = pd.read_csv(csv_path, index_col=0, sep=";")

    # 5. Background coefficients
    csv_path = os.path.join(nope_fold, "back_coeffs.csv")
    back_df = pd.read_csv(csv_path, sep=";")

    # 7. Fylke-Sone
    csv_path = os.path.join(nope_fold, "regine_fysone.csv")
    fy_df = pd.read_csv(csv_path, sep=";")

    # Convert par_list to lower case
    par_list = [i.lower() for i in par_list]

    # Process data
    # 1. Land use
    # 1.1 Land areas
    # Join lu datasets
    area_df = pd.concat([reg_df, lc_df, la_df], axis=1, sort=True)
    area_df.index.name = "regine"
    area_df.reset_index(inplace=True)

    # Fill NaN
    area_df.fillna(value=0, inplace=True)

    # Get total area of categories
    area_df["a_sum"] = (
        area_df["a_wood_km2"]
        + area_df["a_agri_km2"]
        + area_df["a_upland_km2"]
        + area_df["a_glacier_km2"]
        + area_df["a_urban_km2"]
        + area_df["a_sea_km2"]
        + area_df["a_lake_km2"]
    )

    # If total exceeds overall area, calc correction factor
    area_df["a_cor_fac"] = np.where(
        area_df["a_sum"] > area_df["a_reg_km2"],
        area_df["a_reg_km2"] / area_df["a_sum"],
        1,
    )

    # Apply correction factor
    area_cols = [
        "a_wood_km2",
        "a_agri_km2",
        "a_upland_km2",
        "a_glacier_km2",
        "a_urban_km2",
        "a_sea_km2",
        "a_lake_km2",
        "a_sum",
    ]

    for col in area_cols:
        area_df[col] = area_df[col] * area_df["a_cor_fac"]

    # Calc 'other' column
    area_df["a_other_km2"] = area_df["a_reg_km2"] - area_df["a_sum"]

    # Combine 'glacier' and 'upland' as 'upland'
    area_df["a_upland_km2"] = area_df["a_upland_km2"] + area_df["a_glacier_km2"]

    # Add 'land area' column
    area_df["a_land_km2"] = area_df["a_reg_km2"] - area_df["a_sea_km2"]

    # Tidy
    del area_df["a_glacier_km2"], area_df["a_sum"], area_df["a_cor_fac"]

    # 1.2. Join background coeffs
    area_df = pd.merge(area_df, back_df, how="left", on="regine")

    # 1.3. Join agri coeffs
    area_df = pd.merge(area_df, fy_df, how="left", on="regine")
    area_df = pd.merge(area_df, agri_df, how="left", on="fylke_sone")

    # 2. Discharge
    # Sum LTA to vassom level
    lta_df = area_df[["vassom", "q_reg_m3/s"]].groupby("vassom").sum().reset_index()
    lta_df.columns = ["vassom", "q_lta_m3/s"]

    # Join
    q_df = pd.merge(lta_df, q_df, how="left", on="vassom")

    # Calculate corr fac
    q_df["q_fac"] = q_df["q_yr_m3/s"] / q_df["q_lta_m3/s"]

    # Join and reset index
    df = pd.merge(area_df, q_df, how="left", on="vassom")
    df.index = df["regine"]
    del df["regine"]

    # Calculate regine-specific flow for this year
    for col in ["q_sp_m3/s/km2", "runoff_mm/yr", "q_reg_m3/s"]:
        df[col] = df[col] * df["q_fac"]

        # Fill NaN
        df[col].fillna(value=0, inplace=True)

    # Tidy
    del df["q_fac"], df["q_yr_m3/s"], df["q_lta_m3/s"]

    # 3. Point sources
    # 3.1. Aqu, ren, ind
    # List of data to concat later
    df_list = [
        df,
    ]

    # Set indices
    for pt_df in [aqu_df, ren_df, ind_df]:
        if pt_df is not None:
            pt_df.index = pt_df["regine"]
            del pt_df["regine"]
            df_list.append(pt_df)

    # Join
    df = pd.concat(df_list, axis=1, sort=True)
    df.index.name = "regine"
    df.reset_index(inplace=True)

    # Fill NaN
    for typ in ["aqu", "ren", "ind"]:
        for par in par_list:
            col = "%s_%s_tonnes" % (typ, par)
            if col in df.columns:
                df[col].fillna(value=0, inplace=True)
            else:  # Create cols of zeros
                df[col] = 0

    # 3.2. Spr
    # Get total land area and area of cultivated land in each kommune
    kom_df = df[["komnr", "a_land_km2", "a_agri_km2"]]
    kom_df = kom_df.groupby("komnr").sum()
    kom_df.reset_index(inplace=True)
    kom_df.columns = ["komnr", "a_kom_km2", "a_agri_kom_km2"]

    if spr_df is not None:
        # Join 'spredt' to kommune areas
        kom_df = pd.merge(kom_df, spr_df, how="left", on="komnr")

    else:  # Create cols of zeros
        for par in par_list:
            kom_df["spr_%s_tonnes" % par.lower()] = 0

    # Join back to main df
    df = pd.merge(df, kom_df, how="left", on="komnr")

    # Distribute loads
    for par in par_list:
        # Over agri
        df["spr_agri"] = (
            df["spr_%s_tonnes" % par] * df["a_agri_km2"] / df["a_agri_kom_km2"]
        )

        # Over all area
        df["spr_all"] = df["spr_%s_tonnes" % par] * df["a_land_km2"] / df["a_kom_km2"]

        # Use agri if > 0, else all
        df["spr_%s_tonnes" % par] = np.where(
            df["a_agri_kom_km2"] > 0, df["spr_agri"], df["spr_all"]
        )

    # Delete intermediate cols
    del df["spr_agri"], df["spr_all"]

    # Fill NaN
    df["a_kom_km2"].fillna(value=0, inplace=True)
    df["a_agri_kom_km2"].fillna(value=0, inplace=True)

    for par in par_list:
        # Fill
        df["spr_%s_tonnes" % par].fillna(value=0, inplace=True)

    # 4. Diffuse
    # Loop over pars
    for par in par_list:
        # Background inputs
        # Woodland
        df["wood_%s_tonnes" % par] = (
            df["a_wood_km2"]
            * df["q_sp_m3/s/km2"]
            * df["c_wood_mg/l_%s" % par]
            * 0.0864
            * 365
        )

        # Upland
        df["upland_%s_tonnes" % par] = (
            df["a_upland_km2"]
            * df["q_sp_m3/s/km2"]
            * df["c_upland_mg/l_%s" % par]
            * 0.0864
            * 365
        )

        # Lake
        df["lake_%s_tonnes" % par] = (
            df["a_lake_km2"] * df["c_lake_kg/km2_%s" % par] / 1000
        )

        # Urban
        df["urban_%s_tonnes" % par] = (
            df["a_urban_km2"] * df["c_urban_kg/km2_%s" % par] / 1000
        )

        # Agri from Bioforsk
        # Background
        df["agri_back_%s_tonnes" % par] = (
            df["a_agri_km2"] * df["agri_back_%s_kg/km2" % par] / 1000
        )

        # Point
        df["agri_pt_%s_tonnes" % par] = (
            df["a_agri_km2"] * df["agri_point_%s_kg/km2" % par] / 1000
        )

        # Diffuse
        df["agri_diff_%s_tonnes" % par] = (
            df["a_agri_km2"] * df["agri_diff_%s_kg/km2" % par] / 1000
        )

    # 5. Retention and transmission
    # Join
    df = pd.merge(df, ret_df, how="left", on="regine")

    # Fill NaN
    for par in par_list:
        # Fill NaN
        df["ret_%s" % par].fillna(value=0, inplace=True)

        # Calculate transmission
        df["trans_%s" % par] = 1 - df["ret_%s" % par]

    # 6. Aggregate values
    # Loop over pars
    for par in par_list:
        # All point sources
        df["all_point_%s_tonnes" % par] = (
            df["spr_%s_tonnes" % par]
            + df["aqu_%s_tonnes" % par]
            + df["ren_%s_tonnes" % par]
            + df["ind_%s_tonnes" % par]
            + df["agri_pt_%s_tonnes" % par]
        )

        # Natural diffuse sources
        df["nat_diff_%s_tonnes" % par] = (
            df["wood_%s_tonnes" % par]
            + df["upland_%s_tonnes" % par]
            + df["lake_%s_tonnes" % par]
            + df["agri_back_%s_tonnes" % par]
        )

        # Anthropogenic diffuse sources
        df["anth_diff_%s_tonnes" % par] = (
            df["urban_%s_tonnes" % par] + df["agri_diff_%s_tonnes" % par]
        )

        # All sources
        df["all_sources_%s_tonnes" % par] = (
            df["all_point_%s_tonnes" % par]
            + df["nat_diff_%s_tonnes" % par]
            + df["anth_diff_%s_tonnes" % par]
        )

    # Get cols of interest
    # Basic_cols
    col_list = ["regine", "regine_ned", "a_reg_km2", "runoff_mm/yr", "q_reg_m3/s"]

    # Param specific cols
    #    par_cols = ['trans_%s', 'aqu_%s_tonnes', 'ind_%s_tonnes', 'ren_%s_tonnes',
    #                'spr_%s_tonnes', 'all_point_%s_tonnes', 'nat_diff_%s_tonnes',
    #                'anth_diff_%s_tonnes', 'all_sources_%s_tonnes']

    # Changed 21/11/2018. See e-mail from John Rune received 20/11/2018 at 16.15
    # Now include 'urban' and 'agri_diff' as separate categories
    par_cols = [
        "trans_%s",
        "aqu_%s_tonnes",
        "ind_%s_tonnes",
        "ren_%s_tonnes",
        "spr_%s_tonnes",
        "agri_pt_%s_tonnes",
        "all_point_%s_tonnes",
        "urban_%s_tonnes",
        "agri_diff_%s_tonnes",
        "nat_diff_%s_tonnes",
        "anth_diff_%s_tonnes",
        "all_sources_%s_tonnes",
    ]

    # Build col list
    for name in par_cols:
        for par in par_list:
            # Get col
            col_list.append(name % par)

    # Get cols
    df = df[col_list]

    # Remove rows where regine_ned is null
    df = df.query("regine_ned == regine_ned")

    # Fill Nan
    df.fillna(value=0, inplace=True)

    # 7. Write output
    df.to_csv(out_csv, encoding="utf-8", index=False)

    return df


def read_obs_data(cal_prop, seed=1):
    """ Reads observed data file for 155 RID sites from 1990 to 2016. Joins in
        basic station properties and splits into calibration and validation
        datasets.
    
    Args:
        cal_prop: Float. Between 0 and 1. Fraction of dataset to use for 
                  calibration. The rest is for validation.
        seed:     Int. For repeatability
        
    Returns:
        Tuple (obs_df, cal_df, val_df). obs_df is the entire dataset
    """
    import pandas as pd
    import numpy as np

    # Read obs data
    in_csv = (
        r"C:\Data\James_Work\Staff\Oeyvind_K\Elveovervakingsprogrammet"
        r"\NOPE\NOPE_RID_Calibration_Data\rid_all_obs_loads_flows_1990_2016.csv"
    )
    obs_df = pd.read_csv(in_csv)

    # Drop NaN
    obs_df.dropna(how="any", inplace=True)

    # Read station data
    in_xlsx = (
        r"C:\Data\James_Work\Staff\Oeyvind_K\Elveovervakingsprogrammet"
        r"\Data\RID_Sites_List.xlsx"
    )
    stn_df = pd.read_excel(in_xlsx, sheet_name="RID_All")
    stn_df = stn_df[["station_id", "nve_vassdrag_nr", "rid_group"]]

    # Join vassdrag nrs
    obs_df = pd.merge(obs_df, stn_df, how="left", on="station_id")

    # Split cal and val
    # NB: obs_df.sample randomises the rows, then np.split()
    # divides data into chunks at the desired split points
    cal_df, val_df = np.split(
        obs_df.sample(frac=1, random_state=seed), [int(cal_prop * len(obs_df)),]
    )

    return (obs_df, cal_df, val_df)


def build_input_dict(st_yr, end_yr, par_list):
    """ Build a dictionary of input data for running NOPE in calibration
        mode. Designed to improve performance compared to looping over
        dataframes.
        
    Args:
        st_yr:    Int. Start year of interest
        end_yr:   Int. Start year of interest
        par_list: List. Parameters of interest
        
    Returns:
        Dict with keys ('regine', year) => {'variable':value}
    """
    import pandas as pd
    import os

    # Annual input folder
    nope_fold = (
        r"C:\Data\James_Work\Staff\Oeyvind_K\Elveovervakingsprogrammet"
        r"\NOPE\NOPE_Annual_Inputs"
    )

    # Container for output
    df_list = []

    # Loop over CSV
    for year in range(st_yr, end_yr + 1):
        # Read data
        in_csv = os.path.join(nope_fold, "nope_input_data_%s.csv" % year)
        df = pd.read_csv(in_csv)

        # Add year
        df["year"] = year

        # Add to output
        df_list.append(df)

    # Combine
    df = pd.concat(df_list, axis=0)

    # Add diffuse
    for par in par_list:
        df["all_diff_%s_tonnes" % par] = (
            df["nat_diff_%s_tonnes" % par] + df["anth_diff_%s_tonnes" % par]
        )

    # Get cols of interest
    par_cols = ["trans_%s", "all_point_%s_tonnes", "all_diff_%s_tonnes"]
    cols = ["regine", "year", "q_reg_m3/s",] + [
        i % j for i in par_cols for j in par_list
    ]
    df = df[cols]

    # Convert to dict for speed later
    in_data = df.set_index(["regine", "year"]).T.to_dict()

    return in_data
