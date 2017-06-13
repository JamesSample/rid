#------------------------------------------------------------------------------
# Name:        useful_rid_code.py
# Purpose:     Functions for updated RID analysis.
#
# Author:      James Sample
#
# Created:     13/06/2017
# Copyright:   (c) James Sample and NIVA   
#------------------------------------------------------------------------------
""" This file contains useful functions for the updated RID analysis. Much of
    this code is based on 
    
    C:\Data\James_Work\Staff\Heleen_d_W\ICP_Waters\Upload_Template\useful_resa2_code.py
"""

def extract_water_chem(stn_id, par_list, st_dt, end_dt, engine, plot=False):
    """ Extracts time series for the specified station and parameters. Returns 
        a dataframe and an interactive grid plot.

    Args:
        stn_id:    Int. Valid RESA2 STATION_ID
        par_list:  List of valid RESA2 PARAMETER_NAMES
        st_dt:     Str. Start date as 'yyyy-mm-dd'
        end_dt:    Str. End date as 'yyyy-mm-dd'
        engine:    SQL-Alchemy 'engine' object already connected to RESA2
        plot:      Bool. Choose whether to return a grid plot as well as the 
                   dataframe

    Returns:
        If plot is False, returns a dataframe of water chemsitry, otherwise
        returns a tuple (wc_df, figure object)
    """
    import matplotlib.pyplot as plt 
    import datetime as dt
    import mpld3
    import pandas as pd
    import seaborn as sn

    # Check stn is valid
    sql = ('SELECT * FROM resa2.stations '
           'WHERE station_id = %s' % stn_id)   

    stn_df = pd.read_sql_query(sql, engine)
    
    assert len(stn_df) == 1, 'Station code not found.' 
    
    # Get station_code
    stn_code = stn_df['station_code'].iloc[0]
    
    # Check pars are valid
    if len(par_list)==1:  
        sql = ("SELECT * FROM resa2.parameter_definitions "
               "WHERE name = '%s'" % par_list[0])
    else:
        sql = ('SELECT * FROM resa2.parameter_definitions '
               'WHERE name in %s' % str(tuple(par_list)))

    par_df = pd.read_sql_query(sql, engine)
    
    assert len(par_df) == len(par_list), 'One or more parameters not found.'
    
    # Get results for ALL pars for sites and period of interest
    sql = ("SELECT * FROM resa2.water_chemistry_values2 "
           "WHERE sample_id IN (SELECT water_sample_id FROM resa2.water_samples "
           "WHERE station_id = %s "
           "AND sample_date <= DATE '%s' "
           "AND sample_date >= DATE '%s')" % (stn_df['station_id'].iloc[0],
                                              end_dt,
                                              st_dt))    

    wc_df = pd.read_sql_query(sql, engine)
    
    # Get all sample dates for sites and period of interest
    sql = ("SELECT water_sample_id, station_id, sample_date "
           "FROM resa2.water_samples "
           "WHERE station_id = %s "
           "AND sample_date <= DATE '%s' "
           "AND sample_date >= DATE '%s'" % (stn_df['station_id'].iloc[0],
                                             end_dt,
                                             st_dt))    

    samp_df = pd.read_sql_query(sql, engine)
    
    
    # Join in par IDs based on method IDs
    sql = ('SELECT * FROM resa2.wc_parameters_methods')
    meth_par_df = pd.read_sql_query(sql, engine)
    
    wc_df = pd.merge(wc_df, meth_par_df, how='left',
                     left_on='method_id', right_on='wc_method_id')
    
    # Get just the parameters of interest
    wc_df = wc_df.query('wc_parameter_id in %s' % str(tuple(par_df['parameter_id'].values)))
    
    # Join in sample dates
    wc_df = pd.merge(wc_df, samp_df, how='left',
                     left_on='sample_id', right_on='water_sample_id')
    
    # Join in parameter units
    sql = ('SELECT * FROM resa2.parameter_definitions')
    all_par_df = pd.read_sql_query(sql, engine)
    
    wc_df = pd.merge(wc_df, all_par_df, how='left',
                     left_on='wc_parameter_id', right_on='parameter_id')
    
    # Join in station codes
    wc_df = pd.merge(wc_df, stn_df, how='left',
                     left_on='station_id', right_on='station_id')
    
    # Convert units
    wc_df['value'] = wc_df['value'] * wc_df['conversion_factor']
    
    # Combine name and unit columns
    wc_df['variable'] = wc_df['name'] + '_' + wc_df['unit'].map(str)
    
    # Extract columns of interest
    wc_df = wc_df[['station_code', 'sample_date', 'name', 'unit', 'value']]
    
    if plot:
        # Get number of rows
        if (len(par_list) % 3) == 0:
            rows = int(len(par_list) / 3)
        else:
            rows = int(len(par_list) / 3) + 1
        
        # Setup plot grid
        fig, axes = plt.subplots(nrows=rows, ncols=3,
                                 sharex=False, sharey=False, 
                                 figsize=(12, rows*2.5),
                                 squeeze=False)
        axes = axes.flatten()

        # Loop over series
        idx = 0
        for pid, par in enumerate(par_list):
            # Extract series
            ts = wc_df.query('(station_code==@stn_code) & (name==@par)')

            # Get unit
            unit = ts['unit'].iloc[0]

            # Date index and sort
            ts = ts[['sample_date', 'value']]
            ts.index = ts['sample_date']
            del ts['sample_date']
            ts.sort_index(inplace=True)

            # Plot
            ts.plot(ax=axes[idx], legend=False)
            axes[idx].set_title('%s at %s' % (par, 
                                              stn_code.decode('windows-1252')), 
                                fontsize=16)
            axes[idx].set_xlabel('')
            if unit:
                axes[idx].set_ylabel('%s (%s)' % (par, 
                                                  unit.decode('windows-1252')),
                                     fontsize=14)
            else:
                axes[idx].set_ylabel(par, fontsize=14)
            
            # Set x-limits
            axes[idx].set_xlim([dt.datetime.strptime(st_dt, '%Y-%m-%d').date(),
                                dt.datetime.strptime(end_dt, '%Y-%m-%d').date()]) 
            
            idx += 1

        # Delete unwanted axes
        for idx, ax in enumerate(axes):
            if (idx + 1) > len(par_list):
                fig.delaxes(axes[idx])
        
        # Tidy
        plt.tight_layout()
        
    # Restructure df
    wc_df['unit'] = wc_df['unit'].str.decode('windows-1252')
    wc_df['par'] = wc_df['name'] + '_' + wc_df['unit'].map(unicode)
    del wc_df['station_code'], wc_df['name'], wc_df['unit']
    
    wc_df = wc_df.pivot(index='sample_date', columns='par', values='value')
    
    if plot:
        return (wc_df, fig)
    else:
        return wc_df

def extract_discharge(stn_id, st_dt, end_dt, engine, plot=False):
    """ Extracts daily flow time series for the selected site. Returns a 
        dataframe and, optionally, a plot object
        
    Args:
        stn_id:    Int. Valid RESA2 STATION_ID
        st_dt:     Str. Start date as 'yyyy-mm-dd'
        end_dt:    Str. End date as 'yyyy-mm-dd'
        engine:    SQL-Alchemy 'engine' object already connected to RESA2
        plot:      Bool. Choose whether to return a grid plot as well as the 
                   dataframe
    
    Returns:
        If plot is False, returns a dataframe of flows, otherwise
        returns a tuple (q_df, figure object)
    """
    import matplotlib.pyplot as plt 
    import datetime as dt
    import mpld3
    import pandas as pd
    import seaborn as sn

    # Check stn is valid
    sql = ("SELECT * FROM resa2.stations "
           "WHERE station_id = %s" % stn_id)
    stn_df = pd.read_sql_query(sql, engine)
    
    assert len(stn_df) == 1, 'Error in station code.' 

    # Get station_code
    stn_code = stn_df['station_code'].iloc[0]
    
    # Check a discharge station is defined for this WC station
    sql = ("SELECT * FROM resa2.default_dis_stations "
           "WHERE station_id = '%s'" % stn_df['station_id'].iloc[0])
    dis_df = pd.read_sql_query(sql, engine)
    
    assert len(dis_df) == 1, 'Error identifying discharge station.'
    
    # Get ID for discharge station
    dis_stn_id = dis_df['dis_station_id'].iloc[0]

    # Get catchment areas
    # Discharge station
    sql = ("SELECT area FROM resa2.discharge_stations "
           "WHERE dis_station_id = %s" % dis_stn_id)
    area_df = pd.read_sql_query(sql, engine)    
    dis_area = area_df['area'].iloc[0]

    # Chemistry station
    sql = ("SELECT catchment_area FROM resa2.stations "
           "WHERE station_id = %s" % stn_df['station_id'].iloc[0])
    area_df = pd.read_sql_query(sql, engine)    
    wc_area = area_df['catchment_area'].iloc[0]
    
    # Get the daily discharge data for this station
    sql = ("SELECT xdate, xvalue FROM resa2.discharge_values "
           "WHERE dis_station_id = %s "
           "AND xdate >= DATE '%s' "
           "AND xdate <= DATE '%s'" % (dis_stn_id,
                                       st_dt,
                                       end_dt))
    q_df = pd.read_sql_query(sql, engine)
    q_df.columns = ['date', 'flow_m3/s']
    q_df.index = q_df['date']
    del q_df['date']
    
    # Scale flows by area
    q_df = q_df*wc_area/dis_area
    
    # Convert to daily
    q_df = q_df.resample('D').mean()
    
    # Plot
    if plot:
        ax = q_df.plot(legend=False, figsize=(12,5))
        ax.set_xlabel('')
        ax.set_ylabel('Flow (m3/s)', fontsize=16)
        ax.set_title('Discharge at %s' % stn_code.decode('windows-1252'),
                     fontsize=16)
        plt.tight_layout()
        
        return (q_df, ax)
    
    else:    
        return q_df