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
    
def estimate_loads(stn_id, par_list, year, engine):
    """ Estimates annual pollutant loads for specified site and year.
        
    Args:
        stn_id:    Int. Valid RESA2 STATION_ID
        par_list:  List of valid RESA2 PARAMETER_NAMES
        year:      Int. Year of interest
        engine:    SQL-Alchemy 'engine' object already connected to RESA2
    
    Returns:
        Dataframe of annual loads
    """
    import pandas as pd
    import numpy as np
    
    # Dict with factors for unit conversions
    unit_dict = {'SPM':[1.E9, 'tonnes'],     # mg to tonnes
                 'TOC':[1.E9, 'tonnes'],     # mg to tonnes
                 'PO4-P':[1.E12, 'tonnes'],  # ug to tonnes
                 'TOTP':[1.E12, 'tonnes'],   # ug to tonnes
                 'NO3-N':[1.E12, 'tonnes'],  # ug to tonnes
                 'NH4-N':[1.E12, 'tonnes'],  # ug to tonnes
                 'TOTN':[1.E12, 'tonnes'],   # ug to tonnes
                 'SiO2':[1.E9, 'tonnes'],    # mg to tonnes
                 'Ag':[1.E12, 'tonnes'],     # ug to tonnes
                 'As':[1.E12, 'tonnes'],     # ug to tonnes
                 'Pb':[1.E12, 'tonnes'],     # ug to tonnes
                 'Cd':[1.E12, 'tonnes'],     # ug to tonnes
                 'Cu':[1.E12, 'tonnes'],     # ug to tonnes
                 'Zn':[1.E12, 'tonnes'],     # ug to tonnes
                 'Ni':[1.E12, 'tonnes'],     # ug to tonnes
                 'Cr':[1.E12, 'tonnes'],     # ug to tonnes
                 'Hg':[1.E12, 'kg']}         # ng to kg
    
    # Get water chem data
    wc_df = extract_water_chem(stn_id, par_list, 
                               '%s-01-01' % year,
                               '%s-12-31' % year,
                                engine, plot=False)
    
    # Get flow data
    q_df = extract_discharge(stn_id, 
                             '%s-01-01' % year, 
                             '%s-12-31' % year,
                             engine, plot=False)
    
    # Tidy dfs
    wc_df.index.name = 'datetime'
    wc_df['date'] = wc_df.index.date
    wc_df.reset_index(inplace=True)    
    q_df['date'] = q_df.index.date
    
    # Join on date
    wc_df = pd.merge(wc_df, q_df, how='left', on='date')
    
    # Get list of chem cols
    cols = wc_df.columns
    par_unit_list = [i for i in cols if not i in ['datetime', 'date', 'flow_m3/s']]

    # Calc intervals t_i
    # Get sample dates
    dates = list(wc_df['datetime'].values)

    # Add st and end of year
    dates.insert(0, np.datetime64('%s-01-01' % year))
    dates.append(np.datetime64('%s-12-31' % year))
    dates = np.array(dates)

    # Calc differences in seconds between dates and divide by 2
    secs = (np.diff(dates) / np.timedelta64(1, 's')) / 2.

    # Add "before" and "after" intervals to df
    wc_df['t_i'] = secs[1:] + secs[:-1]
    
    # Estimate loads
    # Denominator
    wc_df['Qi_ti'] = wc_df['flow_m3/s']*wc_df['t_i']
    
    # Numerator
    for par in par_unit_list:
        par_l = par.split('/')[0]
        wc_df[par_l] = 1000*wc_df['flow_m3/s']*wc_df['t_i']*wc_df[par]
    
    # Flow totals
    sum_df = wc_df.sum()
    sigma_Qi_ti = sum_df['Qi_ti']                  # Denominator
    sigma_q = (q_df['flow_m3/s']*60*60*24).sum()   # Q_r
    
    # Chem totals
    data_dict = {}
    for par in par_unit_list:
        # Identify variable
        par_l = par.split('/')[0]
        var_name = par_l.split('_')[0]
        
        # Get unit and conv factor
        conv_fac = unit_dict[var_name][0]
        unit = unit_dict[var_name][1]
        data_dict[var_name+'_'+unit] = (sigma_q*sum_df[par_l])/(conv_fac*sigma_Qi_ti)
    
    # Convert to df
    l_df = pd.DataFrame(data_dict, index=[stn_id])
    
    return l_df