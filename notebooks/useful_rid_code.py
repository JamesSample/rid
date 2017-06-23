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
        a dataframe of chemistry and LOD values, a dataframe of duplicates and,
        optionally, an interactive grid plot.

    Args:
        stn_id:    Int. Valid RESA2 STATION_ID
        par_list:  List of valid RESA2 PARAMETER_NAMES
        st_dt:     Str. Start date as 'yyyy-mm-dd'
        end_dt:    Str. End date as 'yyyy-mm-dd'
        engine:    SQL-Alchemy 'engine' object already connected to RESA2
        plot:      Bool. Choose whether to return a grid plot as well as the 
                   dataframe

    Returns:
        If plot is False, returns (chem_df, dup_df), otherwise
        returns (chem_df, dup_df, fig_obj)
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
    
    # Extract columns of interest
    wc_df = wc_df[['station_code', 'sample_date', 'name', 'unit', 
                   'value', 'flag1', 'entered_date_x']]

    # Check for duplicates
    dup_df = wc_df[wc_df.duplicated(subset=['station_code',
                                            'sample_date',
                                            'name'], 
                                            keep=False)].sort_values(by=['station_code', 
                                                                         'sample_date', 
                                                                         'name'])
    
    dup_df['station_code'] = dup_df['station_code'].str.decode('windows-1252')
    
    if len(dup_df) > 0:
        print ('WARNING\nThe database contains duplicated values for some station-'
               'date-parameter combinations.\nOnly the most recent values '
               'will be used, but you should check the repeated values are not '
               'errors.\nThe duplicated entries are returned in a separate '
               'dataframe.\n')
        
        # Choose most recent record for each duplicate
        wc_df.sort_values(by='entered_date_x', inplace=True, ascending=True)

        # Drop duplicates
        wc_df.drop_duplicates(subset=['station_code', 'sample_date', 'name'],
                              keep='last', inplace=True)
        
        # Tidy
        del wc_df['entered_date_x']               
        wc_df.reset_index(inplace=True, drop=True)
        
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
    wc_df['flag'] = wc_df['name'] + '_flag'
    del wc_df['station_code'], wc_df['name'], wc_df['unit']

    # DF of values
    wc_val_df = wc_df[['sample_date', 'par', 'value']]
    wc_val_df = wc_val_df.pivot(index='sample_date', columns='par', values='value')

    # DF of LOD flags
    wc_lod_df = wc_df[['sample_date', 'flag', 'flag1']]
    wc_lod_df = wc_lod_df.pivot(index='sample_date', columns='flag', values='flag1')

    # Join
    wc_df = pd.merge(wc_val_df, wc_lod_df, how='left',
                     left_index=True, right_index=True)
    
    # Sort
    wc_df.sort_index(inplace=True, ascending=True)
    
    if plot:
        return (wc_df, dup_df, fig)
    else:
        return (wc_df, dup_df)

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
    
    # Linear interpolation of NoData gaps
    q_df.interpolate(method='linear', inplace=True)
    
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
    wc_df, dup_df = extract_water_chem(stn_id, par_list, 
                                       '%s-01-01' % year,
                                       '%s-12-31' % year,
                                       engine, plot=False)
    
    # Get flow data
    q_df = extract_discharge(stn_id, 
                             '%s-01-01' % year, 
                             '%s-12-31' % year,
                             engine, plot=False)
    
    # Adjust LOD values
    # Get list of chem cols
    cols = wc_df.columns
    par_unit_list = [i for i in cols if i.split('_')[1] != 'flag']

    # loop over cols
    for par_unit in par_unit_list:
        par = par_unit.split('_')[0]

        # Get vals
        val_df = wc_df[par_unit].values

        # Get LOD flags
        lod_df = wc_df[par+'_flag'].values

        # Number of LOD values
        n_lod = (lod_df == '<').sum()

        # Prop <= LOD
        p_lod = (100.*n_lod)/len(lod_df)

        # Adjust ALL values
        ad_val_df = (val_df / 100.)*(100. - p_lod)

        # Update ONLY the LOD values
        val_df[(lod_df=='<')] = ad_val_df[(lod_df=='<')]

        # Update values in wc_df
        wc_df[par_unit] = val_df
    
    # Tidy dfs
    wc_df.index.name = 'datetime'
    wc_df['date'] = wc_df.index.date
    wc_df.reset_index(inplace=True)    
    q_df['date'] = q_df.index.date
    
    # Join on date
    wc_df = pd.merge(wc_df, q_df, how='left', on='date')
    
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

def remove_row(table, row):
    """ Function to remove a row from a Word table. See:
        https://groups.google.com/forum/#!topic/python-docx/aDumlNvK6GM
        
    Args:
        table: Table object
        row:   Row object
    
    Returns:
        None, The table object is modified in-place.
    """
    tbl = table._tbl
    tr = row._tr
    tbl.remove(tr)
    
def update_cell(row_txt, par_txt, value,
                col_dict, row_dict, tab):
    """ Update a cell in a table specified by row and col names.
    
    Args:
        row_txt:  Str. Label to identify row of interest
        col_txt:  Str. Label to identify col of interest
        value:    New value for cell
        col_dict: Dict indexing columns
        row_dict: Dict indexing rows
        tab:      Table object 
    
    Returns:
        None. The table object is modified in-place.
    """
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    
    # Get row and col indices
    col = col_dict[par_txt]
    row = row_dict[row_txt]

    # Get cell
    cell = tab.cell(row, col)

    # Modify value
    if isinstance(value, float) or isinstance(value, int):
        cell.text = '%.2f' % value  
    elif isinstance(value, str):
        cell.text = value
    else:
        raise ValueError('Unexpected data type.')

    # Align right
    p = tab.cell(row, col).paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT

def write_word_water_chem_tables(stn_df, year, in_docx, engine):
    """ Creates Word tables summarising flow and water chemistry data
        for the RID_11 and RID_36 stations. Replaces Tore's 
        NIVA.RESAEXTENSIONS Visual Studio project.
    
    Args:
        stn_df:  Dataframe listing the 47 sites of interest. Must
                 include [station_id, station_code, station_name]
        year:    Int. Year of interest
        in_docx: Str. Raw path to Word document. This should be a
                 *COPY* of rid_water_chem_tables_template.docx. Do
                 not use the original template as the files will be 
                 modified
        engine:  SQL-Alchemy 'engine' object already connected to RESA2
        
    Returns:
        None. The specified Word document is modified and saved.
    """
    import pandas as pd
    import numpy as np
    from docx import Document
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    # Chem pars of interest
    par_list = ['pH', 'KOND', 'TURB860', 'SPM', 'TOC', 'PO4-P', 
                'TOTP', 'NO3-N', 'NH4-N', 'TOTN', 'SiO2', 'Ag', 
                'As', 'Pb', 'Cd', 'Cu', 'Zn', 'Ni', 'Cr', 'Hg']

    # Open the document
    doc = Document(in_docx)

    # Set styles for 'Normal' template in this doc
    style = doc.styles['Normal']

    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(8)

    p_format = style.paragraph_format
    p_format.space_before = Pt(0)
    p_format.space_after = Pt(0)

    # Loop over tables
    for tab_id, tab in enumerate(doc.tables):
        # Get station name
        stn_name = tab.cell(0, 0).text

        print 'Processing:', stn_name

        # Get station ID
        stn_id = stn_df.query('station_name == @stn_name')['station_id'].values[0]

        print '    Extracting water chemistry data...'

        # Get WC data
        wc_df, dup_df  = extract_water_chem(stn_id, par_list, 
                                            '%s-01-01' % year,
                                            '%s-12-31' % year,
                                            engine, plot=False)

        # Get list of cols of interest for later
        cols = wc_df.columns
        par_unit_list = [i for i in cols if i.split('_')[1] != 'flag']
        par_unit_list.append('Qs_m3/s')

        # Add date col (ignoring time)
        wc_df['date'] = wc_df.index.date

        # Reset index
        wc_df.reset_index(inplace=True)

        print '    Extracting flow data...'

        # Get flow data
        q_df = extract_discharge(stn_id, 
                                 '%s-01-01' % year,
                                 '%s-12-31' % year,
                                 engine, plot=False)

        # Add date col (ignoring time)
        q_df['date'] = q_df.index.date

        # Join flows to chem
        df = pd.merge(wc_df, q_df, how='left', on='date')

        # Set index
        df.index = df['sample_date']

        # Tidy
        df['Qs_m3/s'] = df['flow_m3/s']
        del df['date'], df['flow_m3/s'], df['sample_date']
        df.sort_index(inplace=True)

        print '    Writing sample dates...'

        # Write sample dates to first col
        dates = df.index.values

        for idx, dt in enumerate(dates):
            # Get cell
            cell = tab.cell(idx+3, 0) # Skip 3 rows of header

            # Modify value
            cell.text = pd.to_datetime(str(dt)).strftime('%d.%m.%Y %H:%M')

            # Align right
            p = tab.cell(idx+3, 0).paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT 

        print '    Deleting empty rows...'

        # Delete empty rows (each blank table has space for 20 samples, but 
        # usually there are fewer)
        # Work out how many rows to delete
        first_blank = len(dates)+3
        n_blank = 23 - len(dates) - 3
        blank_ids = [first_blank,]*n_blank # Delete first blank row n times 

        for idx in blank_ids:
            # Delete rows
            row = tab.rows[idx]
            remove_row(tab, row)     

        print '    Writing data values...'

        # Extract text to index rows
        row_dict = {}
        for idx, cell in enumerate(tab.column_cells(0)):
            for paragraph in cell.paragraphs:
                row_dict[paragraph.text] = idx 

        # Extract text to index cols
        col_dict = {}
        for idx, cell in enumerate(tab.row_cells(1)):
            for paragraph in cell.paragraphs:
                col_dict[paragraph.text] = idx 

        # Loop over df cols
        for idx, df_row in df.iterrows():
            # Get the date
            dt_tm = idx.strftime('%d.%m.%Y %H:%M')

            # Loop over variables
            for par_unit in par_unit_list:
                # Get just the par
                par = par_unit.split('_')[0]

                # Update the table
                update_cell(dt_tm, par, df_row[par_unit],
                            col_dict, row_dict, tab)

        print '    Writing summary statistics...'

        # Add flag col (all None) for Qs
        df['Qs_flag'] = None

        # Loop over cols
        for df_col in par_unit_list:
            # Get just the par
            par = df_col.split('_')[0]

            # Calc statistics
            # 1. Lower av. - assume all LOD values are 0
            # Get vals
            val_df = df[df_col].values

            # Get LOD flags
            lod_df = df[par+'_flag'].values

            # Update ONLY the LOD values
            val_df[(lod_df=='<')] = 0

            # Average
            lo_av = val_df.mean()
            update_cell('Lower avg.', par, lo_av,
                        col_dict, row_dict, tab)   

            # 2. Upper av. - assume all LOD values are LOD
            up_av = df[df_col].mean()
            update_cell('Upper avg..', par, up_av,
                        col_dict, row_dict, tab)

            # 3. Min
            par_min = df[df_col].min()
            update_cell('Minimum', par, par_min,
                        col_dict, row_dict, tab)

            # 4. Max
            par_max = df[df_col].max()
            update_cell('Maximum', par, par_max,
                        col_dict, row_dict, tab)

            # 5. More than 70% above LOD?
            lod_df = df[par+'_flag'].values
            pct_lod = (100.*(lod_df=='<').sum())/len(lod_df) # % at or below LOD
            pct_lod = 100 - pct_lod                          # % above LOD
            if pct_lod > 70:
                lod_gt_70 = 'yes'
            else:
                lod_gt_70 = 'no'
            update_cell('More than 70%LOD', par, lod_gt_70,
                        col_dict, row_dict, tab)

            # 6. n samples
            n_samps = len(df[[df_col]].dropna(how='any'))
            update_cell('n', par, n_samps,
                        col_dict, row_dict, tab)

            # 7. Std. Dev.
            par_std = df[df_col].std() 
            update_cell('St.dev', par, par_std,
                        col_dict, row_dict, tab)

        print '    Done.'

        # Save after each table
        doc.save(in_docx)

    print 'Finished.'