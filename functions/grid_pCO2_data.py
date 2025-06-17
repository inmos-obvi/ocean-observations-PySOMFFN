
import numpy as np
import pandas as pd

def grid_pCO2_data(pco2_df_filt, socat, t_i):

    pco2_float_gridded = np.zeros((len(socat['ylat']), len(socat['xlon'])))
    pco2_float_gridded[:] = np.NaN
    pco2_float_adjusted_gridded = np.zeros((len(socat['ylat']), len(socat['xlon'])))
    pco2_float_adjusted_gridded[:] = np.NaN    
    
    time_index = np.logical_and(pco2_df_filt['JULD']>=socat.tmnth_bnds[t_i][0].values, pco2_df_filt['JULD']<socat.tmnth_bnds[t_i][1].values)

    # grid data if it exists, otherwise save an empty month file 
    if len(pco2_df_filt[time_index])>0:
        # print(socat.tmnth_bnds[t_i].values)
        # grid the float data to the same grid as socat
        for x_i in range(0, len(socat['xlon'])):
            for y_i in range(13, 166):
                # print(socat['xlon'][x_i].values)
                lon_index = np.logical_and(pco2_df_filt['LONGITUDE']>=socat['xlon'][x_i].values-0.5, pco2_df_filt['LONGITUDE']<socat['xlon'][x_i].values+0.5)
                lat_index = np.logical_and(pco2_df_filt['LATITUDE']>=socat['ylat'][y_i].values-0.5, pco2_df_filt['LATITUDE']<socat['ylat'][y_i].values+0.5)

                grid_index = np.logical_and(lat_index,lon_index)
                grid_index = np.logical_and(grid_index,time_index)
                if len(pco2_df_filt[grid_index]['Float pCO2'])>0:
                    pco2_float_gridded[y_i, x_i] = np.nanmean(pco2_df_filt[grid_index]['Float pCO2'])
                    pco2_float_adjusted_gridded[y_i, x_i] = np.nanmean(pco2_df_filt[grid_index]['float_pco2_chosen'])
    
    # save monthly files
    month_pd = pd.to_datetime(socat.tmnth_bnds[t_i][0].values)
    base_filename = '../intermediate_data/' + str(month_pd.year) + '_' + str(month_pd.month)

    np.save(base_filename + '_pco2_float_gridded', pco2_float_gridded, allow_pickle=False)
    np.save(base_filename + '_pco2_float_adjusted_gridded', pco2_float_adjusted_gridded, allow_pickle=False)

    return