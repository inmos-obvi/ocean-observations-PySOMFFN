#!/bin/env python3
############################################################################################
####                                                                                    ####
####    NAME         : som-ffn_core.py                                                  ####
####    EDITED       : Daniel Burt              VLIZ        daniel.burt@vliz.be         ####
####                    - Initial feature development and integration with              ####
####                      translation by Soren Berger.                                  ####
####    LAST EDIT    : 11.02.2026                                                       ####
####    DESCRIPTION  : Core file for running Self-Organising Map - Feed Forward         ####
####                   Network (SOM-FFN) method based on the MATLAB implementation      ####
####                   of Peter Landschuetzer and originally described in:              ####
####                    -  Landschuetzer et al. (2013) Biogeosciences                   ####
####                                                                                    ####
####                   This Python implementation is under development within the       ####
####                   Past, Present and Future Marine Climate Change Group of the      ####
####                   Flanders Marine Institute (VLIZ), Belgium.                       ####
####                                                                                    ####
####                   Call distributed functions from related function files.          ####
####                   FFN function file is handled as a plug-in with simultaneous      ####
####                   development by the PPFCC group at VLIZ.                          ####
####                                                                                    ####
####                   STEP 1: Self-Organizing Map                                      ####
####                                                                                    ####
####                   STEP 2: Feed Forward Network                                     ####
####                                                                                    ####
####    DEPENDENCIES : Python 3.12.3                                                    ####
####                                                                                    ####
####                                                                                    ####
############################################################################################



####  IMPORT FUNCTIONS
from selforganizingmap import SelfOrganizingMap
from feedforwardnetwork import FeedForwardNetwork


####  DEFINE CLASSES
som = SelfOrganizingMap()
ffn = FeedForwardNetwork()



####  EXECUTE SOM-FFN ALGORITHM  ===========================================================

####  STEP 1: Self-Organizing Map

# ##  Define input data for Self-Organising Map
# som_input = {
#     'mld': '/work/bg1446/m300722/machine-learning/model-code/python_som-ffn_3d/input-data/mld_clim_v2024.mat',
#     'sss': '/work/bg1446/m300722/machine-learning/model-code/python_som-ffn_3d/input-data/sss_v2024.mat',
#     'sst': '/work/bg1446/m300722/machine-learning/model-code/python_som-ffn_3d/input-data/sst_v2024.mat',
#     'pressure': '/work/bg1446/m300722/machine-learning/model-code/python_som-ffn_3d/input-data/pres_v2024.nc',
#     'data_all': '/work/bg1446/m300722/machine-learning/model-code/python_som-ffn_3d/input-data/atm_co2_grid_v2024.nc',
#     'data_all': '/work/bg1446/m300722/machine-learning/model-code/python_som-ffn_3d/input-data/atm_pco2_grid_v2024.nc'
#     # 'chl': '/work/bg1446/m300722/machine-learning/model-code/python_som-ffn_3d/input-data/chl_v2024.mat'
#     # 'data_taka': '/work/bg1446/m300722/machine-learning/model-code/python_som-ffn_3d/input-data/Taka_pCO2_eth_v2024.mat'
# } 

# ##  Call Functions for Feed Forward Network Operation
# som.LoadInputData(som_input)
# som.CalculateMeanMonths()
# # som.PlotInputsMonthly()  # optional
# som.ReshapeRearrange()
# som.IdentifyProvinces(som_sigma = 1.75, som_learning_rate = 1.0, number_of_epochs = 1000000)  # values upwards of 200000 work best
# # som.LoadComparisonProvinces()  # optional TODO
# # som.PlotProvinces(plot_type = 'mode-variability')  # optional TODO expand with additional visualisation options
# som.WriteProvinces(fpath_output = '/work/bg1446/m300722/machine-learning/model-code/python_som-ffn_3d/input-data', fileext = 'nc')  # optional


####  STEP 2: Feed Forward Network

##  Define input data for Feed Forward Network
ffn_input = {
    'mld': '/work/bg1446/m300722/machine-learning/model-code/python_som-ffn_3d/input-data/mld_clim_v2024.mat',
    'sss': '/work/bg1446/m300722/machine-learning/model-code/python_som-ffn_3d/input-data/sss_v2024.mat',
    'sst': '/work/bg1446/m300722/machine-learning/model-code/python_som-ffn_3d/input-data/sst_v2024.mat',
    # 'provinces': '/work/bg1446/m300722/machine-learning/model-code/python_som-ffn_3d/input-data/som-output_provinces.nc',  # disabled for cross-val-01
    'fco2_ave_weighted': '/work/bg1446/m300722/machine-learning/model-code/python_som-ffn_3d/input-data/SOCATv2024_tracks_gridded_monthly.nc'
} 

##  Call Functions for Feed Forward Network Operation
ffn.LoadInputData(ffn_input)
ffn.CropInputData()
ffn.PrepareInputs()
ffn.PrepareFolds(n_folds               = 5, 
                 intermediate_dir_path = '/work/bg1446/m300722/machine-learning/intermediate-data/3d_pco2_cross-val-01')
ffn.OptimiseHyperparameters(n_trials       = 50, 
                            n_folds        = 5,
                            fpath_database = '/work/bg1446/m300722/machine-learning/intermediate-data/3d_pco2_cross-val-01')
