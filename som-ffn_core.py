#!/bin/env python3
############################################################################################
####                                                                                    ####
####    NAME         : som-ffn_core.py                                                  ####
####    EDITED       : Daniel Burt              VLIZ        daniel.burt@vliz.be         ####
####                    - Initial feature development and integration with              ####
####                      translation by Soren Berger.                                  ####
####    LAST EDIT    : 21.02.2025                                                       ####
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

##  Define input data for Self-Organising Map
som_input = {
    'mld': '../input-data/mld_clim_v2018.mat',
    'sss': '../input-data/sss_v2018.mat',
    'sst': '../input-data/sst_v2018.mat',
    # 'pressure': './input-data/pres_v2024.nc',
    # 'data_all': '../input-data/atm_co2_grid_v2024.nc',
    'data_all': '../input-data/atm_pco2_grid_v2018.mat'
    # 'chl': './input-data/chl_v2024.mat'
    # 'data_taka': './input-data/Taka_pCO2_eth_v2024.mat'
} 

# som_input = {
#     'mld': './input-data/mld_clim_v2024.mat',
#     'sss': './input-data/sss_v2024.mat',
#     'sst': './input-data/sst_v2024.mat',
#     'pressure': './input-data/pres_v2024.nc',
#     'data_all': './input-data/atm_co2_grid_v2024.nc',
#     'data_all': './input-data/atm_pco2_grid_v2024.nc'
#     # 'chl': './input-data/chl_v2024.mat'
#     # 'data_taka': './input-data/Taka_pCO2_eth_v2024.mat'
# } 

# ##  Call Functions for Feed Forward Network Operation
# som.LoadInputData(som_input)
# som.CalculateMeanMonths()
# som.PlotInputsMonthly()  # optional
# som.ReshapeRearrange()
# som.IdentifyProvinces(som_sigma = 1.75, som_learning_rate = 1.0, number_of_epochs = 20000)  # values upwards of 200000 work best
# # som.LoadComparisonProvinces()  # optional TODO
# som.PlotProvinces(plot_type = 'mode-variability')  # optional TODO expand with additional visualisation options
# som.WriteProvinces(fileext = 'nc')  # optional


####  STEP 2: Feed Forward Network

##  Define input data for Feed Forward Network
ffn_input = {
    'mld': '../input-data/mld_clim_v2018.mat',
    'sss': '../input-data/sss_v2018.mat',
    'sst': '../input-data/sst_v2018.mat',
    'provinces': '../input-data/provinces.nc',
    'fco2_ave_weighted': '../input-data/pco2obsV2018.mat'
} 

# ffn_input = {
#     'mld': '../input-data/mld_clim_v2024.mat',
#     'sss': '../input-data/sss_v2024.nc',
#     'sst': '../input-data/sst_v2024.mat',
#     'provinces': '../input-data/provinces.nc',
#     'fco2_ave_weighted': '../input-data/SOCATv2024_tracks_gridded_monthly.nc'
# } 


##  Call Functions for Feed Forward Network Operation
ffn.LoadInputData(ffn_input)
ffn.CropInputData()
# ffn.PlotInputData()
ffn.PrepareInputs()
# ffn.MakeTrainModel(length_patience = 10)
ffn.MakeTrainModelLoop(length_patience = 10)
# ffn.PlotDiagnostic()  # FIXME not tested with "loop" approach
# ffn.PredictEstimate()
ffn.PredictEstimateLoop() 
# ffn.PlotPrediction(plot_type = 'mean-comparison')
