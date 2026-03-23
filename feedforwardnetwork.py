############################################################################################
####                                                                                    ####
####    NAME         : feedforwardnetwork.py                                            ####
####    EDITED       : Peter Landschuetzer      UEA     peter.landschuetzer@vliz.be     ####
####                    - Initial development and formulation in MATLAB                 ####
####                   Soren Berger             MPI-M       soren.berger@mpimet.mpg.de  ####
####                    - Initial translation from MATLAB to Python                     #### 
####                   Andrea van Langen Roson  VLIZ        andrea.van.langen@vliz.be   ####
####                   Maurie Keppens           VLIZ        maurie.keppens@vliz.be      ####
####                   Arianna Olivelli         VLIZ        arianna.olivelli@vliz.be    ####
####                   Daniel Burt              VLIZ        daniel.burt@vliz.be         ####
####                    - Refactoring of Python code                                    ####
####                    - Adding new functions                                          ####
####                    - Adaptation of Optuna Optimisation for TensorFlow KERAS        ####
####    LAST EDIT    : 11.02.2026                                                       ####
####    DESCRIPTION  : This Python implementation is under development within the       ####
####                   Past, Present and Future Marine Climate Change Group of the      ####
####                   Flanders Marine Institute (VLIZ), Belgium.                       ####
####                                                                                    ####
####                   Class file for running Self-Organising Map component of          ####
####                   SOM-FFN method based on the MATLAB implementation of Peter       ####
####                   Landschuetzer and originally described in:                       ####
####                    -  Landschuetzer et al. (2013) Biogeosciences                   ####
####                                                                                    ####
####    DEPENDENCIES : Python 3.12.3                                                    ####
####                    - CartoPy 0.22.0                                                ####
####                    - MatPlotLib 3.6.3                                              ####
####                    - NumPy 1.26.4                                                  ####
####                    - Optuna 4.7.0  [with optuna-integration]                       ####
####                    - SciPy 1.11.4                                                  ####
####                    - Sklearn 1.4.1.post1                                           ####
####                    - TensorFlow 3.13.0 (https://www.tensorflow.org/)               ####
####                    - Xarray 2024.2.0                                               ####
####                   c.f. environments                                                ####
####                                                                                    ####
####                                                                                    ####
############################################################################################


####  IMPORT PACKAGES
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cartopy as cr
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
import optuna
import scipy as sp
import tensorflow as tf
import time
import warnings
import xarray as xr

#### IMPORT MODULES
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState
from sklearn import model_selection
from tensorflow import keras


####  DEFINE CLASS
class FeedForwardNetwork:
    '''
        NAME        : FeedForwardNetwork
        CONTAINS    :   
                        - ApplyHalo                             : Apply longitudinal halo to to input feature in order to prevent grid discontinuities
                        - BuildModel                            : Assemble model layers and compile neural network
                        - Cleanup                               : Clears class attributes from memory
                        - CropInputData                         : Control crop functions for loaded input datasets
                        - CropInputDataGeo                      : Crop loaded datasets to specified latitude and longitude ranges
                        - CropInputDataTime                     : Crop loaded datasets to specified time range
                        - ExtendArrays                          : Extend input data arrays along time axis
                        - GenerateYearMonthList                 : Create array for time [year month] data; FIXME why is this necessary?
                        - GetModelWeights                       : Return weights from trained FFN model
                        - LoadInputData                         : Load predictor and observation data
                        - LoadModelWeights                      : Return Keras tensor of model weights
                        - MakeTrainModel                        : Make and train neural network using TensorFlow's KERAS
                        - MakeTrainModelLoop                    : Loop across SOM provinces to make and train neural networks using TensorFlow's KERAS
                        - OptimiseHyperparameters               : Run hyperparameter optimisation study with Optuna
                        - OptimiseHyperparameters_objective     : Optuna objective function; determines average validation loss across folds
                        - PredictEstimate                       : Use trained neural network to generate final prediction from estimate dataset
                        - PredictEstimateLoop                   : Loop across SOM provinces and use trained neural networks to generate final prediction from estimate dataset
                        - PrepareFolds                          : Arrange input data into folds for hyperparameter tuning and save folds
                        - PrepareInputs                         : Reshape and rearrange predictor and observation data for input to neural network
                        - PlotDiagnostic                        : Plot diagnostics for neural network performance
                        - PlotDiagnosticLoop                    : Loop across SOM provinces to plot diagnostics for neural network performances
                        - PlotInputData                         : Plot predictor and observation data
                        - PlotPrediction                        : Control function for prediction plotting functions.
                        - PlotPredictionMeanComp                : Plot mean pCO2 estimate and comparison with observational dataset
                        - PlotPredictionMeanVar                 : Plot pCO2 estimate mean and variance
    '''

    
    def ApplyHalo(self, data_array, variable_name):
        '''
            NAME        : ApplyHalo
            EDITED      : Daniel Burt       (VLIZ)      22.04.2025
            DESCRIPTION : Apply longitudinal halo to array to prevent grid discontinuities.
            ARGUMENTS   : data_array        (array)     array of data
                          variable_name     (string)    identity of data being processed
        '''

        # instantiate array
        halo_array = np.full((self.array_shape_haloed), np.nan, dtype = np.float32)

        # process longitudes separately
        if variable_name == 'lon':

            # pass data to halo array
            halo_array[:, :, 0]     = data_array[:, :, -2] - 360
            halo_array[:, :, 1]     = data_array[:, :, -1] - 360
            halo_array[:, :, 2:362] = data_array[:, :,  :]
            halo_array[:, :, 362]   = data_array[:, :,  2] + 360 
            halo_array[:, :, 363]   = data_array[:, :,  3] + 360

            return halo_array

        else:

            # pass data to halo array
            halo_array[:, :, 0]     = data_array[:, :, -2]
            halo_array[:, :, 1]     = data_array[:, :, -1]
            halo_array[:, :, 2:362] = data_array[:, :,  :]
            halo_array[:, :, 362]   = data_array[:, :,  2]
            halo_array[:, :, 363]   = data_array[:, :,  3]

            return halo_array


    def BuiildModel(self,
                    number_hidden_layers         = 2,
                    number_neurons_hidden_layers = [16, 16],
                    activation_function          = 'relu',
                    learning_rate                = 1e-4,
                    dropout_rate                 = 0.0):
        '''
            NAME        : BuildModel
            EDITED      : Daniel Burt                       (VLIZ)          06.02.2025
            DESCRIPTION : Assemble layers and compile neural network according to user provided configuration
            ARGUMENTS   : number_hidden_layers              (int)           number of hidden layers for model; must be equal to or less than elements in list for number neurons in hidden layers
                          number_neurons_hidden_layers      (list)          list of the number of neurons in each of the hidden layers
                          activation_function               (string)        selected activation function to use in neural network
                          learning_rate                     (float)         learning rate applied to neural network during training
                          dropout_rate                      (float)         drouput rate for neurons in neural network layer for training
        '''

        # verify that number of layers doesn't exceed number of neurons provided
        if not number_hidden_layers <= len(number_neurons_hidden_layers):
            print('ERROR: Number of Hiden Layers requested exceeds the number of neurons prescribed in list.')
            exit()

        # instantiate list of model layers
        model_layers = []

        # add initial normalizer layer trained with adapt() on provided training input features
        model_layers.append(self.normalizer)

        # loop through hidden layers
        for layer_idx in range(number_hidden_layers):
            model_layers.append(keras.layers.Dense(units = number_neurons_hidden_layers[layer_idx], activation = activation_function))
            if dropout_rate > 0.0:
                model_layers.append(keras.layers.Dropout(rate = dropout_rate))

        # add final linear layer
        model_layers.append(keras.layers.Dense(units = 1, activation = 'linear'))

        # instantiate neural network model
        self.neural_network_model = keras.Sequential(model_layers)
                    
        # compile neural network model
        self.neural_network_model.compile(loss      = 'mean_absolute_error',
                                          optimizer = keras.optimizers.Adam(learning_rate = learning_rate),
                                          metrics   = ['R2Score']
                                          )

    
    def Cleanup(self):
        '''
            NAME        : Cleanup
            EDITED      : Daniel Burt       (VLIZ)      23.04.2025
            DESCRIPTION : Cleanup function to clear class attributes from memory.
        '''

        for attr in list(self.__dict__.keys()):
            delattr(self, attr)


    def CropInputData(self, 
                      year_initial  = 1980,  year_final    = 2023,  # FIXME based on v2024 datasets
                      month_initial = 1,     month_final   = 12, 
                      latitude_min  = -90.,  latitude_max  = 90., 
                      longitude_min = -180., longitude_max = 180.):
        '''
            NAME        : CropInputData
            EDITED      : Daniel Burt       (VLIZ)      19.02.2025
            DESCRIPTION : Core function to control cropping of input datasets
            ARGUMENTS   : year_initial      (int)           Initial year crop threshold
                          year_final        (int)           Final year crop threshold
                          month_initial     (int)           Initial month crop threshold
                          month_final       (int)           Final month crop threshold
                          latitude_min      (float)         Minimum latitude crop threshold
                          latitude_max      (float)         Maximum latitude crop threshold
                          longitude_min     (float)         Minimum longitude crop threshold
                          longitude_max     (float)         Maximum longitude crop threshold
                          arctic            (boolean)       Boolean switch to apply Arctic mask
                          coastal           (boolean)       Boolean switch to apply coastal mask
                          mediterranean     (boolean)       Boolean switch to apply Mediterranean mask
        '''

        # evaluate input array dictionary
        if len(self.input_array_dict) == 0:
            print('ERROR: Input data not found. Please call function: "LoadInputData" before "CropInputData".')
            exit()

        # crop input dataset time range  -->> required to ensure input and observation data cover same time period
        self.CropInputDataTime(year_initial, year_final, month_initial, month_final)

        # crop input dataset latitude and longitude range
        if latitude_min > -90. or latitude_max < 90. or longitude_min > -180. or longitude_max < 180.:
            self.CropInputDataGeo(latitude_min, latitude_max, longitude_min, longitude_max)


    def CropInputDataGeo(self,
                         latitude_min, latitude_max,
                         longitude_min, longitude_max):
        '''
            NAME        : CropInputDataGeo
            EDITED      : Daniel Burt       (VLIZ)      19.02.2025
            DESCRIPTION : Crop input datasets geographically
            ARGUMENTS   : latitude_min      (float)         Minimum latitude crop threshold
                          latitude_max      (float)         Maximum latitude crop threshold
                          longitude_min     (float)         Minimum longitude crop threshold
                          longitude_max     (float)         Maximum longitude crop threshold
        '''

        # loop through input array dictionary
        for input_variable in self.input_array_dictionary.keys():

            # evaluate latitude minimum
            if latitude_min > -90.:

                # remove latitudes below minimum crop threshold
                self.input_array_dict[input_variable] = np.where(self.input_array_dict['lat'] >= latitude_min, self.input_array_dict[input_variable], np.nan)

            # evaluate latitude maximum
            if latitude_max < 90.:

                # remove latitudes above maximum crop threshold
                self.input_array_dict[input_variable] = np.where(self.input_array_dict['lat'] <= latitude_max, self.input_array_dict[input_variable], np.nan)

            # evaluate longitude minimum
            if longitude_min > -180.:

                # remove longitudes below minimum crop threshold
                self.input_array_dict[input_variable] = np.where(self.input_array_dict['lon'] >= longitude_min, self.input_array_dict[input_variable], np.nan)

            # evaluate longitude maximum
            if longitude_max < 180.:

                # remove longitude above maximum crop threshold
                self.input_array_dict[input_variable] = np.where(self.input_array_dict['lon'] <= longitude_max, self.input_array_dict[input_variable], np.nan)

    
    def CropInputDataTime(self, 
                          year_initial, year_final,
                          month_initial, month_final):
        '''
            NAME        : CropInputDataTime
            EDITED      : Daniel Burt       (VLIZ)      19.02.2025
            DESCRIPTION : Crop input datasets along time axis.
            ARGUMENTS   : year_initial      (int)           Initial year crop threshold
                          year_final        (int)           Final year crop threshold
                          month_initial     (int)           Initial month crop threshold
                          month_final       (int)           Final month crop threshold
        '''

        # generate dataset labels for time crop
        yearmonth_labels_crop = self.GenerateYearMonthList(year_initial  = year_initial,  year_final  = year_final,
                                                           month_initial = month_initial, month_final = month_final)

        # instantiate lists
        self.length_time_axes = []
        input_key_list   = list(self.input_array_dict.keys())

        # loop through input array dictionary keys
        for input_variable in input_key_list:

            # evaluate key
            if input_variable != 'lat' and input_variable != 'lon':

                # add to list
                self.length_time_axes.append(self.input_array_dict[input_variable].shape[0])

        # loop through dataset time axes
        for n in range(len(self.length_time_axes)):

            # retrieve input variable
            input_variable = input_key_list[n]

            # evaluate input variable --> skip lat/ lon
            if input_variable == 'lat' or input_variable == 'lon':
                continue

            # evaluate length of time axis against crop year month label list
            if self.length_time_axes[n] > len(yearmonth_labels_crop):

                # determine start year of longest time axis; assume shared end-year FIXME v2024 datasets
                start_year = int(2024 - (self.length_time_axes[n]/ 12))

                # generate base year month labels
                yearmonth_labels_base = self.GenerateYearMonthList(year_initial  = start_year, year_final  = 2023,  # FIXME v2024 datasets
                                                                   month_initial = 1,          month_final = 12)

                # loop through default year month label list
                for t in range(len(yearmonth_labels_base)):

                    # evaluate year month labels
                    if yearmonth_labels_base[t] not in yearmonth_labels_crop:

                        # evaluate for assessed dataset and crop year month data
                        self.input_array_dict[input_variable][t, :, :] = np.nan


    def ExtendArrays(self, data_array, array_shape):
        '''
            NAME        : ExtendArrays
            EDITED      : Daniel Burt       (VLIZ)      27.01.2025
            DESCRIPTION : Extend array to defined array shape.
                          Assumes extension occurs along time axis and data have the
                          same end year.
                          Fills extension with NumPy NaNs.
            ARGUMENTS   : data_array        (array)     array of data
                          array_shape       (tuple)     tuple of defined target array shape
        '''

        # evaluate shape of data array
        if data_array.shape != array_shape:

            # instantiate new array
            data_array_extended = np.full(array_shape, np.nan, dtype = np.float32)

            # determine start index of existing data array
            start_index = array_shape[0] - data_array.shape[0]

            # insert data array
            data_array_extended[start_index:] = data_array

            # return extended data array
            return data_array_extended
        
        else:

            # return original array
            return data_array


    def GenerateYearMonthList(self, 
                              year_initial  = 1980, year_final  = 2023,  # FIXME based on v2024 datasets
                              month_initial = 1,    month_final = 12):
        '''
            NAME        : GenerateYearMonthList
            EDITED      : Daniel Burt       (VLIZ)      27.01.2025
            DESCRIPTION : Generate a list of year-month date labels (inclusive)
            ARGUMENTS   : year_initial      (int)           Initial year label
                          year_final        (int)           Final year label
                          month_initial     (int)           Initial month label
                          month_final       (int)           Final month label 
        '''

        # instantiate list
        yearmonth_labels = []

        # determine time range
        time_range = ((year_final + 1) - year_initial) * 12

        # loop through time range
        for t in range(time_range):

            # instantiate temporary variables
            year  = (t//12) + year_initial
            month = (t%12) + 1

            # evaluate label for label exceptions
            if year == year_initial and month < month_initial:
                continue
            elif year == year_final and month > month_final:
                continue
            else:

                # append valid labels to list
                yearmonth_labels.append([year, month])

        return yearmonth_labels
    

    def GetModelWeights(self, fpath_output = None, file_identifier = None):
        '''
            NAME        : GetModelWeights
            EDITED      : Daniel Burt       (VLIZ)      13.10.2025
            DESCRIPTION : Return Keras tensor of model weights. Also stores weights using Keras built-in I/O
            ARGUMENTS   : fpath_output          (Str)       File path to storage directory
                          file_identifier       (Str)       String to identify stored file
        '''

        # check output filepath argument
        if fpath_output is None:
            fpath_output = './output-plots/'
        else:
            fpath_output = fpath_output

        # obtain neural network model weights tensor
        model_weights = self.neural_network_model.get_weights()

        # write weights to file for storage
        if file_identifier is None:
            np.savez(f"{fpath_output}/ffn-model-weights.npz", *model_weights)
        else:
            np.savez(f"{fpath_output}/ffn-model-weights_{file_identifier}.npz", *model_weights)

        # return object to Class exterior
        return model_weights


    def LoadInputData(self, input_dictionary):
        '''
            NAME        : LoadInputData
            EDITED      : Daniel Burt       (VLIZ)      19.02.2025
            DESCRIPTION : Function for loading data for input features.
                          Read data from MATLAB or netCDF data file formats.
                          Data is extracted as NumPy Arrays in dimensions 
                          [months latitude longitude].
                          Define latitude and longitude arrays for simplicity 
                          as arrays read in from data files can have varying
                          dimension [180 360].
            ARGUMENTS   : input_dictionary      (dict)        dictionary of variable names and file paths
        '''

        # evaluate input dictionary
        if len(input_dictionary) == 0:
            print('ERROR: Input data not found. Please give dictionary of variable names and file paths as input to function.')
            exit()

        # identify observational key
        self.observation_variable = list(input_dictionary.keys())[-1]

        # instantiate dictionary for input arrays
        self.input_array_dict = dict()

        # loop through input dictionary
        for input_variable in input_dictionary.keys():

            # evaluate given file path
            if not os.path.isfile(input_dictionary[input_variable]):
                print(f"ERROR: File {input_dictionary[input_variable]} for {input_variable} not found. Please enter valid filepath.")
                exit()

            # evaluate data format
            if input_dictionary[input_variable][-4:] == '.mat':
                
                # read matlab data file format
                self.input_array_dict[input_variable] = sp.io.loadmat(input_dictionary[input_variable])[input_variable].astype('float32')

            elif input_dictionary[input_variable][-3:] == '.nc':

                # read netcdf data file format
                self.input_array_dict[input_variable] = xr.load_dataset(input_dictionary[input_variable])[input_variable].values.astype('float32')

        # define latitude and longitude arrays
        self.input_array_dict['lat'] = np.tile(np.linspace( -89.5,  89.5, 180, dtype = np.float32), (360, 1)).T
        self.input_array_dict['lon'] = np.tile(np.linspace(-179.5, 179.5, 360, dtype = np.float32), (180, 1))

        # Diagnostic  -->>  Memory Consumption
        for key in self.input_array_dict.keys():
            print(f"{key:<18s}: {self.input_array_dict[key].nbytes / (1024**2):.3f} MB")
        print('')
    

    def LoadModelWeights(self, fpath_input = None, file_identifier = None):
        '''
            NAME        : LoadModelWeights
            EDITED      : Daniel Burt       (VLIZ)      13.10.2025
            DESCRIPTION : Return Keras tensor of model weights. Also stores weights using Keras built-in I/O
            ARGUMENTS   : fpath_output          (Str)       File path to storage directory
                          file_identifier       (Str)       String to identify stored file
        '''

        # check output filepath argument
        if fpath_input is None:
            fpath_input = './output-plots/'
        else:
            fpath_input = fpath_input

        # read weights from file in storage
        if file_identifier is None:
            if os.path.isfile(f"{fpath_input}/ffn-model-weights.npz"):
                data = np.load(f"{fpath_input}/ffn-model-weights.npz")
            else:
                print(f"WARNING: File {fpath_input}/ffn-model-weights.npz not found. Model using random weight initialisation.")
        else:
            if os.path.isfile(f"{fpath_input}/ffn-model-weights_{file_identifier}.npz"):
                data = np.load(f"{fpath_input}/ffn-model-weights_{file_identifier}.npz")
            else:
                print(f"WARNING: File {fpath_input}/ffn-model-weights_{file_identifier}.npz not found. Model using random weight initialisation.")

        # reconstruct neural network model weights tensors
        model_weights = [data[key] for key in data.files]

        # return object to Class exterior
        return model_weights


    def MakeTrainModel(self, 
                       number_hidden_layers         = 2,
                       number_neurons_hidden_layers = [16, 16], 
                       activation_function          = 'relu', 
                       learning_rate                = 1e-4,
                       dropout_rate                 = 0.0,
                       length_patience              = 10, 
                       batch_size                   = 128,
                       epochs                       = 500, 
                       previous_weights             = None):
        '''
            NAME        : TrainModel
            EDITED      : Daniel Burt               (VLIZ)          10.02.2026
            DESCRIPTION : Make and train machine learning model
            ARGUMENTS   : number_hidden_neurons     (int)           number of neurons in hidden layer of machine learning model
                          activation_function       (string)        selected activation function corresponding to TensorFlow keras: https://keras.io/api/layers/activations/
                          learning_rate             (float)         learning rate of machine learning model
                          epochs                    (int)           number of times machine learning model presented with data
                          previous_weights          (array)         weights of previously trained model [optional]
        '''

        # set random seed for reproducibility
        tf.random.set_seed(1)

        # subsample training dataset to generate training and testing (unseen) datasets
        training_input, self.testing_input, training_target, self.testing_target = model_selection.train_test_split(self.dataset_input, 
                                                                                                                    self.dataset_target, 
                                                                                                                    test_size = 0.2)

        # initialise time recording for diagnostic reporting
        time_ini = time.time()

        # make normalization layer
        self.normalizer = keras.layers.Normalization(axis = -1)

        # compute mean and variance of train subsample of training dataset  --  TODO does this compromise performance of testing/ validation data? assume same mean/ variance
        self.normalizer.adapt(training_input)

        # # instantiate neural network model
        # self.neural_network_model = keras.Sequential([normalizer,
        #                                               keras.layers.Dense(units = number_hidden_neurons , activation = activation_function),
        #                                               keras.layers.Dense(units = 1, activation = 'linear')
        #                                               ])
        
        # # compile neural network model
        # self.neural_network_model.compile(loss = 'mean_absolute_error',
        #                                   optimizer = keras.optimizers.Adam(0.001),
        #                                   metrics = ['R2Score']
        #                                   )

        # call model builder function
        self.BuiildModel(number_hidden_layers, number_neurons_hidden_layers, activation_function, learning_rate, dropout_rate)

        # check for previous weights in correct format
        if previous_weights is not None and isinstance(previous_weights, list):

            # build dense layers using a single input before setting weights
            _ = self.neural_network_model(training_input[:1])

            # skip indices to handle dropout layer spacing
            idx_skip = 1

            # loop through hidden layers
            for layer_idx in range(number_hidden_layers + 1):

                # account for additional dropout layers
                if dropout_rate > 0.0:

                    # set weights of the dense layers to the weights produced from training of the previous layer; allow Normalization layer to use new weights
                    self.neural_network_model.layers[(layer_idx + idx_skip)].set_weights(previous_weights[(3 + (layer_idx * 2)):(5 + (layer_idx * 2))])

                    # increment index skip
                    idx_skip += 1

                # do not account for dropout layers
                else:

                    # set weights of the dense layers to the weights produced from training of the previous layer; allow Normalization layer to use new weights
                    self.neural_network_model.layers[(layer_idx + 1)].set_weights(previous_weights[(3 + (layer_idx * 2)):(5 + (layer_idx * 2))])

        else:

            # continue using randomly initialised weights
            print('WARNING: Model proceeding with randomly initialised weights.')

        
        # report summary  -->> what does this do? TODO
        self.neural_network_model.summary()

        # define stop condition
        stop_condition = keras.callbacks.EarlyStopping(monitor              = "val_loss",
                                                       min_delta            = 0,
                                                       patience             = length_patience,
                                                       verbose              = 0,
                                                       mode                 = "auto",
                                                       baseline             = None,
                                                       restore_best_weights = True,
                                                       start_from_epoch     = 0
                                                       )
        
        # train neural network model
        self.training_history = self.neural_network_model.fit(x                     = training_input,
                                                              y                     = training_target,  
                                                              batch_size            = batch_size,
                                                              epochs                = epochs,
                                                              verbose               = 2,
                                                              callbacks             = [stop_condition],
                                                              validation_split      = 0.2,  # statistically sufficient to sequentially partition randomly shuffled data
                                                              validation_batch_size = batch_size,
                                                              )
                
        # end timer and report timing diagnostic
        time_fin = time.time()
        time_elapsed = time_fin - time_ini
        print(f"Feed Forward Network training ended after {time_elapsed} seconds")
        print('')
        

    def MakeTrainModelLoop(self, number_hidden_neurons = 60, activation_function = 'relu', length_patience = 10, epochs = 500):
        '''
            NAME        : TrainModel
            EDITED      : Daniel Burt   (VLIZ)      10.02.2026
            DESCRIPTION : Make and train machine learning model
            ARGUMENTS   : number_hidden_neurons     (int)           number of neurons in hidden layer of machine learning model
                          activation_function       (string)        selected activation function corresponding to TensorFlow keras: https://keras.io/api/layers/activations/
                          learning_rate             (float)         learning rate of machine learning model
                          epochs                    (int)           number of times machine learning model presented with data
        '''

        # instantiate dictionary
        self.neural_network_model_dictionary = dict()
        self.neural_network_history_dictionary = dict()
        self.neural_network_test_dictionary = dict()
        self.province_dict = dict()

        # set random seed for reproducibility
        tf.random.set_seed(1)

        # determine number of provinces
        number_provinces = int(np.nanmax(self.column_dict['provinces'][:, -1])) + 1

        # loop through provinces
        for province in range(number_provinces):

            # instantiate sub-dictionary
            self.neural_network_test_dictionary[province] = dict()

            # subsample training dataset for training and test subsets
            training_input, self.neural_network_test_dictionary[province]['testing_input'], training_target, self.neural_network_test_dictionary[province]['test_target'] = model_selection.train_test_split(self.dataset_input[self.province_training.ravel() == province], 
                                                                                                                                                                                                             self.dataset_target[self.province_training.ravel() == province], 
                                                                                                                                                                                                             test_size = 0.2)

            # initialise time recording for diagnostic reporting
            time_ini = time.time()

            # make normalization layer  -->> what is the normalization layer?? TODO
            normalizer = keras.layers.Normalization(axis = -1)

            # compute mean and variance of train subsample of training dataset
            normalizer.adapt(training_input)

            # instantiate neural network model
            self.neural_network_model_dictionary[province] = keras.Sequential([normalizer,
                                                                               keras.layers.Dense(units = number_hidden_neurons , activation = activation_function),
                                                                               keras.layers.Dense(units = 1, activation = 'linear')
                                                                               ])
            
            # compile neural network model
            self.neural_network_model_dictionary[province].compile(loss = 'mean_absolute_error',
                                                                   optimizer = keras.optimizers.Adam(0.001),
                                                                   metrics = ['R2Score']
                                                                   )
            
            # report summary
            self.neural_network_model_dictionary[province].summary()

            # define stop condition
            stop_condition = keras.callbacks.EarlyStopping(monitor = "val_loss",
                                                          min_delta = 0,
                                                          patience = length_patience,
                                                          verbose = 0,
                                                          mode = "auto",
                                                          baseline = None,
                                                          restore_best_weights = True,
                                                          start_from_epoch = 0
                                                          )
            
            # train neural network model
            self.neural_network_history_dictionary[province] = self.neural_network_model_dictionary[province].fit(x                = training_input,
                                                                                                                  y                = training_target,
                                                                                                                  validation_split = 0.2,  # statistically sufficient to sequentially partition randomly shuffled data
                                                                                                                  # validation_data = (self.neural_network_test_dictionary[province]['training'], self.neural_network_test_dictionary[province]['validate']),
                                                                                                                  verbose          = 2,
                                                                                                                  callbacks        = [stop_condition],
                                                                                                                  epochs           = epochs)
                             
            # end timer and report timing diagnostic
            time_fin = time.time()
            time_elapsed = time_fin - time_ini
            print(f"Feed Forward Network training ended after {time_elapsed} seconds")
            print('')

    
    def OptimiseHyperparameters(self, n_trials = 50, n_folds = 5, fpath_database = None):
        '''
            NAME        : OptimiseHyperparameters
            EDITED      : Daniel Burt       (VLIZ)          11.02.2026
            DESCRIPTION : Run hyperparameter optimisation study with Optuna
            ARGUMENTS   : n_trials          (integer)       number of experiments for Optuna objective function
                          n_folds           (integer)       number of data folds prepared
                          fpath_work        (string)        file path for 
        '''

        # set default path for database storage
        if fpath_database is None:
            fpath_database = '.'

        # define storage path for sqlite database object
        dpath_storage = f'sqlite:///{fpath_database}/optuna_study_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.db'

        # Arianna sets number of CPU threads in PyTorch here -- not necessary for GPU build

        # instantiate study, sampler and pruner
        pruner  = optuna.pruners.MedianPruner(n_warmup_steps = 20)  # wait at least 20 epochs before considering pruning
        sampler = optuna.samplers.TPESampler(seed = 42)
        study   = optuna.create_study(direction      = 'minimize', 
                                      pruner         = pruner, 
                                      sampler        = sampler,
                                      storage        = dpath_storage,
                                      load_if_exists = True)  # database timestamped; only applies for shared database filenames

        # run optuna study
        study.optimize(lambda trial: self.OptimiseHyperparameters_objective(trial, n_folds), n_trials = n_trials)

        # retrieve study information
        pruned_trials    = study.get_trials(deepcopy = False, states = [TrialState.PRUNED])
        completed_trials = study.get_trials(deepcopy = False, states = [TrialState.COMPLETE])
        
        # report study result
        print('Optuna Study Statistics:')
        print(f'  -> Number of Finished  Trials: {len(study.trials)}')
        print(f'  -> Number of Pruned    Trials: {pruned_trials}')
        print(f'  -> Number of Completed Trials: {completed_trials}')
        print('')
        print(f'  -> Best Trial:')
        trial = study.best_trial
        print(f'     -> Value     : {trial.value}')
        print(f'     -> Parameters:')
        for key, value in trial.params.items():
            print(f'     -> {key}: {value}')


    def OptimiseHyperparameters_objective(self, trial, n_folds):
        '''
            NAME        : OptimiseHyperparameters_objective
            EDITED      : Daniel Burt       (VLIZ)              11.02.2026
            DESCRIPTION : Optuna objective function; determines average validation loss across folds
            ARGUMENTS   : trial             (optuna.trial)      Optuna trial object
                          n_folds           (integer)           Specify number of prepared data folds to loop across
        '''

        # instantiate constants
        epochs          = 5000
        length_patience = 20

        # Arianna sets a CPU or GPU defininition

        # set random seed for reproducibility -- Arianna sets different seed for each trial; why?
        tf.random.set_seed(1)

        # define hyperparameters to be tuned and available options
        n_layers = trial.suggest_int('n_layers', 2, 3)

        n_units_l0 = trial.suggest_categorical('n_units_l0', [16, 32, 64, 128, 256])
        n_units_l1 = trial.suggest_categorical('n_units_l1', [16, 32, 64, 128, 256])
        n_units_l2 = trial.suggest_categorical('n_units_l2', [16, 32, 64, 128, 256])

        layer_sizes = [n_units_l0, n_units_l1, n_units_l2][:n_layers]

        # enforce layer structure
        if n_layers == 2:

            # prune structures where last layer is larger than first layer
            if not (layer_sizes[0] >= layer_sizes[1]):
                raise optuna.TrialPruned()

        elif n_layers == 3:

            # prune structures where middle layer is not the largest layer
            if not (layer_sizes[0] <= layer_sizes[1] >= layer_sizes[2]):
                raise optuna.TrialPruned()
            
        activation_fn = trial.suggest_categorical('activation', ['relu', 'tanh', 'leaky_relu'])
        learning_rate = trial.suggest_float('lr', 1e-4, 1e-2, log = True)
        batch_size    = trial.suggest_categorical('batch_size', [128, 256, 512, 1024, 2048])
        dropout_rate  = trial.suggest_float('dropout', 0.0, 0.5)

        # report trial hyperparameters to terminal for logging
        print(f'Trial {trial.number:03d}: Layers = {n_layers}, Sizes = {layer_sizes}, Activation Function = {activation_fn}, Learning Rate = {learning_rate:.5f}, Batch Size = {batch_size}, Dropout = {dropout_rate}')

        # instantiate lists for cross-validation metrics
        fold_losses = []
        fold_r2s    = []

        # loop through defined training folds
        for fold_idx in range(n_folds):  # FIXME add flexibility of folds; must match PrepareFolds()
            
            # retrieve training and validation data from fold dictionary
            training_input  = self.dict_fold_data[f'training_fold_{fold_idx}_input']
            training_target = self.dict_fold_data[f'training_fold_{fold_idx}_target']
            validate_input  = self.dict_fold_data[f'validate_fold_{fold_idx}_input']
            validate_target = self.dict_fold_data[f'validate_fold_{fold_idx}_target']

            # initialise time recording for diagnostic reporting
            time_ini = time.time()

            # make normalization layer
            self.normalizer = keras.layers.Normalization(axis = -1)

            # compute mean and variance of train subsample of training dataset  -- TODO does this compromise performance of testing/ validation data? assume same mean/ variance
            self.normalizer.adapt(training_input)
            
            # call model builder function
            self.BuiildModel(number_hidden_layers         = n_layers, 
                             number_neurons_hidden_layers = layer_sizes, 
                             activation_function          = activation_fn, 
                             learning_rate                = learning_rate, 
                             dropout_rate                 = dropout_rate)

            # define stop condition
            stop_condition = [keras.callbacks.EarlyStopping(monitor              = "val_loss",
                                                           min_delta            = 0,
                                                           patience             = length_patience,
                                                           verbose              = 0,
                                                           mode                 = "auto",
                                                           baseline             = None,
                                                           restore_best_weights = True,
                                                           start_from_epoch     = 0
                                                           ),
                              TFKerasPruningCallback(trial, 'val_loss')
                             ]
            
            # train neural network model
            self.training_history = self.neural_network_model.fit(x                     = training_input,
                                                                  y                     = training_target,
                                                                  batch_size            = batch_size,
                                                                  epochs                = epochs,
                                                                  verbose               = 2,
                                                                  callbacks             = stop_condition,
                                                                  validation_data       = (validate_input, validate_target),
                                                                  validation_batch_size = batch_size
                                                                  )
                    
            # end timer and report timing diagnostic
            time_fin = time.time()
            time_elapsed = time_fin - time_ini
            print(f'          Training for Fold {fold_idx} ended after {time_elapsed} seconds')

            # retrieve index for best epoch from neural network training history
            best_epoch = np.argmin(self.training_history.history['val_loss'])

            # store evaluation in lists
            fold_losses.append(self.training_history.history['val_loss'][best_epoch])
            fold_r2s.append(self.training_history.history['val_R2Score'][best_epoch])

        # compute evaluate mean across folds
        mean_loss = np.mean(fold_losses)
        mean_r2   = np.mean(fold_r2s)

        # return metrics to Optuna trial object
        trial.set_user_attr('mean_r2', mean_r2)
        trial.set_user_attr('fold_val_losses', fold_losses)
        trial.set_user_attr('fold_val_r2s', fold_r2s)

        # report statistics for logging
        print(f'          Mean Loss (MAE) across folds: {mean_loss:.4f}')
        print(f'          Mean R2         across folds: {mean_r2:.4f}')
        print(f'          Fold Losses (MAE)           : {fold_losses}')
        print(f'          Fold R2s                    : {fold_r2s}')

        return mean_loss
        

    def PredictEstimate(self, fpath_output = None, file_identifier = None, fileext = None):
        '''
            NAME        : PredictEstimate
            EDITED      : Daniel Burt       (VLIZ)      27.05.2025
            DESCRIPTION : Use trained neural network model to generate final prediction from estimate dataset
                          Predicted dataset is also written to data file
            ARGUMENTS   : fpath_output      (string)        output file path for saving data
                          fileext           (string)        output file data format: 'mat' or 'nc' available
        '''

        # check output filepath argument
        if fpath_output is None:
            fpath_output = './output-plots/'
        else:
            fpath_output = fpath_output

        # check output filetype argument
        if fileext is None:
            fileext = 'nc'
        else:
            if fileext == 'nc' or fileext == 'mat':
                fileext = fileext
            else:
                print(f"ERROR: Designated output file extension: {fileext} is not recognised. Please use 'mat' or 'nc' data formats.")

        # predict pCO2 using estimate dataset
        ffn_estimate = self.neural_network_model.predict(self.dataset_predict)

        # map one-dimensional neural-network estimate back into three-dimensions (time, lat, lon)
        self.pco2_estimate = np.full(self.array_shape, np.nan, dtype = np.float32)
        # self.pco2_estimate = np.full(self.array_shape_haloed, np.nan, dtype = np.float32)
        self.pco2_estimate = np.ravel(self.pco2_estimate)
        self.pco2_estimate[~self.valid_mask_predict] = np.ravel(ffn_estimate)
        self.pco2_estimate = self.pco2_estimate.reshape(self.array_shape)
        # self.pco2_estimate = self.pco2_estimate.reshape(self.array_shape_haloed)

        # save estimate to data file
        if fileext == 'mat':

            # configure dictionary
            out_dict = {'ffn_estimate': self.pco2_estimate,
            # out_dict = {'ffn_estimate': self.pco2_estimate[:, :, 2:362],
                        'lat': self.input_array_dict['lat'],
                        'lon': self.input_array_dict['lon']
                        }

            # write provinces to MATLAB data file format
            if file_identifier is None:
                sp.io.savemat(f"{fpath_output}/ffn-output_estimate.mat", out_dict)
            else:
                sp.io.savemat(f"{fpath_output}/ffn-output_estimate_{file_identifier}.mat", out_dict)
        
        elif fileext == 'nc':

            # configure Xarray DataArray
            DA_out = xr.DataArray(self.pco2_estimate,
            # DA_out = xr.DataArray(self.pco2_estimate[:, :, 2:362],
                                  dims = ['time', 'lat', 'lon'],
                                  coords = {'time': range(self.array_shape[0]),
                                            'lat' : np.linspace( -89.5,  89.5, 180, dtype = np.float32), 
                                            'lon' : np.linspace(-179.5, 179.5, 360, dtype = np.float32)},
                                  name = 'ffn_estimate')

            # write provinces to netCDF data file format
            if file_identifier is None:
                DA_out.to_netcdf(f"{fpath_output}/ffn-output_estimate.nc")
            else:
                DA_out.to_netcdf(f"{fpath_output}/ffn-output_estimate_{file_identifier}.nc")

            # clear memory
            del DA_out

    
    def PredictEstimateLoop(self, fpath_output = None, fileext = None):
        '''
            NAME        : PredictEstimate
            EDITED      : Daniel Burt       (VLIZ)      21.02.2025
            DESCRIPTION : Use trained neural network model to generate final prediction from estimate dataset
                          Predicted dataset is also written to data file
            ARGUMENTS   : fpath_output      (string)        output file path for saving data
                          fileext           (string)        output file data format: 'mat' or 'nc' available
        '''

        # check output filepath argument
        if fpath_output is None:
            fpath_output = './output-plots/'
        else:
            fpath_output = fpath_output

        # check output filetype argument
        if fileext is None:
            fileext = 'nc'
        else:
            if fileext == 'nc' or fileext == 'mat':
                fileext = fileext
            else:
                print(f"ERROR: Designated output file extension: {fileext} is not recognised. Please use 'mat' or 'nc' data formats.")

        # instantiate array to fill with estimate
        ffn_estimate = np.full(self.province_estimate.shape, np.nan, dtype = np.float32)

        # determine number of provinces
        number_provinces = int(np.nanmax(self.column_dict['provinces'][:, -1])) + 1

        # loop through provinces
        for province in range(number_provinces):

            # predict pCO2 using estimate dataset
            ffn_estimate[self.province_estimate.ravel() == province] = self.neural_network_model_dictionary[province].predict(self.dataset_predict[self.province_estimate.ravel() == province])

        # map one-dimensional neural-network estimate back into three-dimensions (time, lat, lon)
        self.pco2_estimate = np.full(self.array_shape, np.nan, dtype = np.float32)
        self.pco2_estimate = np.ravel(self.pco2_estimate)
        self.pco2_estimate[~self.valid_mask_predict] = np.ravel(ffn_estimate)
        self.pco2_estimate = self.pco2_estimate.reshape(self.array_shape)

        # save estimate to data file
        if fileext == 'mat':

            # configure dictionary
            out_dict = {'pco2_estimate': self.pco2_estimate,
                        'lat': self.input_array_dict['lat'],
                        'lon': self.input_array_dict['lon']
                        }

            # write provinces to MATLAB data file format
            sp.io.savemat(f"{fpath_output}/ffn-output_pco2-estimate.mat", out_dict)
        
        elif fileext == 'nc':

            # configure Xarray DataArray
            DA_out = xr.DataArray(self.pco2_estimate,
                                  dims = ['time', 'lat', 'lon'],
                                  coords = {'time': range(self.array_shape[0]),
                                            'lat' : np.linspace( -89.5,  89.5, 180, dtype = np.float32), 
                                            'lon' : np.linspace(-179.5, 179.5, 360, dtype = np.float32)},
                                  name = 'pco2_estimate')

            # write provinces to netCDF data file format
            DA_out.to_netcdf(f"{fpath_output}/ffn-output_pco2-estimate_loop.nc")

    
    def PrepareFolds(self, n_folds = 4, intermediate_dir_path = './working-directory'):
        '''
            NAME        : PrepareFolds
            EDITED      : Daniel Burt               (VLIZ)          10.02.2026
            DESCRIPTION : Arrange input data into folds for hyperparameter tuning and save folds or load prepared folds.
            ARGUMENTS   : n_folds                   (int)           number of folds of data to generate; currently, hard-coded
                          intermediate_dir_path     (string)        path to working directory where fold data to be stored
        '''

        # verify directory path for reading/ writing fold files
        if not os.path.isdir(intermediate_dir_path):
            print(f'ERROR: Invalid directory path detected - {intermediate_dir_path}')
            exit()

        # instantiate boolean arraay
        valid_fold_files = np.full((18,), False)

        # instantiate dictionary for fold data
        self.dict_fold_data = dict()

        # define file paths
        list_fold_fnames = ['validate__fold-0-input.npy',  'validate__fold-0-target.npy',  'validate__fold-1-input.npy',  'validate__fold-1-target.npy',
                            'validate__fold-2-input.npy',  'validate__fold-2-target.npy',  'validate__fold-3-input.npy',  'validate__fold-3-target.npy',
                            'training__fold-0-input.npy',  'training__fold-0-target.npy',  'training__fold-1-input.npy',  'training__fold-1-target.npy',
                            'training__fold-2-input.npy',  'training__fold-2-target.npy',  'training__fold-3-input.npy',  'training__fold-3-target.npy',
                            'testing__withheld-input.npy', 'testing__withheld-target.npy']

        # loop through fold file names
        for idx, fname in enumerate(list_fold_fnames):
            
            # verify file paths for reading/ writing fold files
            if os.path.isfile(f'{intermediate_dir_path}/{fname}'):
                valid_fold_files[idx] = True

        # load fold files if possible
        if np.all(valid_fold_files):

            # load training data into dictionary
            for i in range(n_folds):
                self.dict_fold_data[f'training_fold_{i}_input']  = np.load(f'{intermediate_dir_path}/training__fold-{i}-input.npy')
                self.dict_fold_data[f'training_fold_{i}_target'] = np.load(f'{intermediate_dir_path}/training__fold-{i}-target.npy')

            # load testing data into dictionary
            for i in range(n_folds):
                self.dict_fold_data[f'validate_fold_{i}_input']   = np.load(f'{intermediate_dir_path}/validate__fold-{i}-input.npy')
                self.dict_fold_data[f'validate_fold_{i}_target']  = np.load(f'{intermediate_dir_path}/validate__fold-{i}-target.npy')

            # load testing data into dictionary
            self.dict_fold_data['testing_input']  = np.load(f'{intermediate_dir_path}/testing__withheld-input.npy')
            self.dict_fold_data['testing_target'] = np.load(f'{intermediate_dir_path}/testing__withheld-target.npy')
            
        else:

            # define fold (temporal) parameters [FIXME currently hard-coded]
            fold_0_years     = np.array((1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000      ))
            fold_1_years     = np.array((2002, 2003, 2004, 2005,       2007, 2008))
            fold_2_years     = np.array((2009, 2010, 2011,       2013))
            fold_3_years     = np.array((      2015, 2016, 2017, 2018))
            fold_4_years     = np.array((2019, 2020,       2022, 2023, 2024))

            withheld_years   = np.array((2001, 2006, 2012, 2014, 2021))                    # randomly selected 1 year from each fold (80:20)

            # instantiate array of years 
            dummy_year_array = np.ones((self.array_shape))
            # year_array       = np.arange(1971, 2025, 1)     # [FIXME currently hard-coded]

            # assign years to dummy year array
            year_idx         = 1971
            for i in range(self.array_shape[0]):

                # assign year value to dummy array
                dummy_year_array[i, :, :] = dummy_year_array[i, :, :] * year_idx

                # increment year index
                if i%12 == 11:
                    year_idx += 1

            # columnise dummy years
            dummy_year_column     = np.ravel(dummy_year_array).reshape(-1, 1)

            # mask dummy years for training dataset; only need to consider training, cross-validation not applied to estimate dataset
            dummy_year_training   = dummy_year_column[~self.valid_mask_training]

            # instantiate year mask array
            year_mask_array       = np.full((withheld_years.shape[0], dummy_year_training.shape[0]), False)  # [FIXME currently hard-coded]

            # withhold data for testing
            if withheld_years.shape[0] == n_folds:
        
                for idx, year in enumerate(withheld_years):
                    year_mask_array[idx, :] = (dummy_year_training[:, 0] == year)

            else:

                # loop through folds  FIXME hard-coded
                for fold_idx in range(n_folds):
                    year_mask_array[fold_idx, :] = np.logical_or((dummy_year_training[:, 0] == withheld_years[(fold_idx * 2)]), (dummy_year_training[:, 0] == withheld_years[(fold_idx * 2) + 1])) 

            # collapse array dimensions
            withheld_years_mask   = year_mask_array.any(axis = 0)
            
            # apply mask for withheld years
            withheld_input_data   = self.dataset_input[withheld_years_mask]   # stacked input features
            withheld_target_data  = self.dataset_target[withheld_years_mask]   # target observations
            available_input_data  = self.dataset_input[~withheld_years_mask]  # stacked input features
            available_target_data = self.dataset_target[~withheld_years_mask]  # target observations
            available_dummy_years = dummy_year_training[~withheld_years_mask]

            # report information regarding withheld data
            print(f'Withholding years {withheld_years} for testing.')
            print(f'Withholding {withheld_input_data.shape[0]} data points for testing, which represent {(withheld_input_data.shape[0]/ self.dataset_input.shape[0]) * 100:.2f}% of available data.')

            # save validation dataset to file
            np.save(f'{intermediate_dir_path}/testing__withheld-input.npy',  withheld_input_data)
            np.save(f'{intermediate_dir_path}/testing__withheld-target.npy', withheld_target_data)
            print('Withheld data for testing has been saved to files.')
            print('')

            # store validation data in dictionary
            self.dict_fold_data['testing_input']  = withheld_input_data
            self.dict_fold_data['testing_target'] = withheld_target_data

            # loop through folds
            for fold in range(n_folds):

                # retrieve designated years for current fold
                if fold == 0:
                    fold_years = fold_0_years
                elif fold == 1:
                    fold_years = fold_1_years
                elif fold == 2:
                    fold_years = fold_2_years
                elif fold == 3:
                    fold_years = fold_3_years
                elif fold == 4:
                    fold_years = fold_4_years

                # refresh year mask array 
                year_mask_array = np.full((fold_years.shape[0], available_dummy_years.shape[0]), False)

                # loop through fold years
                for idx, year in enumerate(fold_years):
                    year_mask_array[idx, :] = (available_dummy_years[:, 0] == year)

                # collapse array dimensions
                fold_years_mask = year_mask_array.any(axis = 0)

                # apply fold mask to remaining data
                fold_training_input  = available_input_data[~fold_years_mask]   # training data   - stacked input features
                fold_training_target = available_target_data[~fold_years_mask]  # training data   - target observations
                fold_validate_input  = available_input_data[fold_years_mask]    # validation data - stacked input features
                fold_validate_target = available_target_data[fold_years_mask]   # validation data - target observations

                print(f'Fold {fold}')
                print(f'Holding years {fold_years} for cross-validation.')
                print(f'Holding {fold_validate_input.shape[0]} data points for cross-validation, which represent {(fold_validate_input.shape[0]/ available_input_data.shape[0]) * 100:.2f}% of available data.')

                # save fold data to file
                np.save(f'{intermediate_dir_path}/training__fold-{fold}-input.npy',  fold_training_input)
                np.save(f'{intermediate_dir_path}/training__fold-{fold}-target.npy', fold_training_target)
                np.save(f'{intermediate_dir_path}/validate__fold-{fold}-input.npy',  fold_validate_input)
                np.save(f'{intermediate_dir_path}/validate__fold-{fold}-target.npy', fold_validate_target)
                print('Training and validation data has been saved to file.')
                print('')

                # store training data into dictionary
                self.dict_fold_data[f'training_fold_{fold}_input']  = fold_training_input
                self.dict_fold_data[f'training_fold_{fold}_target'] = fold_training_target

                # store testing data into dictionary
                self.dict_fold_data[f'validate_fold_{fold}_input']  = fold_validate_input
                self.dict_fold_data[f'validate_fold_{fold}_target'] = fold_validate_target
   

    def PrepareInputs(self):
        '''
            NAME        : PrepareInputs
            EDITED      : Daniel Burt       (VLIZ)      10.02.2026
            DESCRIPTION : Reshape input arrays and arrange datasets ready for training
        '''

        # evaluate input array dictionary
        if len(self.input_array_dict) == 0:
            print('ERROR: Input data not found. Please call function: "LoadInputData" before "PrepareInputs".')
            exit()

        # evaluate length time axis list
        if len(self.length_time_axes) == 0:
            print('ERROR: List of time-axis lengths not found. Please call function: "CropInputData" before "PrepareInputs".')
            exit()
        
        # instantiate holding variables
        longest_length = 0

        # loop through available axes
        for idx in range(len(self.length_time_axes)):

            # evaluate axis length
            if self.length_time_axes[idx] > longest_length:

                # hold axis length and index
                longest_length = self.length_time_axes[idx]

        # define target array shape
        self.array_shape        = (longest_length, 180, 360)
        # self.array_shape_haloed = (longest_length, 180, 364)

        # instantiate dictionary of extended arrays
        extended_input_array_dict = dict()

        # loop through input variables
        for input_variable in list(self.input_array_dict.keys()):

            # evaluate input variable
            if input_variable not in ['year', 'month', 'lat', 'lon', 'provinces']:

                # extend input data arrays
                extended_input_array_dict[input_variable] = self.ExtendArrays(self.input_array_dict[input_variable], self.array_shape)
                # extended_input_array = self.ExtendArrays(self.input_array_dict[input_variable], self.array_shape)

                # # apply halo to input data arrays
                # extended_input_array_dict[input_variable] = self.ApplyHalo(extended_input_array, input_variable)

                # clear original input arrays from memory
                del self.input_array_dict[input_variable] #, extended_input_array

            # # evaluate input variable
            # elif input_variable in ['lat', 'lon']:

            #     # instantiate extended arrays
            #     # extended_input_array_dict[input_variable] = np.zeros(self.array_shape, dtype = np.float16)
            #     extended_input_array    = np.zeros(self.array_shape, dtype = np.float16)

            #     # extend latitude and longitude arrays
            #     extended_input_array_dict[input_variable][:] = self.input_array_dict[input_variable]
            #     # extended_input_array[:] = self.input_array_dict[input_variable]

            #     # # apply halo to input data arrays
            #     # extended_input_array_dict[input_variable] = self.ApplyHalo(extended_input_array, input_variable)

            #     # clear original input arrays from memory
            #     del self.input_array_dict[input_variable], extended_input_array

        # instantiate extended arrays
        # extended_input_array_dict['spatial_a']  = np.zeros(self.array_shape, dtype = np.float16)
        # extended_input_array_dict['spatial_b']  = np.zeros(self.array_shape, dtype = np.float16)
        # extended_input_array_dict['spatial_c']  = np.zeros(self.array_shape, dtype = np.float16)
        # extended_input_array_dict['year']       = np.zeros(self.array_shape, dtype = np.float16)
        # extended_input_array_dict['month']      = np.zeros(self.array_shape, dtype = np.float16)
        extended_input_array_dict['temporal_a'] = np.zeros(self.array_shape, dtype = np.float16)
        extended_input_array_dict['temporal_b'] = np.zeros(self.array_shape, dtype = np.float16)
        extended_input_array_dict['temporal_c'] = np.zeros(self.array_shape, dtype = np.float16)
        extended_input_array_dict['provinces']  = np.zeros(self.array_shape, dtype = np.float16)
        # extended_input_array_dict['year']       = np.zeros(self.array_shape_haloed, dtype = np.float16)
        # extended_input_array_dict['month']      = np.zeros(self.array_shape_haloed, dtype = np.float16)
        # extended_input_array_dict['provinces']  = np.zeros(self.array_shape_haloed, dtype = np.float16)

        # compute n-vectors for continuous spatial input  -->> Sasse et al. (2013)
        # extended_input_array_dict['spatial_a'][:] = np.sin(np.radians(self.input_array_dict['lat']))[None, :, :]
        # extended_input_array_dict['spatial_b'][:] = (np.sin(np.radians(self.input_array_dict['lon'])) * np.cos(np.radians(self.input_array_dict['lat'])))[None, :, :]
        # extended_input_array_dict['spatial_c'][:] = (-np.cos(np.radians(self.input_array_dict['lon'])) * np.cos(np.radians(self.input_array_dict['lat'])))[None, :, :]

        # # determine start year
        # start_year = int(2024 - (longest_length// 12))  # FIXME based on v2024

        # loop through time index of data
        for t in range(0, longest_length):

            # compute monotonic time encoding to preserve long-term trends
            extended_input_array_dict['temporal_c'][t, :, :] = t/ (longest_length - 1.)

        # instantiate months for conservation of seasonality
        months = np.arange(0, 12)
        # broadcast_months = months[:, np.newaxis, np.newaxis]

        # compute circular time encoding for continuity and preserve seasonality --> Gregor, Kok & Monteiro (2017)
        temporal_a           = np.cos(months * ((2 * np.pi)/ 12))
        temporal_b           = np.sin(months * ((2 * np.pi)/ 12))
        broadcast_temporal_a = temporal_a[:, np.newaxis, np.newaxis]
        broadcast_temporal_b = temporal_b[:, np.newaxis, np.newaxis]

        # loop through number of years of data
        for t in range(0, longest_length, 12):

            # # broadcast month and year to extended array
            # extended_input_array_dict['month'][t:t+12, :, :]     = broadcast_months
            # extended_input_array_dict['year'][t:t+12, :, :]      = start_year + t//12

            # compute circular time encoding for continuity and preserve seasonality --> Gregor, Kok & Monteiro (2017)
            extended_input_array_dict['temporal_a'][t:t+12, :, :] = broadcast_temporal_a
            extended_input_array_dict['temporal_b'][t:t+12, :, :] = broadcast_temporal_b

            # broadcast provinces to extended array
            if 'provinces' in self.input_array_dict.keys():
                extended_input_array_dict['provinces'][t:t+12, :, :] = self.input_array_dict['provinces']

        # clear original input arrays from memory
        if 'provinces' in self.input_array_dict.keys():
            del self.input_array_dict['provinces']

        # instantiate lists and dictionaries
        self.column_dict         = dict()
        columnised_input_list    = []
        valid_mask_training_list = []
        valid_mask_predict_list  = []

        # loop through dictionary keys
        for input_variable in list(extended_input_array_dict.keys()):

            # evaluate for observations key
            if input_variable is not self.observation_variable:

                # flatten input arrays into one dimensional columns
                columnised_input_list.append(np.ravel(extended_input_array_dict[input_variable]).reshape(-1, 1))
                
                # store valid masks
                valid_mask_predict_list.append(np.isnan(np.ravel(extended_input_array_dict[input_variable]).reshape(-1, 1)))

            # store valid masks
            valid_mask_training_list.append(np.isnan(np.ravel(extended_input_array_dict[input_variable]).reshape(-1, 1)))

            # evaluate for variables to keep
            if input_variable == self.observation_variable or input_variable == 'provinces':

                self.column_dict[input_variable] = np.ravel(extended_input_array_dict[input_variable]).reshape(-1, 1)

            # clear extended input arrays from memory
            del extended_input_array_dict[input_variable]

        # combine valid masks  -->> propagate boolean values: True  -->>  FIXME should province be included in valid masking?
        self.valid_mask_training = np.ravel(np.logical_or.reduce(valid_mask_training_list))
        self.valid_mask_predict  = np.ravel(np.logical_or.reduce(valid_mask_predict_list))

        # stack one dimensional columns into vectors of [year, month, lat, lon, columns of input datasets, obs, province]
        dataset_input_full = np.column_stack(columnised_input_list)

        # remove NaNs from datasets
        self.dataset_predict = dataset_input_full[~self.valid_mask_predict]
        if 'provinces' in self.column_dict.keys():
            self.province_estimate = self.column_dict['provinces'][~self.valid_mask_predict]
        self.dataset_input   = dataset_input_full[~self.valid_mask_training]
        if 'provinces' in self.column_dict.keys():
            self.province_training = self.column_dict['provinces'][~self.valid_mask_training]
        self.dataset_target  = self.column_dict[self.observation_variable][~self.valid_mask_training]

        # clear input columns from memory
        del columnised_input_list, valid_mask_training_list, valid_mask_predict_list

        # report shapes for diagnostics
        print('Diagnostic : Shape of datasets')
        print(f"Dummy      : {dataset_input_full.shape} - {dataset_input_full.nbytes / (1024**2):.3f} MB")
        print(f"Training   : {self.dataset_input.shape} - {self.dataset_input.nbytes / (1024**2):.3f} MB")
        print(f"Estimation : {self.dataset_predict.shape} - {self.dataset_predict.nbytes / (1024**2):.3f} MB")
        print('')


    def PlotDiagnostic(self, diagnostic_var, fpath_output_plot = None):
        '''
            NAME        : PlotDiagnostic
            EDITED      : Daniel Burt       (VLIZ)      10.02.2026
            DESCRIPTION : Plot diagnostics for neural network performance
                          Subplot Two illustrates scatter of predicted and true data
                          Subplot Three illustrates the frequency distribution of error magnitudes
            ARGUMENTS   : diagnostic_var        (string)        variable being diagnosed
                          fpath_output_plot     (string)        filepath for saving output plot
                          
        '''

        # check output plot filepath argument
        if fpath_output_plot is None:
            fpath_output_plot = './output-plots/ffn-output_model-diagnostic.png'
        else:
            fpath_output_plot = fpath_output_plot

        # determine number of training epochs
        number_training_epochs = len(self.training_history.history['val_loss'])
        if number_training_epochs < 100:
            epoch_lim = int(10 * np.ceil(number_training_epochs/ 10))
        elif number_training_epochs >= 100 and number_training_epochs < 250:
            epoch_lim = int(50 * np.ceil(number_training_epochs/ 50))
        elif number_training_epochs >= 250:
            epoch_lim = int(100 * np.ceil(number_training_epochs/ 100))

        # evaluate model using test data
        loss, r2score = self.neural_network_model.evaluate(self.testing_input, 
                                                           self.testing_target, 
                                                           verbose = 2)

        # determine predictions from reserved test data
        test_prediction  = self.neural_network_model.predict(self.testing_input)

        # determine error magnitudes
        error_magnitudes = self.testing_target - test_prediction

        # calculate line of best fit
        slope, intercept = np.polyfit(np.ravel(self.testing_target), np.ravel(test_prediction), 1)
        best_fit_line    = slope * self.testing_target + intercept

        # determine bin boundaries for histogram
        bins_lower_lim = 10 * np.floor(np.nanmin(error_magnitudes)/ 10)
        bins_upper_lim = 10 * np.ceil( np.nanmax(error_magnitudes)/ 10)
        bin_edges      = np.arange(bins_lower_lim, (bins_upper_lim + 10), 10)

        # define fontsizes
        size_suptitle = 22
        size_subtitle = 20
        size_labtitle = 16
        size_ticks    = 12

        # instantiate figure and axes
        fig, axs = plt.subplots(nrows = 1, ncols = 3,
                                gridspec_kw = {'wspace': 0.25, 'hspace': 0.},
                                figsize = (21, 7)
                                )
        
        # set colour of empty space to white
        fig.patch.set_facecolor('white')

        # set figure title
        fig.suptitle("Model Diagnostics", fontsize = size_suptitle, fontweight = 'bold', y =0.98)
        fig.text(0.5, 0.89, f"Loss: {loss:.3f}, R$^{2}$: {r2score:.3f}", ha = 'center', fontsize = size_labtitle)

        # plot subplot one
        _ = axs[0].plot(self.training_history.history['loss'], 
                        label = 'Training', 
                        color = 'grey', 
                        linestyle = '-', 
                        linewidth = 1.5)
        _ = axs[0].plot(self.training_history.history['val_loss'], 
                        label = 'Internal Validation', 
                        color = 'black', 
                        linestyle = '-', 
                        linewidth = 1.5)
        
        # define subplot title, labels and ticks
        axs[0].set_title('Training History', fontsize = size_subtitle)
        axs[0].set_xlabel('Epoch', fontsize = size_labtitle)
        # axs[0].set_ylabel('Mean Absolute Error (ppm)', fontsize = size_labtitle)  # pCO2
        if diagnostic_var == 'temperature':
            axs[0].set_ylabel(r'Mean Absolute Error ($\degree$C)', fontsize = size_labtitle)
        else:
            axs[0].set_ylabel(r'Mean Absolute Error ($\mu$mol kg$^{-1}$)', fontsize = size_labtitle)
        axs[0].tick_params(labelsize = size_ticks)
        axs[0].legend(fontsize = (size_ticks + 2))
        axs[0].set_box_aspect(1)

        # define subplot limits
        axs[0].set_xlim([0, epoch_lim])
        # axs[0].set_ylim([0, 10])  # pCO2
        if diagnostic_var == 'temperature':
            axs[0].set_ylim([0, 1])
        else:
            axs[0].set_ylim([0, 100])

        # plot subplot two
        _ = axs[1].scatter(x = self.testing_target,
                           y = test_prediction,
                           color = 'black',
                           s = 10)
        _ = axs[1].plot(self.testing_target,
                        best_fit_line,
                        color = 'red',
                        linestyle = '-',
                        linewidth = 1.5)
        # _ = axs[1].plot([0, 3000],  # pCO2
        #                 [0, 3000],
        if diagnostic_var == 'temperature':
            _ = axs[1].plot([-10, 50],
                            [-10, 50],
                            color = 'gray',
                            linestyle = '--',
                            linewidth = 1.5)
        else:
            _ = axs[1].plot([1800, 2500],
                            [1800, 2500],
                            color = 'gray',
                            linestyle = '--',
                            linewidth = 1.5)
        
        # define subplot title, labels and ticks
        axs[1].set_title('Observed vs Predicted', fontsize = size_subtitle)
        if diagnostic_var == 'temperature':
            axs[1].set_xlabel(r'Observations ($\degree$C)', fontsize = size_labtitle)
            axs[1].set_ylabel(r'Predictions ($\degree$C)', fontsize = size_labtitle)
        else:
            axs[1].set_xlabel(r'Observations ($\mu$mol kg$^{-1}$)', fontsize = size_labtitle)
            axs[1].set_ylabel(r'Predictions ($\mu$mol kg$^{-1}$)', fontsize = size_labtitle)
        axs[1].tick_params(labelsize = size_ticks)
        axs[1].set_box_aspect(1)

        # define subplot limits
        if diagnostic_var == 'temperature':
            axs[1].set_xlim([-5, 40])
            axs[1].set_ylim([-5, 40])
        else:
            axs[1].set_xlim([1850, 2450])
            axs[1].set_ylim([1850, 2450])

        # plot subplot three
        _ = axs[2].hist(error_magnitudes, 
                        bins = bin_edges, 
                        color = 'black')

        # define subplot title, labels and ticks
        axs[2].set_title('Prediction Error Distribution', fontsize = size_subtitle)
        if diagnostic_var == 'temperature':
            axs[2].set_xlabel(r'Magnitude of Prediction Error ($\degree$C)', fontsize = size_labtitle)
        else:
            axs[2].set_xlabel(r'Magnitude of Prediction Error ($\mu$mol kg$^{-1}$)', fontsize = size_labtitle)
        axs[2].set_ylabel('Count', fontsize = size_labtitle)
        axs[2].tick_params(labelsize = size_ticks)
        axs[2].set_box_aspect(1)

        # adjust figure padding
        fig.subplots_adjust(top = 0.85, bottom = 0.05)

        # save figure to output directory
        plt.savefig(fpath_output_plot, bbox_inches = 'tight', dpi = 100)

        plt.close()


    def PlotDiagnosticLoop(self, fpath_output_plot = None):
        '''
            NAME        : PlotDiagnosticLoop
            EDITED      : Daniel Burt       (VLIZ)      24.03.2025
            DESCRIPTION : Plot diagnostics for neural network performance
                          Subplot One illustrates scatter of predicted and true data
                          Subplot Two illustrates the frequency distribution of error magnitudes
            ARGUMENTS   : fpath_output_plot     (string)        filepath for saving output plot
        '''

        # check output plot filepath argument
        if fpath_output_plot is None:
            fpath_output_plot_prefix = './output-plots/ffn-output_model-diagnostic'
        else:
            if '.png' in fpath_output_plot: 
                fpath_output_plot_prefix = {fpath_output_plot[:-4]}
            else:
                fpath_output_plot_prefix = fpath_output_plot

        # determine number of provinces
        number_provinces = int(np.nanmax(self.column_dict['provinces'][:, -1])) + 1

        # loop through provinces
        for province in range(number_provinces):

            # complete output path string
            fpath_output_plot = f'{fpath_output_plot_prefix}_province-{province:02d}.png'

            # determine number of training epochs
            number_training_epochs = len(self.neural_network_history_dictionary[province].history['loss'])
            if number_training_epochs < 100:
                epoch_lim = int(10 * np.ceil(number_training_epochs/ 10))
            elif number_training_epochs >= 100 and number_training_epochs < 250:
                epoch_lim = int(50 * np.ceil(number_training_epochs/ 50))
            elif number_training_epochs >= 250:
                epoch_lim = int(100 * np.ceil(number_training_epochs/ 100))

            # evaluate model using test data
            loss, mean_abs_error = self.neural_network_model_dictionary[province].evaluate(self.neural_network_test_dictionary[province]['testing_input'], 
                                                                                           self.neural_network_test_dictionary[province]['testing_target'], 
                                                                                           verbose = 2)

            # determine predictions from reserved test data
            test_prediction  = self.neural_network_model_dictionary[province].predict(self.neural_network_test_dictionary[province]['testing_input'])

            # determine error magnitudes
            error_magnitudes = self.neural_network_test_dictionary[province]['testing_target'] - test_prediction

            # calculate line of best fit
            slope, intercept = np.polyfit(np.ravel(self.neural_network_test_dictionary[province]['testing_target']), np.ravel(test_prediction), 1)
            best_fit_line    = slope * self.neural_network_test_dictionary[province]['testing_target'] + intercept

            # determine bin boundaries for histogram
            bins_lower_lim = 10 * np.floor(np.nanmin(error_magnitudes)/ 10)
            bins_upper_lim = 10 * np.ceil( np.nanmax(error_magnitudes)/ 10)
            bin_edges      = np.arange(bins_lower_lim, (bins_upper_lim + 10), 10)

            # define fontsizes
            size_suptitle = 22
            size_subtitle = 20
            size_labtitle = 16
            size_ticks    = 12

            # instantiate figure and axes
            fig, axs = plt.subplots(nrows = 1, ncols = 3,
                                    gridspec_kw = {'wspace': 0.25, 'hspace': 0.},
                                    figsize = (21, 7)
                                    )
            
            # set colour of empty space to white
            fig.patch.set_facecolor('white')

            # set figure title
            fig.suptitle(f"Model Diagnostics - Province {province}", fontsize = size_suptitle, fontweight = 'bold', y =0.98)
            fig.text(0.5, 0.89, f"Loss: {loss:.3f}, R$^{2}$: {mean_abs_error:.3f}", ha = 'center', fontsize = size_labtitle)

            # plot subplot one
            _ = axs[0].plot(self.neural_network_history_dictionary[province].history['loss'], 
                            label = 'Training', 
                            color = 'grey', 
                            linestyle = '-', 
                            linewidth = 1.5)
            _ = axs[0].plot(self.neural_network_history_dictionary[province].history['val_loss'], 
                            label = 'Internal Validation', 
                            color = 'black', 
                            linestyle = '-', 
                            linewidth = 1.5)
            
            # define subplot title, labels and ticks
            axs[0].set_title('Training History', fontsize = size_subtitle)
            axs[0].set_xlabel('Epoch', fontsize = size_labtitle)
            axs[0].set_ylabel('Mean Absolute Error (°C)', fontsize = size_labtitle)
            axs[0].tick_params(labelsize = size_ticks)
            axs[0].legend(fontsize = (size_ticks + 2))
            axs[0].set_box_aspect(1)

            # define subplot limits
            axs[0].set_xlim([0, epoch_lim])
            axs[0].set_ylim([0, 10])

            # plot subplot two
            _ = axs[1].scatter(x = self.neural_network_test_dictionary[province]['testing_target'],
                            y = test_prediction,
                            color = 'black',
                            s = 10)
            _ = axs[1].plot(self.neural_network_test_dictionary[province]['testing_target'],
                            best_fit_line,
                            color = 'red',
                            linestyle = '-',
                            linewidth = 1.5)
            _ = axs[1].plot([0, 3000],
                            [0, 3000],
                            color = 'gray',
                            linestyle = '--',
                            linewidth = 1.5)
            
            # define subplot title, labels and ticks
            axs[1].set_title('Observed vs Predicted', fontsize = size_subtitle)
            axs[1].set_xlabel('Observations (°C)', fontsize = size_labtitle)
            axs[1].set_ylabel('Predictions (°C)', fontsize = size_labtitle)
            axs[1].tick_params(labelsize = size_ticks)
            axs[1].set_box_aspect(1)

            # define subplot limits
            axs[1].set_xlim([-5, 45])
            axs[1].set_ylim([-5, 45])

            # plot subplot three
            _ = axs[2].hist(error_magnitudes, 
                            bins = bin_edges, 
                            color = 'black')

            # define subplot title, labels and ticks
            axs[2].set_title('Prediction Error Distribution', fontsize = size_subtitle)
            axs[2].set_xlabel('Magnitude of Prediction Error (°C)', fontsize = size_labtitle)
            axs[2].set_ylabel('Count', fontsize = size_labtitle)
            axs[2].tick_params(labelsize = size_ticks)
            axs[2].set_box_aspect(1)

            # adjust figure padding
            fig.subplots_adjust(top = 0.85, bottom = 0.05)

            # save figure to output directory
            plt.savefig(fpath_output_plot, bbox_inches = 'tight', dpi = 100)

            # close figure to conserve memory
            plt.close()


    def PlotInputData(self, fpath_output_plot = './output-plots/ffn-inputs-mean.png', plot_variable_list = None):
        '''
            NAME        : PlotInputData
            EDITED      : Daniel Burt       (VLIZ)      19.02.2025
            DESCRIPTION : Plot input and observation data
        '''

        # evaluate input array dictionary
        if len(self.input_array_dict) == 0:
            print('ERROR: Input data not found. Please call function: "LoadInputData" before "PlotInputData".')
            exit()

        # evaluate plot variables list
        if plot_variable_list is None:
            print('WARNING: No input variables given for PlotInputData. Input variables will be used sequentially to a maximum of four variables.')
            plot_variable_list = list(self.input_array_dict.keys())
        elif not isinstance(plot_variable_list, list):
            print('WARNING: Input variables not provided in list format. Input variables will be used sequentially to a maximum of four variables.')
            plot_variable_list = list(self.input_array_dict.keys())
        elif isinstance(plot_variable_list, list):

            # instantiate boolean
            unmatched_variable_found = False

            # loop through plot variable list
            for var in plot_variable_list:

                # evaluate input dictionary
                if var not in self.input_array_dict.keys():
                    unmatched_variable_found = True

            # evaluate unmatched variable detection
            if unmatched_variable_found:
                print('WARNING: Listed variables do not match available inputs. Input variables will be used sequentially to a maximum of four variables.')
                plot_variable_list = list(self.input_array_dict.keys())

        # evaluate length of plot variable list
        if len(plot_variable_list) > 4:
            print('WARNING: Exceeded maximum number of input variables for plotting. Only the first four input variables will be plotted.')
            plot_variable_list = list(plot_variable_list)[:4]

        # silence warnings from land points  -->>  FIXME are ALL points land or is it more important to ignore points with NaNs for all months?
        with warnings.catch_warnings():

            # ignore runtime warnings produced by means of empty slices
            warnings.simplefilter('ignore', category = RuntimeWarning)

            # instantiate mean dictionary
            plot_variable_dict = dict()

            # loop through plot variables
            for plot_variable in plot_variable_list:

                # calculate mean
                plot_variable_dict[plot_variable] = np.nanmean(self.input_array_dict[plot_variable], axis = 0)

        # instantiate projection
        data_crs = cr.crs.PlateCarree()

        # instantiate figure and axes
        fig, axs = plt.subplots(nrows = 2, ncols = 2,
                                subplot_kw = {'projection': cr.crs.Robinson(central_longitude = 0)},
                                gridspec_kw = {'wspace': 0.01, 'hspace': -0.35},
                                figsize = (22, 18.5)
                                )
        
        # set colour of empty space to white
        fig.patch.set_facecolor('white')

        # flatten axes for simplicity
        axs = np.ravel(axs)

        # loop through plot variables
        for plot_variable in plot_variable_list:

            # retrieve index
            idx = plot_variable_list.index(plot_variable)

            # plot contour map of province mode
            plot_mean = axs[idx].pcolormesh(
                                            self.input_array_dict['lon'],
                                            self.input_array_dict['lat'],
                                            plot_variable_dict[plot_variable][:, :],
                                            transform = data_crs,
                                            cmap = plt.cm.get_cmap("jet", 20),
                                            # vmax = 16,
                                            # vmin = 0
                                            )
            
            # configure subplot
            plot_gridlines = axs[idx].gridlines(linewidth = 0.5, color = 'k')
            plot_gridlines.xlocator = tick.FixedLocator(range(-180, 181, 30))
            plot_gridlines.ylocator = tick.FixedLocator(range(-90, 91, 15))

            # Add colourbar
            cbar = fig.colorbar(plot_mean, ax = axs[idx], orientation = 'horizontal', fraction = 0.03, pad = 0.02, aspect = 80, shrink = 0.70)
            
            # modify colourbar labels
            cbar.set_label(plot_variable, fontsize = 20)  # Label for the colorbar
            cbar.ax.tick_params(labelsize = 16)

        plt.savefig(fpath_output_plot, bbox_inches = 'tight', dpi = 100)

        plt.close()


    def PlotPrediction(self, plot_type = 'mean-comparison', fpath_output_plot = None):
        '''
            NAME        : PlotPrediction
            EDITED      : Daniel Burt       (VLIZ)      04.02.2025
            DESCRIPTION : Control function for prediction plotting functions.
            ARGUMENTS   : plot_type             (string)        pointer for selecting plotting functions
                                                                valid inputs: 'mean-comparison', 'mean-variability'
                          fpath_output_plot     (string)        filepath for saving output plot
        '''

        # check output plot filepath argument
        if fpath_output_plot is None:
            self.fpath_output_plot = f"./output-plots/ffn-output_pco2-estimate_{plot_type}.png"
        else:
            self.fpath_output_plot = fpath_output_plot

        # check function arguments for function calls
        if plot_type == 'mean-comparison':

            self.PlotPredictionMeanComp()

        elif plot_type == 'mean-variability':

            self.PlotPredictionMeanVar()

        else:

            print("ERROR: plotting prompt not recognised. Please ented valid prompt from selection: 'mean'")
            exit()

    
    def PlotPredictionMeanComp(self):
        '''
            NAME        : PlotPredictionMean
            EDITED      : Daniel Burt       (VLIZ)      06.02.2025
            DESCRIPTION : Plot mean pCO2 estimate and comparison with observational dataset
        '''

        # process warnings
        with warnings.catch_warnings():

            # ignore runtime warnings produced by means of empty slices
            warnings.simplefilter('ignore', category = RuntimeWarning)

            # calculate temporal mean of pco2 estimate
            pco2_estimate_mean = np.nanmean(self.pco2_estimate, axis = 0)

            # determine bounds of pco2 estimate plot  -->> FIXME make dynamic
            vmax_mean = 440
            vmin_mean = 280
            mean_step = 40

            # calculate temporal variability of pco2 estimate
            pco2_estimate_sdev = np.nanstd(self.pco2_estimate, axis  = 0)

            # determine upper bounds of variability plot  -->> FIXME make dynamic
            vmax_sdev = 100

        # reload observational dataset
        if os.path.isfile('./input-data/SOCATv2024.mat'):

            # load SOCATv2024 data from MATLAB data file
            arr_obs = sp.io.loadmat('./input-data/SOCATv2024.mat')['fco2'].astype('float32')

        # TODO provide user configuration of filename
        elif os.path.isfile('./input-data/SOCATv2024_tracks_gridded_monthly.nc'):

            # load observation data
            arr_obs = xr.load_dataset('./input-data/SOCATv2024_tracks_gridded_monthly.nc')['fco2_ave_weighted'].astype('float32')

        else:
            print("ERROR: Observational dataset not found in directory 'input-data'. Please locate observational dataset in 'input-data'.")

        # process warnings
        with warnings.catch_warnings():

            # ignore runtime warnings produced by means of empty slices
            warnings.simplefilter('ignore', category = RuntimeWarning)

            # determine observational mean
            arr_obs_mean = np.nanmean(arr_obs, axis = 0)

            # determine observational mean delta
            arr_obs_mean_delta = pco2_estimate_mean - arr_obs_mean

            # determine observational variability
            arr_obs_sdev = np.nanstd(arr_obs, axis = 0)

            # determine mask for valid axes
            threshold    = 20
            valid_counts = np.count_nonzero(~np.isnan(arr_obs), axis = 0)
            mask         = valid_counts >= threshold

            # apply valid mask
            arr_obs_sdev[mask] = np.nan

            # calculate delta
            arr_obs_sdev_delta = pco2_estimate_sdev - arr_obs_sdev

            # remove observational dataset from memory
            del arr_obs

        # reload Takahashi dataset
        if os.path.isfile('./input-data/Taka_pCO2_eth_v2024.mat'):

            # load SOCATv2024 data from MATLAB data file
            arr_taka = sp.io.loadmat('./input-data/Taka_pCO2_eth_v2024.mat')['data_taka'].astype('float32')

        # TODO provide user configuration of filename
        elif os.path.isfile('./input-data/Taka_pCO2_eth_v2024.nc'):

            # load observation data
            arr_taka = xr.load_dataset('./input-data/Taka_pCO2_eth_v2024.nc')['data_taka'].astype('float32')

        else:
            print("ERROR: Observational dataset not found in directory 'input-data'. Please locate Takahashi dataset in 'input-data'.")

        # process warnings
        with warnings.catch_warnings():

            # ignore runtime warnings produced by means of empty slices
            warnings.simplefilter('ignore', category = RuntimeWarning)

            # determine Taka dataset mean
            arr_taka_mean = np.nanmean(arr_taka, axis = 0)

            # determine Taka mean delta
            arr_taka_mean_delta = pco2_estimate_mean - arr_taka_mean

            # determine Taka dataset standard deviation
            arr_taka_sdev = np.nanstd(arr_taka, axis = 0)

            # calculate delta
            arr_taka_sdev_delta = pco2_estimate_sdev - arr_taka_sdev

            # remove Taka dataset from memory
            del arr_taka

            # # determine bounds of mean delta plot  -->> FIXME make dynamic
            vmax_delta =  52.5
            vmin_delta = -52.5
            delta_step =  20

        # instantiate projection
        data_crs = cr.crs.PlateCarree()

        # instantiate figure and axes
        fig, axs = plt.subplots(nrows = 3, ncols = 2,
                                subplot_kw = {'projection': cr.crs.Robinson(central_longitude = 0)},
                                gridspec_kw = {'wspace': 0.01, 'hspace': 0.2},
                                figsize = (22, 18.5)
                                )

        # set colour of empty space to white
        fig.patch.set_facecolor('white')

        # flatten axes for simplicity
        axs = np.ravel(axs)

        # plot contour map of Prediction Mean
        plot_prediction_mean = axs[0].pcolormesh(self.input_array_dict['lon'],
                                                self.input_array_dict['lat'],
                                                pco2_estimate_mean[:, :],
                                                transform = data_crs,
                                                cmap = plt.cm.get_cmap("jet", 20),
                                                vmax = vmax_mean,
                                                vmin = vmin_mean
                                                )

        # configure subplot gridlines
        plot_gridlines = axs[0].gridlines(linewidth = 0.5, color = 'k')
        plot_gridlines.xlocator = tick.FixedLocator(range(-180, 181, 30))
        plot_gridlines.ylocator = tick.FixedLocator(range(-90, 91, 15))

        # configure subplot
        axs[0].set_facecolor('black')

        # Add colourbar
        cbar = fig.colorbar(plot_prediction_mean, 
                            ax = axs[0], 
                            orientation = 'horizontal', 
                            extend = 'both',
                            fraction = 0.03, 
                            pad = 0.02, 
                            aspect = 80, 
                            shrink = 0.70)

        # modify colourbar labels
        cbar.set_label(r'$p$CO$_{2}$ Mean (ppm)', fontsize = 20)  # Label for the colorbar
        cbar.set_ticks(np.arange(vmin_mean, (vmax_mean + mean_step), mean_step))
        cbar.ax.tick_params(labelsize = 16)

        # plot contour map of Prediction Variability
        plot_prediction_variability = axs[1].pcolormesh(self.input_array_dict['lon'],
                                                        self.input_array_dict['lat'],
                                                        pco2_estimate_sdev[:, :],
                                                        transform = data_crs,
                                                        cmap = plt.cm.get_cmap("jet", 20),
                                                        vmax = vmax_sdev,
                                                        vmin = 0
                                                        )

        # configure subplot gridlines
        plot_gridlines = axs[1].gridlines(linewidth = 0.5, color = 'k')
        plot_gridlines.xlocator = tick.FixedLocator(range(-180, 181, 30))
        plot_gridlines.ylocator = tick.FixedLocator(range(-90, 91, 15))

        # configure subplot
        axs[1].set_facecolor('black')

        # Add colourbar
        cbar = fig.colorbar(plot_prediction_variability, 
                            ax = axs[1], 
                            orientation = 'horizontal', 
                            extend = 'max',
                            fraction = 0.03, 
                            pad = 0.02, 
                            aspect = 80, 
                            shrink = 0.70)

        # modify colourbar labels
        cbar.set_label(r'$p$CO$_{2}$ $\sigma$ (ppm)', fontsize = 20)  # Label for the colorbar
        cbar.ax.tick_params(labelsize = 16)

        # plot contour map of SOCAT Mean Delta
        plot_obs_mean_delta = axs[2].pcolormesh(self.input_array_dict['lon'],
                                                self.input_array_dict['lat'],
                                                arr_obs_mean_delta[:, :],
                                                transform = data_crs,
                                                cmap = plt.cm.get_cmap("bwr", 21),
                                                vmax = vmax_delta,
                                                vmin = vmin_delta
                                                )

        # configure subplot gridlines
        plot_gridlines = axs[2].gridlines(linewidth = 0.5, color = 'k')
        plot_gridlines.xlocator = tick.FixedLocator(range(-180, 181, 30))
        plot_gridlines.ylocator = tick.FixedLocator(range(-90, 91, 15))

        # configure subplot
        axs[2].set_facecolor('black')

        # Add colourbar
        cbar = fig.colorbar(plot_obs_mean_delta, 
                            ax = axs[2], 
                            orientation = 'horizontal', 
                            extend = 'both',
                            fraction = 0.03, 
                            pad = 0.02, 
                            aspect = 80, 
                            shrink = 0.70)

        # modify colourbar labels
        cbar.set_label(r'$\Delta p$CO$_{2}$ Mean [Prediction - SOCAT] (ppm)', fontsize = 20)  # Label for the colorbar
        cbar.set_ticks(np.arange(vmin_delta + 12.5, (vmax_delta - 12.5 + delta_step), delta_step))
        cbar.ax.tick_params(labelsize = 16)

        # plot contour map of SOCAT Variability Delta
        plot_obs_sdev_delta = axs[3].pcolormesh(self.input_array_dict['lon'],
                                                self.input_array_dict['lat'],
                                                arr_obs_sdev_delta[:, :],
                                                transform = data_crs,
                                                cmap = plt.cm.get_cmap("bwr", 21),
                                                vmax = vmax_delta,
                                                vmin = vmin_delta
                                                )

        # configure subplot gridlines
        plot_gridlines = axs[3].gridlines(linewidth = 0.5, color = 'k')
        plot_gridlines.xlocator = tick.FixedLocator(range(-180, 181, 30))
        plot_gridlines.ylocator = tick.FixedLocator(range(-90, 91, 15))

        # configure subplot
        axs[3].set_facecolor('black')

        # Add colourbar
        cbar = fig.colorbar(plot_obs_sdev_delta, 
                            ax = axs[3], 
                            orientation = 'horizontal', 
                            extend = 'both',
                            fraction = 0.03, 
                            pad = 0.02, 
                            aspect = 80, 
                            shrink = 0.70)

        # modify colourbar labels
        cbar.set_label(r'$\Delta p$CO$_{2}$ $\sigma$ [Prediction - SOCAT] (ppm)', fontsize = 20)  # Label for the colorbar
        cbar.set_ticks(np.arange(vmin_delta + 12.5, (vmax_delta - 12.5 + delta_step), delta_step))
        cbar.ax.tick_params(labelsize = 16)

        # plot contour map of Takahashi Mean Delta
        plot_taka_mean_delta = axs[4].pcolormesh(self.input_array_dict['lon'],
                                                 self.input_array_dict['lat'],
                                                 arr_taka_mean_delta[:, :],
                                                 transform = data_crs,
                                                 cmap = plt.cm.get_cmap("bwr", 21),
                                                 vmax = vmax_delta,
                                                 vmin = vmin_delta
                                                 )

        # configure subplot gridlines
        plot_gridlines = axs[4].gridlines(linewidth = 0.5, color = 'k')
        plot_gridlines.xlocator = tick.FixedLocator(range(-180, 181, 30))
        plot_gridlines.ylocator = tick.FixedLocator(range(-90, 91, 15))

        # configure subplot
        axs[4].set_facecolor('black')

        # Add colourbar
        cbar = fig.colorbar(plot_taka_mean_delta, 
                            ax = axs[4], 
                            orientation = 'horizontal', 
                            extend = 'both',
                            fraction = 0.03, 
                            pad = 0.02, 
                            aspect = 80, 
                            shrink = 0.70)

        # modify colourbar labels
        cbar.set_label(r'$\Delta p$CO$_{2}$ Mean [Prediction - Takahashi] (ppm)', fontsize = 20)  # Label for the colorbar
        cbar.set_ticks(np.arange(vmin_delta + 12.5, (vmax_delta - 12.5 + delta_step), delta_step))
        cbar.ax.tick_params(labelsize = 16)

        # plot contour map of Takahashi Variability Delta
        plot_taka_sdev_delta = axs[5].pcolormesh(self.input_array_dict['lon'],
                                                 self.input_array_dict['lat'],
                                                 arr_taka_sdev_delta[:, :],
                                                 transform = data_crs,
                                                 cmap = plt.cm.get_cmap("bwr", 21),
                                                 vmax = vmax_delta,
                                                 vmin = vmin_delta
                                                 )

        # configure subplot gridlines
        plot_gridlines = axs[5].gridlines(linewidth = 0.5, color = 'k')
        plot_gridlines.xlocator = tick.FixedLocator(range(-180, 181, 30))
        plot_gridlines.ylocator = tick.FixedLocator(range(-90, 91, 15))

        # configure subplot
        axs[5].set_facecolor('black')

        # Add colourbar
        cbar = fig.colorbar(plot_taka_sdev_delta, 
                            ax = axs[5], 
                            orientation = 'horizontal', 
                            extend = 'both',
                            fraction = 0.03, 
                            pad = 0.02, 
                            aspect = 80, 
                            shrink = 0.70)

        # modify colourbar labels
        cbar.set_label(r'$\Delta p$CO$_{2}$ $\sigma$ [Prediction - Takahashi] (ppm)', fontsize = 20)  # Label for the colorbar
        cbar.set_ticks(np.arange(vmin_delta + 12.5, (vmax_delta - 12.5 + delta_step), delta_step))
        cbar.ax.tick_params(labelsize = 16)

        plt.savefig(self.fpath_output_plot, bbox_inches = 'tight', dpi = 100)

        plt.close()

    
    def PlotPredictionMeanVar(self):
        '''
            NAME        : PlotPredictionMeanVar
            EDITED      : Daniel Burt       (VLIZ)      04.02.2025
            DESCRIPTION : Plot mean pCO2 estimate 
        '''

        # process warnings
        with warnings.catch_warnings():

            # ignore runtime warnings produced by means of empty slices
            warnings.simplefilter('ignore', category = RuntimeWarning)

            # calculate temporal mean of pco2 estimate
            pco2_estimate_mean = np.nanmean(self.pco2_estimate, axis = 0)

            # determine bounds of pco2 estimate plot
            vmax_mean = np.nanmax(pco2_estimate_mean)
            vmin_mean = np.nanmin(pco2_estimate_mean)

            # calculate temporal variability of pco2 estimate
            pco2_estimate_sdev = np.nanstd(self.pco2_estimate, axis  = 0)

            # determine upper bounds of variability plot
            vmax_sdev = np.nanmax(pco2_estimate_sdev)

        # instantiate projection
        data_crs = cr.crs.PlateCarree()

        # instantiate figure and axes
        fig, axs = plt.subplots(nrows = 2, ncols = 1,
                                subplot_kw = {'projection': cr.crs.Robinson(central_longitude = 0)},
                                gridspec_kw = {'wspace': 0.01, 'hspace': 0.15},
                                figsize = (22, 18.5)
                                )
        
        # set colour of empty space to white
        fig.patch.set_facecolor('white')

        # flatten axes for simplicity
        axs = np.ravel(axs)

        # plot contour map of province mode
        plot_prediction_mean = axs[0].pcolormesh(
                                                 self.input_array_dict['lon'],
                                                 self.input_array_dict['lat'],
                                                 pco2_estimate_mean[:, :],
                                                 transform = data_crs,
                                                 cmap = plt.cm.get_cmap("tab20", 20),
                                                 vmax = vmax_mean,
                                                 vmin = vmin_mean
                                                 )
        
        # configure subplot
        plot_gridlines = axs[0].gridlines(linewidth = 0.5, color = 'k')
        plot_gridlines.xlocator = tick.FixedLocator(range(-180, 181, 30))
        plot_gridlines.ylocator = tick.FixedLocator(range(-90, 91, 15))

        # Add colourbar
        cbar = fig.colorbar(plot_prediction_mean, ax = axs[0], orientation = 'horizontal', fraction = 0.03, pad = 0.02, aspect = 80, shrink = 0.70)
        
        # modify colourbar labels
        cbar.set_label('pCO$_{2}$ Mean (ppm)', fontsize = 20)  # Label for the colorbar
        cbar.ax.tick_params(labelsize = 16)

        # plot contour map of province mode
        plot_prediction_variability = axs[1].pcolormesh(
                                                        self.input_array_dict['lon'],
                                                        self.input_array_dict['lat'],
                                                        pco2_estimate_sdev[:, :],
                                                        transform = data_crs,
                                                        cmap = plt.cm.get_cmap("tab20", 20),
                                                        vmax = vmax_sdev,
                                                        vmin = 0
                                                        )
        
        # configure subplot
        plot_gridlines = axs[1].gridlines(linewidth = 0.5, color = 'k')
        plot_gridlines.xlocator = tick.FixedLocator(range(-180, 181, 30))
        plot_gridlines.ylocator = tick.FixedLocator(range(-90, 91, 15))
            
        # Add colourbar
        cbar = fig.colorbar(plot_prediction_variability, ax = axs[1], orientation = 'horizontal', fraction = 0.03, pad = 0.02, aspect = 80, shrink = 0.70)
        
        # modify colourbar labels
        cbar.set_label('pCO$_{2}$ Variability (ppm)', fontsize = 20)  # Label for the colorbar
        cbar.ax.tick_params(labelsize = 16)

        plt.savefig(self.fpath_output_plot, bbox_inches = 'tight', dpi = 100)

        plt.close()

