############################################################################################
####                                                                                    ####
####    NAME         : STEP4_main_matlab_ffn.py                                         ####
####    EDITED       : Peter Landschuetzer      UEA     peter.landschuetzer@vliz.be     ####
####                    - Initial development and formulation in MATLAB                 ####
####                   Soren Berger             MPI-M       soren.berger@mpimet.mpg.de  ####
####                    - Initial translation from MATLAB to Python                     #### 
####                   Andrea van Langen Roson  VLIZ        andrea.van.langen@vliz.be   ####
####                   Maurie Keppens           VLIZ        maurie.keppens@vliz.be      ####
####                   Daniel Burt              VLIZ        daniel.burt@vliz.be         ####
####                    - Refactoring of Python code                                    ####
####                    - Adding new functions                                          ####
####    LAST EDIT    : 19.02.2025                                                       ####
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
####                    - SciPy 1.11.4                                                  ####
####                    - Sklearn 1.4.1.post1                                           ####
####                    - TensorFlow 2.18.0 (https://www.tensorflow.org/)               ####
####                    - Xarray 2024.2.0                                               ####
####                                                                                    ####
####                                                                                    ####
############################################################################################


####  IMPORT PACKAGES
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cartopy as cr
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
import scipy as sp
import tensorflow as tf
import time
import warnings
import xarray as xr

#### IMPORT MODULES
from sklearn import model_selection
from tensorflow import keras


####  DEFINE CLASS
class FeedForwardNetwork:
    '''
        NAME        : FeedForwardNetwork
        CONTAINS    :   - LoadInputData         : Load predictor and observation data
                        - CropInputData         : Control crop functions for loaded input datasets
                        - CropInputDataTime     : Crop loaded datasets to specified time range
                        - CropInputDataGeo      : Crop loaded datasets to specified latitude and longitude ranges
                        - PlotInputData         : Plot predictor and observation data
                        - GenerateYearMonthList : Create array for time [year month] data TODO FIXME why is this necessary?
                        - ExtendArrays          : Extend input data arrays along time axis
                        -                       : Reshape and rearrange predictor and observation data for input to neural network TODO
                        -                       : Create Training and Labelling/ Test/ Validation/ Estimation datasets TODO
                        -                       : Split Training dataset into Training and Test datasets TODO
                        -                       : Make and train neural network using TensorFlow's KERAS TODO
                        -                       : Predict pCO2 from neural network TODO
                        -                       : Plot predicted pCO2 TODO
    '''

    def LoadInputData(self, input_dictionary):
        '''
            NAME        : LoadInputData
            EDITED      : Daniel Burt       (VLIZ)      19.02.2025
            DESCRIPTION : Function for loading data for training and validation
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


    def PrepareInputs(self):
        '''
            NAME        : PrepareInputs
            EDITED      : Daniel Burt       (VLIZ)      21.02.2025
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
        self.array_shape = (longest_length, 180, 360)

        # instantiate dictionary of extended arrays
        extended_input_array_dict = dict()

        # loop through input variables
        for input_variable in list(self.input_array_dict.keys()):

            # evaluate input variable
            if input_variable not in ['year', 'month', 'lat', 'lon', 'provinces']:

                # extend input data arrays
                extended_input_array_dict[input_variable] = self.ExtendArrays(self.input_array_dict[input_variable], self.array_shape)

                # clear original input arrays from memory
                del self.input_array_dict[input_variable]

            # evaluate input variable
            elif input_variable in ['lat', 'lon']:

                # instantiate extended arrays
                extended_input_array_dict[input_variable] = np.zeros(self.array_shape, dtype = np.float16)

                # extend latitude and longitude arrays
                extended_input_array_dict[input_variable][:] = self.input_array_dict[input_variable]

                # clear original input arrays from memory
                del self.input_array_dict[input_variable]

        # instantiate extended arrays
        extended_input_array_dict['year']      = np.zeros(self.array_shape, dtype = np.float16)
        extended_input_array_dict['month']     = np.zeros(self.array_shape, dtype = np.float16)
        extended_input_array_dict['provinces'] = np.zeros(self.array_shape, dtype = np.float16)

        # determine start year
        start_year = int(2024 - (longest_length// 12))  # FIXME based on v2024

        # instantiate months
        months = np.arange(1, 13)
        broadcast_months = months[:, np.newaxis, np.newaxis]

        # loop through number of years of data
        for t in range(0, longest_length, 12):

            # broadcast month and year to extended array
            extended_input_array_dict['month'][t:t+12, :, :]     = broadcast_months
            extended_input_array_dict['year'][t:t+12, :, :]      = start_year + t//12

            # broadcast provinces to extended array
            extended_input_array_dict['provinces'][t:t+12, :, :] = self.input_array_dict['provinces']

        # clear original input arrays from memory
        del self.input_array_dict['provinces']

        # instantiate lists and dictionaries
        self.column_dict         = dict()
        columnised_input_list    = []
        valid_mask_training_list = []
        valid_mask_estimate_list = []

        # loop through dictionary keys
        for input_variable in list(extended_input_array_dict.keys()):

            # evaluate for observations key
            if input_variable is not self.observation_variable:

                # flatten input arrays into one dimensional columns
                columnised_input_list.append(np.ravel(extended_input_array_dict[input_variable]).reshape(-1, 1))
                
                # store valid masks
                valid_mask_estimate_list.append(np.isnan(np.ravel(extended_input_array_dict[input_variable]).reshape(-1, 1)))

            # store valid masks
            valid_mask_training_list.append(np.isnan(np.ravel(extended_input_array_dict[input_variable]).reshape(-1, 1)))

            # evaluate for variables to keep
            if input_variable == self.observation_variable or input_variable == 'provinces':

                self.column_dict[input_variable] = np.ravel(extended_input_array_dict[input_variable]).reshape(-1, 1)

            # clear extended input arrays from memory
            del extended_input_array_dict[input_variable]

        # combine valid masks  -->> propagate boolean values: True  -->>  FIXME should province be included in valid masking?
        valid_mask_training      = np.ravel(np.logical_or.reduce(valid_mask_training_list))
        self.valid_mask_estimate = np.ravel(np.logical_or.reduce(valid_mask_estimate_list))

        # stack one dimensional columns into vectors of [year, month, lat, lon, columns of input datasets, obs, province]
        dataset_training_full = np.column_stack(columnised_input_list)

        # remove NaNs from datasets
        self.dataset_estimate  = dataset_training_full[~self.valid_mask_estimate]
        self.province_estimate = self.column_dict['provinces'][~self.valid_mask_estimate]
        self.dataset_training  = dataset_training_full[~valid_mask_training]
        self.province_training = self.column_dict['provinces'][~valid_mask_training]
        self.dataset_validate  = self.column_dict[self.observation_variable][~valid_mask_training]

        # clear input columns from memory
        del columnised_input_list, valid_mask_training_list, valid_mask_estimate_list

        # report shapes for diagnostics
        print('Diagnostic : Shape of datasets')
        print(f"Dummy      : {dataset_training_full.shape} - {dataset_training_full.nbytes / (1024**2):.3f} MB")
        print(f"Training   : {self.dataset_training.shape} - {self.dataset_training.nbytes / (1024**2):.3f} MB")
        print(f"Estimation : {self.dataset_estimate.shape} - {self.dataset_estimate.nbytes / (1024**2):.3f} MB")
        print('')


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
        

    def MakeTrainModel(self, number_hidden_neurons = 60, activation_function = 'relu', length_patience = 10, epochs = 500):
        '''
            NAME        : TrainModel
            EDITED      : Daniel Burt   (VLIZ)      03.02.2025
            DESCRIPTION : Make and train machine learning model
            ARGUMENTS   : number_hidden_neurons     (int)           number of neurons in hidden layer of machine learning model
                          activation_function       (string)        selected activation function corresponding to TensorFlow keras: https://keras.io/api/layers/activations/
                          learning_rate             (float)         learning rate of machine learning model
                          epochs                    (int)           number of times machine learning model presented with data
        '''

        # set random seed for reproducibility
        tf.random.set_seed(1)

        # subsample training dataset for training and test subsets  -->> Necessary? FIXME
        training_train, self.training_test, validate_train, self.validate_test = model_selection.train_test_split(self.dataset_training, 
                                                                                                                  self.dataset_validate, 
                                                                                                                  test_size = 0.2)

        # initialise time recording for diagnostic reporting
        time_ini = time.time()

        # make normalization layer  -->> what is the normalization layer?? TODO
        normalizer = keras.layers.Normalization(axis = -1)

        # compute mean and variance of train subsample of training dataset
        normalizer.adapt(training_train)

        # instantiate neural network model
        self.neural_network_model = keras.Sequential([normalizer,
                                                      keras.layers.Dense(units = number_hidden_neurons , activation = activation_function),
                                                      keras.layers.Dense(units = 1, activation = 'linear')
                                                      ])
        
        # compile neural network model
        self.neural_network_model.compile(loss = 'mean_absolute_error',
                                          optimizer = keras.optimizers.Adam(0.001),
                                          metrics = ['R2Score']
                                          )
        
        # report summary  -->> what does this do? TODO
        self.neural_network_model.summary()

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
        self.training_history = self.neural_network_model.fit(x = training_train,
                                                              y = validate_train,
                                                              validation_split = 0.2,
                                                              verbose = 2,
                                                              callbacks = [stop_condition],
                                                              epochs = epochs
                                                              )
                
        # end timer and report timing diagnostic
        time_fin = time.time()
        time_elapsed = time_fin - time_ini
        print(f"Feed Forward Network training ended after {time_elapsed} seconds")
        print('')
        

    def MakeTrainModelLoop(self, number_hidden_neurons = 60, activation_function = 'relu', length_patience = 10, epochs = 500):
        '''
            NAME        : TrainModel
            EDITED      : Daniel Burt   (VLIZ)      21.02.2025
            DESCRIPTION : Make and train machine learning model
            ARGUMENTS   : number_hidden_neurons     (int)           number of neurons in hidden layer of machine learning model
                          activation_function       (string)        selected activation function corresponding to TensorFlow keras: https://keras.io/api/layers/activations/
                          learning_rate             (float)         learning rate of machine learning model
                          epochs                    (int)           number of times machine learning model presented with data
        '''

        # instantiate dictionary
        self.neural_network_model_dictionary = dict()
        self.province_dict = dict()

        # set random seed for reproducibility
        tf.random.set_seed(1)

        # determine number of provinces
        number_provinces = int(np.nanmax(self.column_dict['provinces'][:, -1])) + 1

        # loop through provinces
        for province in range(number_provinces):

            # subsample training dataset for training and test subsets
            training_train, self.training_test, validate_train, self.validate_test = model_selection.train_test_split(self.dataset_training[self.province_training.ravel() == province], 
                                                                                                                      self.dataset_validate[self.province_training.ravel() == province], 
                                                                                                                      test_size = 0.2)

            # initialise time recording for diagnostic reporting
            time_ini = time.time()

            # make normalization layer  -->> what is the normalization layer?? TODO
            normalizer = keras.layers.Normalization(axis = -1)

            # compute mean and variance of train subsample of training dataset
            normalizer.adapt(training_train)

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
            self.training_history = self.neural_network_model_dictionary[province].fit(x = training_train,
                                                                                       y = validate_train,
                                                                                       validation_data = (self.training_test, self.validate_test),
                                                                                       verbose = 2,
                                                                                       callbacks = [stop_condition],
                                                                                       epochs = epochs
                                                                                       )
                    
            # end timer and report timing diagnostic
            time_fin = time.time()
            time_elapsed = time_fin - time_ini
            print(f"Feed Forward Network training ended after {time_elapsed} seconds")
            print('')

    
    def PredictEstimate(self, fpath_output = None, fileext = None):
        '''
            NAME        : PredictEstimate
            EDITED      : Daniel Burt       (VLIZ)      04.02.2025
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
        ffn_estimate = self.neural_network_model.predict(self.dataset_estimate)

        # map one-dimensional neural-network estimate back into three-dimensions (time, lat, lon)
        self.pco2_estimate = np.full(self.array_shape, np.nan, dtype = np.float32)
        self.pco2_estimate = np.ravel(self.pco2_estimate)
        self.pco2_estimate[~self.valid_mask_estimate] = np.ravel(ffn_estimate)
        self.pco2_estimate = self.pco2_estimate.reshape(self.array_shape)

        # save estimate to data file
        if fileext == 'mat':

            # configure dictionary
            out_dict = {'pco2_estimate': self.pco2_estimate,
                        'lat': self.arr_lat,
                        'lon': self.arr_lon
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
            DA_out.to_netcdf(f"{fpath_output}/ffn-output_pco2-estimate.nc")

    
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
            ffn_estimate[self.province_estimate.ravel() == province] = self.neural_network_model_dictionary[province].predict(self.dataset_estimate[self.province_estimate.ravel() == province])

        # map one-dimensional neural-network estimate back into three-dimensions (time, lat, lon)
        self.pco2_estimate = np.full(self.array_shape, np.nan, dtype = np.float32)
        self.pco2_estimate = np.ravel(self.pco2_estimate)
        self.pco2_estimate[~self.valid_mask_estimate] = np.ravel(ffn_estimate)
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


    def PlotDiagnostic(self, fpath_output_plot = None):
        '''
            NAME        : PlotDiagnostic
            EDITED      : Daniel Burt       (VLIZ)      04.02.2025
            DESCRIPTION : Plot diagnostics for neural network performance
                          Subplot One illustrates scatter of predicted and true data
                          Subplot Two illustrates the frequency distribution of error magnitudes
            ARGUMENTS   : fpath_output_plot     (string)        filepath for saving output plot
        '''

        # check output plot filepath argument
        if fpath_output_plot is None:
            fpath_output_plot = './output-plots/ffn-output_model-diagnostic.png'
        else:
            fpath_output_plot = fpath_output_plot

        # determine number of training epochs
        number_training_epochs = len(self.training_history.history['loss'])
        if number_training_epochs < 100:
            epoch_lim = int(10 * np.ceil(number_training_epochs/ 10))
        elif number_training_epochs >= 100 and number_training_epochs < 250:
            epoch_lim = int(50 * np.ceil(number_training_epochs/ 50))
        elif number_training_epochs >= 250:
            epoch_lim = int(100 * np.ceil(number_training_epochs/ 100))

        # evaluate model using test data
        loss, mean_abs_error = self.neural_network_model.evaluate(self.training_test, 
                                                                  self.validate_test, 
                                                                  verbose = 2)

        # determine predictions from reserved test data
        test_prediction  = self.neural_network_model.predict(self.training_test)

        # determine error magnitudes
        error_magnitudes = self.validate_test - test_prediction

        # calculate line of best fit
        slope, intercept = np.polyfit(np.ravel(self.validate_test), np.ravel(test_prediction), 1)
        best_fit_line    = slope * self.validate_test + intercept

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
        fig.text(0.5, 0.89, f"Loss: {loss:.3f}, R$^{2}$: {mean_abs_error:.3f}", ha = 'center', fontsize = size_labtitle)

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
        axs[0].set_ylabel('Mean Absolute Error (ppm)', fontsize = size_labtitle)
        axs[0].tick_params(labelsize = size_ticks)
        axs[0].legend(fontsize = (size_ticks + 2))
        axs[0].set_box_aspect(1)

        # define subplot limits
        axs[0].set_xlim([0, epoch_lim])
        axs[0].set_ylim([0, 100])

        # plot subplot two
        _ = axs[1].scatter(x = self.validate_test,
                           y = test_prediction,
                           color = 'black',
                           s = 10)
        _ = axs[1].plot(self.validate_test,
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
        axs[1].set_xlabel('Observations (ppm)', fontsize = size_labtitle)
        axs[1].set_ylabel('Predictions (ppm)', fontsize = size_labtitle)
        axs[1].tick_params(labelsize = size_ticks)
        axs[1].set_box_aspect(1)

        # define subplot limits
        axs[1].set_xlim([0, 800])
        axs[1].set_ylim([0, 800])

        # plot subplot three
        _ = axs[2].hist(error_magnitudes, 
                        bins = bin_edges, 
                        color = 'black')

        # define subplot title, labels and ticks
        axs[2].set_title('Prediction Error Distribution', fontsize = size_subtitle)
        axs[2].set_xlabel('Magnitude of Prediction Error (ppm)', fontsize = size_labtitle)
        axs[2].set_ylabel('Count', fontsize = size_labtitle)
        axs[2].tick_params(labelsize = size_ticks)
        axs[2].set_box_aspect(1)

        # adjust figure padding
        fig.subplots_adjust(top = 0.85, bottom = 0.05)

        # save figure to output directory
        plt.savefig(fpath_output_plot, bbox_inches = 'tight', dpi = 100)


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

        # reload observational dataset TODO - temp fix
        if os.path.isfile('../input-data/pco2obsV2018.mat'):

            # load SOCATv2024 data from MATLAB data file
            arr_obs = sp.io.loadmat('../input-data/pco2obsV2018.mat')['fco2'].astype('float32')

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

