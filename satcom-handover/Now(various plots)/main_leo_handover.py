# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:04:01 2023

@author: shada
"""
# Importing necessary modules


#from drl_frameworks import DRL
from coordinates import GeoCordinates
from leo_parameters import LEO_Parameters
from date_time_array import DateTimeArray
from tle_loader import TLE_Loader

#%% Define the DRL Framework

num_episodes = 500

ue_geo_position = GeoCordinates(40 , 116, 0)
del_t = 5 # simulation time in minutes
num_times = 30 # number of simulation samples 
# 10초 = 한 time slot

#drl = DRL(num_episodes)


#%% Training the model

#drl.episodic_learn()

#%% Plotting the training loss

#drl.plot_curves()
# %%
datetimearray = DateTimeArray()
date_time_array = datetimearray.date_time_array_generate()
leos = TLE_Loader().load_leo_satellites()

leo_parameters = LEO_Parameters(ue_geo_position,date_time_array, leos, num_times,del_t)

sat_name, path_loss_matrix, elev_matrix, serv_time_matrix = leo_parameters.calculate_leo_parameters()
leo_parameters.plot_curves(elev_matrix, serv_time_matrix)
# %%
