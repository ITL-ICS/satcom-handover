#은서 >_<

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:04:01 2023

@author: shada
"""
# Importing necessary modules


from drl_frameworks import DRL



#%% Define the DRL Framework

#3000정도가 좋을 듯
num_episodes = 500

# ue_geo_position = GeoCordinates(-62, 50, 0)
# del_t = 5 # simulation time in minutes
# num_times = 30 # number of simulation samples

drl = DRL(num_episodes)


#%% Training the model

drl.episodic_learn()

#%% Plotting the training loss

drl.plot_curves()
# %%
LEO_Parameters.plot_curves()
