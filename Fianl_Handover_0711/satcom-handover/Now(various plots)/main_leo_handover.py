# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:04:01 2023

@author: shada
"""
# Importing necessary modules


#%% from drl_frameworks import DRL
import random
import math

from coordinates import GeoCordinates
from leo_parameters import LEO_Parameters
from date_time_array import DateTimeArray
from tle_loader import TLE_Loader


# Define the DRL Framework

num_episodes = 500

# 랜덤하게 뿌린 100명 유저의 위치 좌표 = coordinates (list type)
def generate_random_coordinates(center_lat = 40, center_lon = 116, radius_km = 110, num_points = 10):
    coordinates = []
    for _ in range(num_points):
        # Generate a random distance and angle
        distance = random.uniform(0, radius_km)
        angle = random.uniform(0, 2 * math.pi)
        
        # Convert distance from kilometers to degrees
        distance_deg = distance / 111.32  # 1 degree ~ 111.32 km

        # Calculate the new latitude and longitude
        delta_lat = distance_deg * math.cos(angle)
        delta_lon = distance_deg * math.sin(angle) / math.cos(math.radians(center_lat))

        new_lat = center_lat + delta_lat
        new_lon = center_lon + delta_lon
        
        # Altitude between 0 and 1000 meters
        alt = random.uniform(0, 1000)
        
        coordinates.append(GeoCordinates(new_lat, new_lon, alt))

    return coordinates

coordinates = generate_random_coordinates()

print(type(coordinates))
for coord in coordinates:
    ecef_coord = coord.geo2ecef()
    print(ecef_coord)

print(type(ecef_coord)) # coordinates의 tuple type


# Center coordinates
center_lat = 40.0
center_lon = 116.0
radius_km = 110  # 220 km diameter

ue_geo_position = coordinates

#ue_geo_position = GeoCordinates(-62, 50, 0)
del_t = 5 # simulation time in minutes
num_times = 30 # number of simulation samples 
# 10초 = 한 time slot

#drl = DRL(num_episodes)


# Training the model

#drl.episodic_learn()

# Plotting the training loss

#drl.plot_curves()
# 
datetimearray = DateTimeArray()
date_time_array = datetimearray.date_time_array_generate()
leos = TLE_Loader().load_leo_satellites()

leo_parameters = LEO_Parameters(ue_geo_position,date_time_array, leos, num_times,del_t)

sat_name, path_loss_matrix, elev_matrix, serv_time_matrix = leo_parameters.calculate_leo_parameters()
leo_parameters.plot_curves(elev_matrix, serv_time_matrix)
# %%
