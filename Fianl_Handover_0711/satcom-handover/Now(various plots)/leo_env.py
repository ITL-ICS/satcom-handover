# -*- coding: utf-8 -*-
"""
Created on Fri May  5 23:38:22 2023

@author: shada
"""

from date_time_array import DateTimeArray
from tle_loader import TLE_Loader
from leo_parameters import LEO_Parameters
from coordinates import GeoCordinates
import random
import numpy as np
import torch
import math

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
    
print(generate_random_coordinates())
coordinates = generate_random_coordinates() # coordinates: <class 'list'>
#coordinates: 각 원소들은 문자열 type으로 되어있음 ex.<coordinates.GeoCordinates object at 0x000002946DD25FD0>


each_element_tuple = [tuple(item) for item in coordinates]

print("each_element_tuple뭘까?",each_element_tuple)
#instances = [GeoCordinates(lat, lon, alt) for lat, lon, alt in coordinates]
#ecef_coordinates = [instance.geo2ecef() for instance in instances]
print("ecef_coordinates타입은?",type(each_element_tuple))


center_lat = 40.0
center_lon = 116.0
radius_km = 110  # 220 km diameter


class LeoEnv: # single-user 기준 가장 best 인 10개의 위성 추출: num_satellites = 10
    def  __init__(self, num_satellites = 10, num_features = 3, ue_geo_position = each_element_tuple, del_t=5, num_times=30):
        
        self.action_space = num_satellites
        self.observation_space = num_satellites * num_features

        ue_geo_position
        self.ue_geo_position = ue_geo_position
        self.del_t = del_t # simulation time in minutes
        self.num_times = num_times # number of simulation samples
        
        self.steps_before_termination = 0
        
        self.state = np.empty((0, 1))
        
        self.previous_action = torch.zeros(self.action_space)
        
        # define the time window for tracking

        self.date_time_array = DateTimeArray(self.del_t, self.num_times).date_time_array_generate()

        self.leos = TLE_Loader().load_leo_satellites()
        
        self.satellite_names, self.path_loss_matrix, self.elev_matrix, self.serv_time_matrix = LEO_Parameters(self.ue_geo_position, self.date_time_array, self.leos, self.num_times, self.del_t).calculate_leo_parameters()

        def compute_state(self, index):
        # index is the current timestamp

        # Creating an observation space of 10 x 3
            self.state = np.empty((0, int(self.observation_space / self.action_space)))

            # Accumulate the path loss for all satellites
            path_loss_timestamp = self.path_loss_matrix[:, index]

            # Find out the best set of candidate satellites based on coverage
            # coverage 기반으로 best 인 위성 set을 고름: path-loss 값 200 기준
            r = np.where(path_loss_timestamp < 200.0)[0]

            # Handle the case where action space is greater than the number of eligible satellites
            if self.action_space - r.shape[0] > 0:
                r_prime = np.random.choice(np.where(path_loss_timestamp == 200.0)[0], self.action_space - r.shape[0], replace=False)
            else:
                r_prime = np.array([])

            # Combine eligible and complementary satellites
            l = np.concatenate([r, r_prime])
            l = np.sort(l)

            # Ensure l is an integer array
            l = l.astype(int)

            # List the best candidate satellites
            self.candidate_satellites = np.array(self.satellite_names)[l]
            print(self.candidate_satellites)
            # Gather the path loss for candidate satellites
            path_loss = path_loss_timestamp[l]

            # Calculate service time and quality for candidate satellites
            avg_elev = np.zeros(self.action_space)
            serv_time = np.zeros(self.action_space)

            for i in range(self.action_space):
                serv_time_indices = np.nonzero(self.serv_time_matrix[l[i], :])[0]
                if len(serv_time_indices) > 0:
                    serv_time_first_index = serv_time_indices[0]
                    serv_time_last_index = serv_time_indices[-1]
                    
                    if index >= serv_time_first_index and self.serv_time_matrix[l[i], index] > 0.0:
                        avg_elev[i] = np.mean(self.elev_matrix[l[i], index:serv_time_last_index+1]) 
                        serv_time[i] = np.sum(self.serv_time_matrix[l[i], index:serv_time_last_index+1]) 
                
            self.state = np.append(self.state, avg_elev)
            self.state = np.append(self.state, serv_time)
            self.state = np.append(self.state, path_loss)

        return self.state
        
    def compute_reward (self, action):
        avg_el = self.state[0:self.action_space]
        t_serv = self.state[self.action_space:2*self.action_space]
        path_loss = self.state[2*self.action_space:3*self.action_space]
        
        # print("Average elevation angle ", avg_el, " Service time ", t_serv, " path loss", path_loss)
        # print("Service time ", np.dot(action, t_serv), " path loss", np.dot(action, path_loss), " for action", action)
        if np.dot(t_serv, action) < 1.0 or np.dot(path_loss, action) > 185.0:
            reward = -25.0
        elif torch.all(torch.eq(action, self.previous_action)):
            reward = 25.0
        else:
            reward =  np.dot(action, 10 *(t_serv / self.del_t) + 10 * (avg_el / np.max (avg_el)) - 10 *  (path_loss / np.max(path_loss)))
            
        return reward 
    
    def step(self, action):
        
        # action is a N x 1 vector and state is a N x 3 matrix, 
        # rows should correspond to the satellites
        
        self.state = self.compute_state(self.steps_before_termination)
        
        # print("Computed state ", self.state)
        
        #rewards

        reward = self.compute_reward(action)
        
        # print("Computed reward ", reward)
          
        self.previous_action = action 
        self.steps_before_termination += 1
        
        if (self.steps_before_termination == self.num_times):
            terminated = True
        else:
            terminated = False
            
        return self.state, reward, terminated
    
    def reset(self):
        self.previous_action = torch.zeros(self.action_space)
        self.previous_action[np.random.randint(self.action_space)] = 1.0
        self.state = self.compute_state(0)
        self.steps_before_termination = 1 # 아주 중요!!!!!!!!!
        
        return self.state


    
    