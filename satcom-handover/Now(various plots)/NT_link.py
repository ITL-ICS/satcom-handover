# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:56:08 2023

@author: shada
"""
# fc = 20GHz
import numpy as np
import math

class NT_Link:
    def __init__(self, sat_geo_position, obs_geo_position):
        self.sat_geo_position = sat_geo_position
        self.obs_geo_position = obs_geo_position
        
    def calculate_distance(self): # 두 점간 거리 계산
        
        obs_ecef = self.obs_geo_position.geo2ecef ()
        sat_ecef = self.sat_geo_position.geo2ecef ()

        return math.dist(sat_ecef,obs_ecef)

    def calculate_path_loss(self,fc = 20): # path loss 계산
        d = self.calculate_distance()
        Lp = 32.45 + 20 * np.log10(d) + 20 * np.log10(fc) # Free-Space Path Loss GHz단위 및 km
        shadowing = np.random.normal(0, 1)
        
        fading_loss = 0
        fading_amp = np.random.rayleigh(scale=np.sqrt(2) * np.power(10, 1 / 20))
        fading_loss = 20 * np.log10(np.abs(fading_amp)) # rayleigh 분포를 따르는 난수
        #print ("Path loss", Lp, " Shadowing", shadowing, " fading_loss", fading_loss)
        
        
        return Lp + shadowing + fading_loss

    def calculate_elevation_angle (self): # elevation angle 계산
        
        distance = self.calculate_distance()
        elevation_angle = math.asin(round((self.sat_geo_position.alt - self.obs_geo_position.alt) / distance,4))
        
        return math.degrees(elevation_angle)

    