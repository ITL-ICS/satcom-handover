# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:56:08 2023

@author: shada
"""
# Single-agent 기준: state_dim = 3 by 10(=best SAT), action_dim = 1 by 10
# 시뮬 논문도 ReLU 함수 사용
import torch
import torch.nn as nn

class MLP(nn.Module): 
    def __init__(self, state_dim, action_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 48)
        self.fc3 = nn.Linear(48, 24)
        self.fc4 = nn.Linear(24, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

#a = MLP()
#print(a.date_time_array_generate())
    