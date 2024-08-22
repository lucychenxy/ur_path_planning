import numpy as np
import json
import os
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, splprep, splev
from torch.utils.data import Dataset, DataLoader

def read_angles():
    """
    input: csv files
    return: start & end point (np.array); control points
    """
    pass


def parametrize_b_spline(trajectory:np.ndarray, k=3):
    """
    input: trajectories joint angles
    return: coefficients 11*6
    """
    u_input = np.linspace(0,1,trajectory.shape[1])
    tck, u = splprep(trajectory, s=0, k=3, u=u_input) 
    coef = tck[1] # a list of ndarray

    return coef


class UR5OptPathDataset(Dataset):
    def __init__(self, data_folder_path):
        file_list = []
        for file_name in os.listdir(data_folder_path):
            # Check if the path is a file (and not a directory)
            if os.path.isfile(os.path.join(data_folder_path, file_name)):
                file_list.append(file_name)

        self.trajectories = {}
        for i, file_name in enumerate(file_list):
            with open(Path(data_folder_path/file_name), 'r') as f:
                trj_data = json.load(f)
                trj_id = i # an integer number
                start_joints = trj_data['start-joints']
                end_joints = trj_data['end-joints']
                control_p = trj_data['control-p']
            
            self.trajectories.update(
                {trj_id: 
                    {
                     'start': start_joints,
                     'end': end_joints,
                     'control_p': control_p
                    }})

    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        start = self.trajectories[idx]['start']
        end = self.trajectories[idx]['end']
        control_p = self.trajectories[idx]['control_p']

        ends = start+end
        ends = torch.tensor(ends)
        coef = np.concatenate(control_p)
        coef = torch.tensor(coef)

        return ends, coef
    

def get_dataloaders(dataset, batch_size, shuffle=True):
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    
    return dataloader

    

