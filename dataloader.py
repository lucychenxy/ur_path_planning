import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, splprep, splev
from torch.utils.data import Dataset, DataLoader, random_split

def read_angles():
    """
    input: csv files
    return: start & end point (np.array); control points
    """
    pass


def parametrize_b_spline():
    """
    input: trajectories joint angles
    return: coefficients 11*6
    """
    pass


class UR5OptPathDataset(Dataset):
    def __init__(self, file_list):
        self.trajectories = {}
        # TODO: Modify Later
        for file_name in file_list:
            with open(file_name, 'r') as f:
                trj_id = ... # an integer number
                start_joints = ...
                end_joints = ...
                control_p = ...
            
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

        # some code to convert the data to tensor
        ends = ... # concatenate the start point and end point as input

        return ends, control_p
    

