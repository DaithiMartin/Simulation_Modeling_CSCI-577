import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Deterministic Policy!) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=256):
        """
        Initialize parameters and build model.

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (int): Number of nodes in first hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed) if seed is not None else None

        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc_units)

        self.fc_total_head = nn.Linear(fc_units, fc_units)
        self.fc_total_water = nn.Linear(fc_units, 1)
        self.fc_total_land = nn.Linear(fc_units, 1)

        self.fc_crop_head = nn.Linear(fc_units, fc_units)
        self.fc_crop_water = nn.Linear(fc_units, action_size // 2)
        self.fc_crop_land = nn.Linear(fc_units, action_size // 2)

        # self.reset_parameters()

    # def reset_parameters(self):
    #     self.fc1.weight.data.uniform_(-3e-3, 3e-3)
    #     self.fc2.weight.data.uniform_(-3e-3, 3e-3)
    #
    #     self.fc_total_head.weight.data.uniform_(-3e-3, 3e-3)
    #     self.fc_total_water.weight.data.uniform_(-3e-3, 3e-3)
    #     self.fc_total_land.weight.data.uniform_(-3e-3, 3e-3)
    #
    #     self.fc_crop_head.weight.data.uniform_(-3e-3, 3e-3)
    #     self.fc_crop_water.weight.data.uniform_(-3e-3, 3e-3)
    #     self.fc_crop_land.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Actor (policy) network that maps states -> actions.

        Args:
            state vector (torch.tensor):
            [batch, available_water, available_land, crops_encoding, cost_encoding]
        """

        max_water = state.T[0].unsqueeze(-1)
        max_land = state.T[1].unsqueeze(-1)

        x_1 = F.relu(self.fc1(state))
        x2 = F.relu(self.fc2(x_1))

        # total allocated resource head
        # sigmoid gives us the proportion of total available water and land to be used
        # we call this utilized land and water
        x_total_head = F.relu(self.fc_total_head(x2))
        total_water_proportion = torch.sigmoid(self.fc_total_water(x_total_head)) * max_water
        total_land_proportion = torch.sigmoid(self.fc_total_land(x_total_head)) * max_land

        # distribution of resources over crops head
        # softmax over the two heads gives us the proportion of utilized land and water
        x_crop_head = F.relu(self.fc_crop_head(x2))
        crop_water_proportions = F.softmax(self.fc_crop_water(x_crop_head), dim=0)
        crop_land_proportions = F.softmax(self.fc_crop_land(x_crop_head), dim=0)

        water_actions = max_water * total_water_proportion * crop_water_proportions
        land_actions = max_land * total_land_proportion * crop_land_proportions

        return torch.cat((water_actions, land_actions), 0)


class Critic(nn.Module):
    """Critic (Action Value!) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=256, fc3_units=128):
        """Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            fc3_units (int): Number of nodes in the third hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed) if seed is not None else None
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)

    def forward(self, state, action):
        """
        Critic (value) network that maps (state, action) pairs -> Q-values.

        Args:
            state (Vector, torch.tensor): [available_water, CropWater_n, CropGrowth_n, CropPrices_n]
                                          Where n is the number of crops dependant of simulation
            action (Vector, torch.tensor): [WaterAmount_n] Where n is the number of crops.
        """
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)
