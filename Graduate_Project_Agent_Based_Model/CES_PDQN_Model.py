import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.fc_water_head_1 = nn.Linear(fc_units, fc_units)
        self.fc_water_head_2 = nn.Linear(fc_units, 101)

        self.fc_land_head_1 = nn.Linear(fc_units, fc_units)
        self.fc_land_head_2 = nn.Linear(fc_units, 101)

        self.dropout_1 = nn.Dropout(p=0.25)
        self.dropout_2 = nn.Dropout(p=0.25)
        self.dropout_3 = nn.Dropout(p=0.25)
        self.dropout_4 = nn.Dropout(p=0.25)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(-3e-3, 3e-3)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

        self.fc_water_head_1.weight.data.uniform_(-3e-3, 3e-3)
        self.fc_water_head_2.weight.data.uniform_(-3e-3, 3e-3)

        self.fc_land_head_1.weight.data.uniform_(-3e-3, 3e-3)
        self.fc_land_head_2.weight.data.uniform_(-3e-3, 3e-3)

        return None

    def forward(self, state):
        """
        Policy network that maps states -> action values! not actions

        Args:
            state vector (torch.tensor):
            [batch, available_water, available_land, crops_encoding, cost_encoding]
        """

        x1 = self.dropout_1(F.relu(self.fc1(state)))
        x2 = self.dropout_2(F.relu(self.fc2(x1)))

        x_water = self.dropout_3(F.relu(self.fc_water_head_1(x2)))
        x_land = self.dropout_4(F.relu(self.fc_land_head_1(x2)))

        water_values = self.fc_water_head_2(x_water)
        land_values = self.fc_land_head_2(x_land)

        return torch.stack((water_values, land_values))

