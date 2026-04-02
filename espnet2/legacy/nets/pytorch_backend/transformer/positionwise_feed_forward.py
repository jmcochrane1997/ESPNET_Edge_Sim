#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positionwise feed forward layer definition."""

import torch
import os
from espnet2.edgeSim.LinearLayerSim import LinearSim
import numpy as np
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
print(f"POSITIONWISE FEED FORWARD SOURCE CODE DEVICE: {DEVICE}")
SIMULATE = os.getenv("APPLY_SIM", "False") # default to False if the environment variable is not set
THRESH = float(os.getenv("UNIT_TEST_THRESHOLD", "0.001")) # default to 0.0001 if not set


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        w1 = self.w_1(x).to(DEVICE)  #         --> LOCAL LINEAR LAYER!
        # @@@@@@@@@@@@@@@@@@ EDGE SIM @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#        print("SIMULATING FIRST LINEAR LAYER IN POSITIONWISE FEED FORWARD...")
        with torch.no_grad():
            weight = self.w_1.weight.data.to(DEVICE)
            bias = self.w_1.bias.data.to(DEVICE)
            linear_sim_layer = LinearSim(Weight=weight, Bias=bias, Error_Dist=None, show_batch_processing=True)
            x_sim_input = x.to(DEVICE) # use the old x as the sim input
            x_sim = linear_sim_layer(x_sim_input).to(DEVICE)
        #print("sim output:"+ str(x_sim))
        #print("gt output:"+ str(w1))
        max_diff = torch.max(torch.abs(w1 - x_sim)).item()
#        print(f"MAX DIFF: {max_diff}")
        if SIMULATE == "False":
            assert torch.allclose(w1.detach().cpu(), x_sim.detach().cpu(), atol=THRESH), f"Output mismatch between original linear layer and simulated linear layer in PositionwiseFeedForward!"
        w1 = x_sim # use the sim output as the new w1 to ensure that the sim layer is actually running during inference
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        
        
        a1 = self.activation(w1)
        d1 = self.dropout(a1)
        w2 = self.w_2(d1).to(DEVICE) #         --> LOCAL LINEAR LAYER!
        # @@@@@@@@@@@@@@@@@@ EDGE SIM @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#        print("SIMULATING SECOND LINEAR LAYER IN POSITIONWISE FEED FORWARD...")
        with torch.no_grad():
            weight = self.w_2.weight.data.to(DEVICE)
            bias = self.w_2.bias.data.to(DEVICE)
            linear_sim_layer = LinearSim(Weight=weight, Bias=bias, Error_Dist=None, show_batch_processing=True)
            x_sim_input = d1.to(DEVICE) # use the old d1 as the sim input
            x_sim = linear_sim_layer(x_sim_input).to(DEVICE)
        #print("sim output:"+ str(x_sim))
        #print("gt output:"+ str(w2))
        max_diff = torch.max(torch.abs(w2 - x_sim)).item()
#        print(f"MAX DIFF: {max_diff}")
        if SIMULATE == "False":
            assert torch.allclose(w2.detach().cpu(), x_sim.detach().cpu(), atol=THRESH), f"Output mismatch between original linear layer and simulated linear layer in PositionwiseFeedForward!"
        w2 = x_sim # use the sim output as the new w2 to ensure that the sim layer is actually running during inference
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        
        return w2   