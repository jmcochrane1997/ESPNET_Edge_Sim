#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positionwise feed forward layer definition."""

import torch


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
        w1 = self.w_1(x)  #         --> LOCAL LINEAR LAYER!
        # @@@@@@@@@@@@@@@@@@ EDGE SIM @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        d = {"x_dim": x.dim(), "w_dim": self.w_1.weight.ndimension()}
        print(d)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        
        
        a1 = self.activation(w1)
        d1 = self.dropout(a1)
        w2 = self.w_2(d1) #         --> LOCAL LINEAR LAYER!
        # @@@@@@@@@@@@@@@@@@ EDGE SIM @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        d = {"x_dim": d1.dim(), "w_dim": self.w_2.weight.ndimension()}
        print(d)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        
        
        return w2   