"""MLP with convolutional gating (cgMLP) definition.

References:
    https://openreview.net/forum?id=RA-zVvZLYIy
    https://arxiv.org/abs/2105.08050

"""

import torch
import os 
from espnet2.legacy.nets.pytorch_backend.nets_utils import get_activation
from espnet2.legacy.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet2.edgeSim.LinearLayerSim import LinearSim
import numpy as np
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
print(f"CGMLP SOURCE CODE DEVICE: {DEVICE}")

SIMULATE = os.getenv("APPLY_SIM", "False") # default to False if the environment variable is not set

class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    """Convolutional Spatial Gating Unit (CSGU)."""

    def __init__(
        self,
        size: int,
        kernel_size: int,
        dropout_rate: float,
        use_linear_after_conv: bool,
        gate_activation: str,
    ):
        super().__init__()

        n_channels = size // 2  # split input channels
        self.norm = LayerNorm(n_channels)
        self.conv = torch.nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size,
            1,
            (kernel_size - 1) // 2,
            groups=n_channels,
        )
        if use_linear_after_conv:
            self.linear = torch.nn.Linear(n_channels, n_channels)
        else:
            self.linear = None

        if gate_activation == "identity":
            self.act = torch.nn.Identity()
        else:
            self.act = get_activation(gate_activation)

        self.dropout = torch.nn.Dropout(dropout_rate)

    def espnet_initialization_fn(self):
        torch.nn.init.normal_(self.conv.weight, std=1e-6)
        torch.nn.init.ones_(self.conv.bias)
        if self.linear is not None:
            torch.nn.init.normal_(self.linear.weight, std=1e-6)
            torch.nn.init.ones_(self.linear.bias)

    def forward(self, x, gate_add=None):
        """Forward method

        Args:
            x (torch.Tensor): (N, T, D)
            gate_add (torch.Tensor): (N, T, D/2)

        Returns:
            out (torch.Tensor): (N, T, D/2)
        """

        x_r, x_g = x.chunk(2, dim=-1)

        x_g = self.norm(x_g)  # (N, T, D/2)
        x_g = self.conv(x_g.transpose(1, 2)).transpose(1, 2)  # (N, T, D/2)
        if self.linear is not None:
            
            # @@@@@@@@@@@@@@@@@@ EDGE SIM @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#            print("SIMULATING LINEAR LAYER IN CSGU...")
            with torch.no_grad():
                weight = self.linear.weight.data.to(DEVICE)
                bias = self.linear.bias.data.to(DEVICE)
                linear_sim_layer = LinearSim(Weight=weight, Bias=bias, Error_Dist=None, show_batch_processing=True)
                x_sim_input = x_g.to(DEVICE) # use the old x_g as the sim input
                x_sim = linear_sim_layer(x_sim_input).to(DEVICE)
            # |
            # V
            
            x_g = self.linear(x_g).to(DEVICE) #         --> LOCAL LINEAR LAYER!
            
            # Ʌ
            # |
            max_diff = torch.max(torch.abs(x_g - x_sim)).item()
#            print(f"MAX DIFF: {max_diff}")
            if SIMULATE == "False":
                assert torch.allclose(x_g.detach().cpu(), x_sim.detach().cpu(), atol=1e-3), f"Output mismatch between original linear layer and simulated linear layer in CSGU!"
            x_g = x_sim # use the sim output as the new x_g to ensure that the sim layer is actually running during inference
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            
   

        if gate_add is not None:
            x_g = x_g + gate_add

        x_g = self.act(x_g)
        out = x_r * x_g  # (N, T, D/2)
        out = self.dropout(out)
        return out


class ConvolutionalGatingMLP(torch.nn.Module):
    """Convolutional Gating MLP (cgMLP)."""

    def __init__(
        self,
        size: int,
        linear_units: int,
        kernel_size: int,
        dropout_rate: float,
        use_linear_after_conv: bool,
        gate_activation: str,
    ):
        super().__init__()

        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(size, linear_units), torch.nn.GELU()
        )
        self.csgu = ConvolutionalSpatialGatingUnit(
            size=linear_units,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            use_linear_after_conv=use_linear_after_conv,
            gate_activation=gate_activation,
        )
        self.channel_proj2 = torch.nn.Linear(linear_units // 2, size)

    def forward(self, x, mask):
        if isinstance(x, tuple):
            xs_pad, pos_emb = x
        else:
            xs_pad, pos_emb = x, None

        # @@@@@@@@@@@@@@@@@@ EDGE SIM @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#        print("SIMULATING FIRST LINEAR LAYER IN CGMLP...")
        with torch.no_grad():
            weight = self.channel_proj1[0].weight.data.to(DEVICE)
            bias = self.channel_proj1[0].bias.data.to(DEVICE)
            linear_sim_layer = LinearSim(Weight=weight, Bias=bias, Error_Dist=None, show_batch_processing=True)
            x_sim_input = xs_pad.to(DEVICE) # use the old xs_pad as the sim input
            x_sim = self.channel_proj1[1](linear_sim_layer(x_sim_input).to(DEVICE)).to(DEVICE)  #don't forget to apply the gelu after the linear layer!
        # |
        # V
        xs_pad = self.channel_proj1(xs_pad).to(DEVICE)  # size -> linear_units         --> LOCAL LINEAR LAYER!
        
        # Ʌ
        # |
        max_diff = torch.max(torch.abs(xs_pad - x_sim)).item()
#        print(f"MAX DIFF: {max_diff}")
        if SIMULATE == "False":
            assert torch.allclose(xs_pad.detach().cpu(), x_sim.detach().cpu(), atol=1e-3), f"Output mismatch between original linear layer and simulated linear layer in CGMLP!"
        xs_pad = x_sim # use the sim output as the new xs_pad to ensure that the sim layer is actually running during inference
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        
        xs_pad = self.csgu(xs_pad)  # ConvolutionalSpatialGatingUnit # linear_units -> linear_units/2    
        
        
        # @@@@@@@@@@@@@@@@@@ EDGE SIM @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#        print("SIMULATING SECOND LINEAR LAYER IN CGMLP...")
        with torch.no_grad():
            weight = self.channel_proj2.weight.data.to(DEVICE)
            bias = self.channel_proj2.bias.data.to(DEVICE)
            linear_sim_layer = LinearSim(Weight=weight, Bias=bias, Error_Dist=None, show_batch_processing=True)
            x_sim_input = xs_pad.to(DEVICE) # use the old xs_pad as the sim input
            x_sim = linear_sim_layer(x_sim_input).to(DEVICE)
        # |
        # V
        
        xs_pad = self.channel_proj2(xs_pad).to(DEVICE)  # linear_units/2 -> size       --> LOCAL LINEAR LAYER!
        
        # Ʌ
        # |
        max_diff = torch.max(torch.abs(xs_pad - x_sim)).item()
#        print(f"MAX DIFF: {max_diff}")
        if SIMULATE == "False":
            assert torch.allclose(xs_pad.detach().cpu(), x_sim.detach().cpu(), atol=1e-3), f"Output mismatch between original linear layer and simulated linear layer in CGMLP!"
        xs_pad = x_sim # use the sim output as the new xs_pad to ensure that the sim layer is actually running during inference
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        if pos_emb is not None:
            out = (xs_pad, pos_emb)
        else:
            out = xs_pad
        return out
