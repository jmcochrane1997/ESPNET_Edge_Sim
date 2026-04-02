import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
torch.manual_seed(42)
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

# load the simulation boolean as an environment variable
SIMULATE = os.getenv("APPLY_SIM", "False")
print(f"APPLY SIMULATION: {SIMULATE}")

# load the standard deviation as an environment variable
std_dev_env = float(os.getenv("STD_DEV", "0.01")) # default to 0.01 if not set
print(f"USING STD DEV: {std_dev_env} FOR ERROR SAMPLING IN LINEAR SIMULATION")
# create the error dist.
mean = 0
std_dev = std_dev_env  #*** NOISE LEVEL FROM ENV VARIABLE !! ***
#normal_dist = torch.distributions.Normal(loc=mean, scale=std_dev)
THRESH = float(os.getenv("UNIT_TEST_THRESHOLD", "0.001")) # default to 0.0001 if not set
print(f"UNIT TEST THRESHOLD: {THRESH}")




class LinearSim(nn.Module):
    def __init__(self, Weight, Bias, Error_Dist, show_batch_processing=False):
        super().__init__()
        self.Weight = Weight
        self.Bias = Bias
        self.Error_Dist = Error_Dist
        if self.Bias is None:
          self.Bias = 0 #torch.zeros(self.Weight.shape[0])
        self.show_batch_processing = show_batch_processing


    def quick_split_W3D(self, matrix):
      '''
      This function splits a weight matrix, W, into two parts: W_+ and W_-, such that W=(W_+)-(W_-). This is done to map each W(_+/-)
      to the same range as X: (0,1)

      INPUT: matrix, a tensor from the state dictionary of the trained model
      '''
      # form the negative of the input matrix
      negative_matrix=-1*matrix
      # use RELU to find the positive matrix W_+
      W_pos=nn.ReLU(inplace=True)(matrix.detach().clone())
      # now, use RELU of the **negative matrix** to find the negative factored matrix W_-
      W_neg=nn.ReLU(inplace=True)(negative_matrix.detach().clone())
      assert ((W_pos - W_neg).cpu().numpy() == matrix.cpu().numpy()).all()
      return W_pos, W_neg


    def ErrorSample(self, num_channels, num_rows, num_cols):
      '''
      Generate (gaussian) error samples
      '''
      samples = torch.randn((num_channels, num_rows, num_cols), device=DEVICE) * std_dev + mean 
      return samples
      #assert self.Error_Dist.dim == 3, "error: expected a three-dimensional error distribution"


    def Matmul_2D(self, X, W, diff_dist=None):
      '''
      Re-factors 2d Matrix multiplication using a sum over Hadamard Products.
      X: Tensor of shape (batch_size, in_features) = (m,n)
      W: Tensor of shape (out_features, in_features) = (k,n)
      diff_dist: Tensor of shape (batch_size, in_features, out_features)
      '''

      assert X.dim()==2 and W.dim()==2
      m, n = X.shape   # m = in_features
      k, n = W.shape   # k = out_features
#      if diff_dist is not None:
#        assert diff_dist.shape == (m, n, k)

      # transpose the weight matrix from (k,n) --> (n,k)
      W = W.T # shape = (n, k)

      # compute the channel-wise Hadamard Products
      Hs = [] #
      for row_index in range(m):
        x_row = X[row_index].view(-1, 1) # reshape each row to be a col vec
        H = x_row * W  #                   broadcast each row of x over W
        Hs.append(H)

      X_times_W = torch.stack(Hs, dim=0) # stack the Hadamard Products along the channel dimension
      assert X_times_W.shape == (m, n, k)
      
      if SIMULATE=="True":
        X_times_W_Plus_Delta = X_times_W + self.ErrorSample(num_channels=m, num_rows=n, num_cols=k).to(X_times_W.device) #torch.zeros(m, n, k, device=X_times_W.device) 
        x_times_W_Plus_Delta_Summed = torch.sum(X_times_W_Plus_Delta, dim=1).view(m,k)   # sum over the row dim and reshape to match the expected output shape
        assert x_times_W_Plus_Delta_Summed.shape == (m, k)
        return x_times_W_Plus_Delta_Summed
      
      elif SIMULATE=="False":
        X_times_W_Plus_Delta = X_times_W + torch.zeros(m, n, k, device=X_times_W.device) 
        x_times_W_Plus_Delta_Summed = torch.sum(X_times_W_Plus_Delta, dim=1).view(m,k)   # sum over the row dim and reshape to match the expected output shape
        assert x_times_W_Plus_Delta_Summed.shape == (m, k)
        return x_times_W_Plus_Delta_Summed
      
      else:
        raise Exception(f"Invalid value for SIMULATE: {SIMULATE}. Expected 'True' or 'False' as a string.")
      
      
    def Matmul(self, X, W, diff_dist=None):
      '''
      Re-implements matrix/tensor multiplication over high-dimensional inputs
      X: Tensor - can be 2d, 3d, or 4d
      W: Tensor - can be 2d, 3d, or 4
      diff_dist: Tensor
      '''
      x_dim = X.dim()
      w_dim = W.dim()

      # (1) Map X, W to 4d tensors  (this standardizes the tensor product over all possible cases listed below)

      # CASE 1
      if X.dim() == 2 and W.dim() == 2:
        X = X.unsqueeze(0).unsqueeze(0)  #  map (m,n) --> (1, 1, m, n)
        W = W.unsqueeze(0).unsqueeze(0)  #  map (k,n) --> (1, 1, n, k)


      # CASE 2
      elif X.dim() == 3 and W.dim() == 3:
        assert X.shape[0] == W.shape[0] # make sure that c_x = c_w
        X = X.unsqueeze(0) #             map (c_x,m,n) --> (1, c_x, m, n)
        W = W.unsqueeze(0) #             map (c_w,k,n) --> (1, c_w, k, n)

      # CASE 3
      elif X.dim() == 4 and W.dim() == 4:
        assert X.shape[1] == W.shape[1] # make sure that c_x = c_w
        X = X
        W = W

      # CASE 4
      elif X.dim() == 2 and W.dim() == 3:
        c_w = W.shape[0]
        X_3d = X.unsqueeze(0).repeat(c_w, 1, 1) # map (m, n) --> (1, m, n) --> (c_x, m, n) where c_x = c_w
        X = X_3d.unsqueeze(0) #                   map (c_x, m, n) --> (1, c_x, m, n)
        W = W.unsqueeze(0) #                     map (c_w, k, n) --> (1, c_w, k, n)

      # CASE 5
      elif X.dim() == 3 and W.dim() == 2:
        c_x = X.shape[0]
        W_3d = W.unsqueeze(0).repeat(c_x, 1, 1) # map (k, n) --> (1, k, n) --> (c_w, k, n) where c_w = c_x
        W = W_3d.unsqueeze(0) #                   map (c_w, k, n) --> (1, c_w, k, n)
        X = X.unsqueeze(0) #                     map (c_x, m, n) --> (1, c_x, m, n)

      # CASE 6
      elif X.dim() == 3 and W.dim() == 4:
        assert X.shape[0] == W.shape[1] # make sure that c_x = c_w
        b_w = W.shape[0]
        X = X.unsqueeze(0).repeat(b_w, 1, 1, 1) # map (c_x, m, n) --> (1, c_x, m, n) --> (b_x, c_x, m, n) where b_x = b_w
        W = W #                                                                  shape = (b_w, c_w, m, n)


      # CASE 7
      elif X.dim() == 4 and W.dim() == 3:
        assert X.shape[1] == W.shape[0] # make sure that c_x = c_w
        b_x = X.shape[0]
        W = W.unsqueeze(0).repeat(b_x, 1, 1, 1) # map (c_w, k, n) --> (1, c_w, k, n) --> (b_w, c_w, k, n) where b_w = b_x
        X = X #                                                                  shape = (b_x, c_x, m ,n)

      # CASE 8
      elif X.dim() == 4 and W.dim() == 2:
        b_x = X.shape[0]
        c_x = X.shape[1]
        W_3d = W.unsqueeze(0).repeat(c_x, 1, 1) #     map (k, n) --> (1, k, n) --> (c_w, k, n) where c_w = c_x
        W = W_3d.unsqueeze(0).repeat(b_x, 1, 1, 1) # map (c_w, k, n) --> (1, c_w, k, n) --> (b_w, c_w, k, n)
        X = X #                                                                     shape = (b_x, c_x, m, n)

      # CASE 9
      elif X.dim() == 2 and W.dim() == 4:
        b_w = W.shape[0]
        c_w = W.shape[1]
        X_3d = X.unsqueeze(0).repeat(c_w, 1, 1) # map (m, n) --> (1, m, n) --> (c_x, m, n) wher c_x = c_w
        X = X_3d.unsqueeze(0).repeat(b_w, 1, 1, 1) # map (c_x, m, n) --> (1, c_x, m, n) --> (b_x, c_x, m, n) where b_x = b_w
        W = W #                                                                     shape = (b_w, c_w, k, n)

      else:
        raise Exception(f"Invalid input dimensions: dim(X) = {X.dim()} and dim(W) = {W.dim()}")

      assert X.dim()==4 and W.dim()==4
      assert X.shape[0] == W.shape[0], "misalligned batch dimension between tensor X and tensor W"
      assert X.shape[1] == W.shape[1], "misalligned channel dimension between tensor X and tensor W"

      # (2) implement tensor multiplication

      # iterate over each batch in the 4d tensor (ex. if a tensor has shape (b, c, m, n) we are iterating over b in this case)
      xw_batch = []
      batch_iterable = tqdm(range(X.shape[0]), desc="Processing Batch", unit="item") if self.show_batch_processing and X.shape[0] > 1 else range(X.shape[0])
      channel_iterable = tqdm(range(X.shape[1]), desc="Processing Channel", unit="item") if self.show_batch_processing and X.shape[1] > 1 else range(X.shape[1])
      for b in batch_iterable:
        # iterate over all channels (c) in batch b
        xw_channels = []
        for c in channel_iterable:
          X_2d = X[b, c, :, :]  # --> index the 2d matrix inside a specific channel of a specific batch
          W_2d = W[b, c, :, :] # --> index the 2d matrix inside a specific channel of a specific batch
          # 2d matrix product
          xw_channels.append(self.Matmul_2D(X_2d, W_2d, diff_dist))

        # stack the 2d matrix products in the channel dimension --> 3d tensor in batch b
        xw_channels_tensor = torch.stack(xw_channels, dim=0)

        # append the 3d stacked tensor to the xw_batch list
        xw_batch.append(xw_channels_tensor)

      # stack the 3d stacked tensors in the batch dimension
      xw_batch_tensor = torch.stack(xw_batch, dim=0)

      # (3) return the final output tensor product based on the values of x_dim and w_dim
      if x_dim == 2 and w_dim == 2:
        assert xw_batch_tensor.shape[0] == 1 and xw_batch_tensor.shape[1] == 1
        return xw_batch_tensor.squeeze(0).squeeze(0)

      elif x_dim == 3 and w_dim == 3:
        assert xw_batch_tensor.shape[0] == 1
        return xw_batch_tensor.squeeze(0)

      elif x_dim == 4 and w_dim == 4:
        return xw_batch_tensor

      elif x_dim == 2 and w_dim == 3:
        assert xw_batch_tensor.shape[0] == 1
        return xw_batch_tensor.squeeze(0)

      elif x_dim == 3 and w_dim == 2:
        assert xw_batch_tensor.shape[0] == 1
        return xw_batch_tensor.squeeze(0)

      elif x_dim == 4 and w_dim == 3:
        return xw_batch_tensor

      elif x_dim == 2 and w_dim == 4:
        return xw_batch_tensor

      elif x_dim == 3 and w_dim == 4:
        return xw_batch_tensor

      elif x_dim == 4 and w_dim == 2:
        return xw_batch_tensor


    def forward(self, X):
        '''
        X: Tensor of shape (batch_size, in_features)
        Output: Tensor of shape (batch_size, out_features)
        Computes the Optical Projection of the Linear Layer. This method includes the normalization, weight split, and post-normalization mappings.
        '''
        # (1) compute the normalization factors for the input and weight matrices
        n_x = torch.max(X)
        w_min = torch.min(self.Weight)
        w_max = torch.max(self.Weight)
        n_w = torch.max(torch.abs(w_min), torch.abs(w_max))
        # (2) perform the normalization
        X_norm = X / n_x
        W_norm = self.Weight / n_w
        # (3) split the normalized weight matrix into positive and negative parts
        W_pos, W_neg = self.quick_split_W3D(W_norm)
        # (4) Compute the first part of the simulated optical matrix product
        Prod_plus = self.Matmul(X_norm, W_pos, self.Error_Dist)
        # (5) Compute the second part of the simulated optical matrix product
        Prod_minus = self.Matmul(X_norm, W_neg, self.Error_Dist)
        # (6) Combine the two parts to get the final simulated optical product
        Post_Processed = n_x * n_w * (Prod_plus - Prod_minus)
        # (7) Add the bias term
        Linear_output = Post_Processed + self.Bias
        return Linear_output


class ScaledDotProdAttention(nn.Module):
    def __init__(self, Error_Dist):
        super().__init__()
        self.Error_Dist = Error_Dist

    def forward(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
      '''
      Implements a version of scaled dot product attention that supports edge simulation.
      Adapted from source code: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
      '''
      L, S = query.size(-2), key.size(-2)
      scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
      attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
      if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

      if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

      if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

      # ///////////////////////////////////////// BEGIN MAIN INFERENCE ////////////////////////////////////////////////
      # this is just a torch.matmul (set weight=key)
      matmul_layer = LinearSim(Weight=(key.transpose(-2, -1) * scale_factor).mT, Bias=None, Error_Dist=self.Error_Dist, show_batch_processing=True)
      attn_weight = matmul_layer(query) #query @ key.transpose(-2, -1) * scale_factor
      attn_weight += attn_bias  # THIS IS WHERE ANY MASKING WOULD TAKE EFFECT!
      attn_weight = torch.softmax(attn_weight, dim=-1)
      attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
      # this is another torch.matmul (set the key at the value)
      matmul_layer_2 = LinearSim(Weight=value.mT, Bias=None, Error_Dist=self.Error_Dist, show_batch_processing=True)
      attn_output =  matmul_layer_2(attn_weight) #attn_weight @ value
      return attn_output