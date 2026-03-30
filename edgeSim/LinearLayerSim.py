import torch
import torch.nn as nn
import numpy as np


def quick_split_W3D(matrix):
    '''
    This function splits a weight matrix, W, into two parts: W_+ and W_-, such that W=(W_+)+(W_-). This is done to mitigate
    error accumulation of the network's fully connected layer. Effectively, this offers an alternative to the previous method
    of matrix preprocessing, which accumulated excess error by performing element-wise addition in the weight matrix by its 
    max value to get it in range [0,a]

    INPUT: matrix, a tensor from the state dictionary of the trained model
    '''
    # form the negative of the input matrix
    negative_matrix=-1*matrix
    # use RELU to find the positive matrix W_+
    W_pos=nn.ReLU(inplace=True)(torch.tensor(matrix))
    # now, use RELU of the **negative matrix** to find the negative factored matrix W_-
    W_neg=nn.ReLU(inplace=True)(torch.tensor(negative_matrix))
    assert ((W_pos - W_neg).cpu().numpy() == matrix.cpu().numpy()).all()
    return W_pos, W_neg


def draw_2D_Matrix_from_3D_diffdist(Diff_vol,m,n):
  '''
  diff volume here is 2d
  '''
  #channel_index=np.random.randint(0,Diff_vol.shape[2]+1)
  idx_0=np.random.randint(0,Diff_vol.shape[0]-m+1)
  idx_1=np.random.randint(0,Diff_vol.shape[1]-n+1)
  return np.array(Diff_vol[idx_0:idx_0+m, idx_1:idx_1+n])



def Matmul_with_2d_diffdist_elements(error_surface,X,W):
  '''
  performs a simulated matrix product between two 2d tensors
  X: 2d input tensor of size [Batch_number, N]
  W: 2d weight tensor of size [M,N]
  Note: THIS ASSUMES THAT X AND W ARE BOTH PREPROCESSED TO BE IN THE RANGE [0,1]/ [-1,1]
  '''
  X_lst=[]
# loop through rows of the input to form the 3D input tensor
  for row_indx in range(X.shape[0]):
    row_to_copy=X[row_indx]
    matrix_from_row=np.tile(row_to_copy,(W.shape[0],1))
    X_lst.append(matrix_from_row)

  X_vstacked=np.vstack(X_lst)

  stacked_weight_matrices=[]
  for matrix_indx in range(X.shape[0]):
    weight_matrix_to_stack=W
    stacked_weight_matrices.append(weight_matrix_to_stack)
 
  W_vstacked=np.vstack(stacked_weight_matrices)
  # FORM THE HADAMARD PRODUCT 
  Hadamard_prod=W_vstacked*X_vstacked
  if np.min(Hadamard_prod)<0 or np.max(Hadamard_prod)>1:
      raise Exception("ERROR: Hadamard product in fully connected layer not in [0,1]")

  # ADD MATRIX OF ERRORS TO HADAMARD PRODUCT
  Hadamard_prod_with_errors=Hadamard_prod + draw_2D_Matrix_from_3D_diffdist(error_surface, Hadamard_prod.shape[0], Hadamard_prod.shape[1]) 
  # SUM HADAMARD PRODUCT TO FORM MATRIX PRODUCT
  summed_Hadamard_prod=np.sum(Hadamard_prod_with_errors, axis=1)
  matrix_prod=np.reshape(summed_Hadamard_prod,(X.shape[0],W.shape[0]))
  return torch.tensor(matrix_prod)