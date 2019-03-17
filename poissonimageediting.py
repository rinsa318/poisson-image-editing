"""
 ----------------------------------------------------
  @Author: rinsa318
  @Affiliation: Waseda University
  @Email: rinsa@suou.waseda.jp
  @Date: 2019-03-14 16:51:51
  @Last Modified by:   rinsa318
  @Last Modified time: 2019-03-17 19:54:27
 ----------------------------------------------------

[textbook]
https://www.cs.unc.edu/~lazebnik/research/fall08/


[referenced code]
https://github.com/roadfromatoz/poissonImageEditing/blob/master/GradientDomainClone.py


"""


import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import cv2




def get_contour(mask):

  '''
  input: binary mask image
  reeturn:  binary contuor image
  '''

  mask = np.array(mask, dtype=np.uint8)
  eroded = cv2.erode(mask, np.ones((3, 3), np.uint8)) # 0~1
  contours = mask * (1 - eroded) # 0~1, uint8 

  return contours





def check_existence(mask, id_h, id_w):

  h, w = mask.shape

  if(0 <= id_h and id_h <= h and 0 <= id_w and id_w <= w):
    if(mask[id_h][id_w]==1):
      return bool(1)

    else:
      return bool(0)

  else:
    return bool(0)





def indicies(mask):

  '''
  input: binary mask image
  reeturn:  index of valid pixel
  '''

  ## height and width
  h, w = mask.shape


  ## index of inside mask --> omega
  omega = np.nonzero(mask)
  y = np.reshape(omega[0], [omega[0].shape[0], 1])
  x = np.reshape(omega[1], [omega[1].shape[0], 1])
  omega_list = np.concatenate([y, x], 1)


  ## flag of neigbourhoods pixel.
  ## write TRUE if neigbourhoods pixel is exist, FALSE otherwise.
  ngb_flag = []
  for index in range(omega_list.shape[0]):
    i, j = omega_list[index]

    ngb_flag.append([check_existence(mask, i, j+1),
                     check_existence(mask, i, j-1),
                     check_existence(mask, i+1, j),
                     check_existence(mask, i-1, j),])

  ngb_flag = np.array(ngb_flag, dtype=bool)

  return omega_list, ngb_flag





def index4omega(omega, id_h, id_w):

  '''
  input: omega, point(id_h, id_w)
  return: index of point in omega
  '''

  p = np.array([id_h, id_w])
  match = np.all(omega==p, axis=1)
  index = np.where(match)[0][0]

  return index




def laplacian_at_index(source, target, index, contuor, ngb_flag):

  '''
  Function to calculate gradient with respect given index.

  input; src, tar --> one channel, same size
         index    --> omega's coordinate[i, j]
         contour  --> coutour mask(mask.shape[0], mask.shape[1])
         ngb_flag --> neigbourhood's flag at [i, j], (4,), bool


  return grad(source) with Dirichlet boundary condition
  '''
  
  ## current location
  i, j = index


  ## In order to use "Dirichlet boundary condition",
  ## if on boundry, add in target intensity --> set constraint grad(source) = target at boundary
  if(contuor[i][j]==1):
    val = (4 * source[i,j]
           - source[i, j+1]
           - source[i, j-1]
           - source[i+1, j]
           - source[i-1, j])
    val += (int(ngb_flag[0]==False) * target[i, j+1]
           + int(ngb_flag[1]==False) * target[i, j-1]
           + int(ngb_flag[2]==False) * target[i+1, j]
           + int(ngb_flag[3]==False) * target[i-1, j])
    return val

  ## If not on boundry, just take grad.
  else:
    val = (4 * source[i,j]
           - source[i, j+1]
           - source[i, j-1]
           - source[i+1, j]
           - source[i-1, j])
    return val





def coefficient_matrix(omega_list, mask, ngb_flag):

  '''
    Create poisson matrix
    --> A

    input: index omega, binary mask image, neigbourhoods flag
    return: Laplacian matrix

  '''  

  ## create empty sparse matrix
  N = omega_list.shape[0]
  A = sp.lil_matrix((N, N))


  for i in range(N):
    A[i, i] = 4
    id_h, id_w = omega_list[i]

    ## fill -1 for surrounding pixel
    ## right
    if(ngb_flag[i][0]):
      j = index4omega(omega_list, id_h, id_w+1)
      A[i, j] = -1

    ## left
    if(ngb_flag[i][1]):
      j = index4omega(omega_list, id_h, id_w-1)
      A[i, j] = -1

    ## bottom
    if(ngb_flag[i][2]):
      j = index4omega(omega_list, id_h+1, id_w)
      A[i, j] = -1

    ## up
    if(ngb_flag[i][3]):
      j = index4omega(omega_list, id_h-1, id_w)
      A[i, j] = -1

  return A




### Run Poisson image editing on given a src, tar, and src_mask
def poisson_blend(src, mask, tar):

  '''
  solve Ax = b

  -> A: poisson matrix
     b: gradient(g)
     x: final pixel value

  '''

  ## output
  blend = tar.copy()
  simple_copy = tar.copy()


  ## create contour mask
  contour = get_contour(mask) # uint8
  mask = np.array(mask, dtype=np.uint8)


  ## get omega, neigbourhoods flag
  omega, ngb_flag = indicies(mask)


  ## fill A
  print("filling coefficient matrix: A")
  A = coefficient_matrix(omega, mask, ngb_flag)
  # print(A.shape)
  # print(A.dtype)


  ## fill b for each color channel
  print("filling gradient matrix: b")
  b_b = np.zeros(omega.shape[0])
  b_g = np.zeros(omega.shape[0])
  b_r = np.zeros(omega.shape[0])

  for index in range(omega.shape[0]):
    y, x = omega[index]
    b_b[index] = laplacian_at_index(src[:, :, 0], tar[:, :, 0], omega[index], contour, ngb_flag[index])
    b_g[index] = laplacian_at_index(src[:, :, 1], tar[:, :, 1], omega[index], contour, ngb_flag[index])
    b_r[index] = laplacian_at_index(src[:, :, 2], tar[:, :, 2], omega[index], contour, ngb_flag[index])


  print("solve Ax = b")
  x_b, info_b = sp.linalg.cg(A, b_b)
  x_g, info_g = sp.linalg.cg(A, b_g)
  x_r, info_r = sp.linalg.cg(A, b_r)
  print(".....done!")



  ## create output by using x
  for index in range(omega.shape[0]):

    i, j = omega[index]
    blend[i][j][0] = np.clip(x_b[index], 0, 255)
    blend[i][j][1] = np.clip(x_g[index], 0, 255)
    blend[i][j][2] = np.clip(x_r[index], 0, 255)

    simple_copy[i][j][0] = src[i][j][0]
    simple_copy[i][j][1] = src[i][j][1]
    simple_copy[i][j][2] = src[i][j][2]


  return blend, simple_copy
