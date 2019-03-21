"""
 ----------------------------------------------------
  @Author: rinsa318
  @Affiliation: Waseda University
  @Email: rinsa@suou.waseda.jp
  @Date: 2019-03-14 16:51:51
  @Last Modified by:   Tsukasa Nozawa
  @Last Modified time: 2019-03-21 23:22:30
 ----------------------------------------------------

[original paper]
[Perez, Patrick](https://ptrckprz.github.io/), Michel Gangnet, and Andrew Blake. 
"Poisson image editing." 
ACM Transactions on graphics (TOG) 22.3 (2003): 313-318. 
[[Paper](http://www.irisa.fr/vista/Papers/2003_siggraph_perez.pdf "Paper")]

[textbook]
https://www.cs.unc.edu/~lazebnik/research/fall08/jia_pan.pdf


[referenced code]
https://github.com/roadfromatoz/poissonImageEditing/blob/master/GradientDomainClone.py
https://github.com/peihaowang/PoissonImageEditing

"""


import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import scipy.io
import cv2
import sys
import os



def get_contour(mask):

  '''
  input: binary mask image
  reeturn:  binary contuor image
  '''

  mask = np.array(mask, dtype=np.uint8)
  eroded = cv2.erode(mask, np.ones((3, 3), np.uint8)) # 0~1
  contours = mask * (1 - eroded) # 0~1, uint8 

  return contours




def get_edge(gray, weight=1):

  '''
  input: gray image
  return: binary edge mask

  --> weight value can change thickness, 1 or 2 is good.

  '''


  ### get edge from filter
  raw_edge = cv2.Canny(gray, 100, 200)
  edge = np.zeros((raw_edge.shape[0], raw_edge.shape[1]), dtype=np.uint8)

  ### make edge bold by using weight vale
  for i in range(edge.shape[0]):
    for j in range(edge.shape[1]):

        if(raw_edge[i][j] != 0):

          for p in range(-weight, weight):
            for q in range(-weight, weight):\

              new_edge_i = min(max(0, i + p), edge.shape[0] - 1)
              new_edge_j = min(max(0, j + q), edge.shape[1] - 1)
              edge[new_edge_i][new_edge_j] = 1

  return edge




def check_existence(mask, id_h, id_w):

  h, w = mask.shape

  if(0 <= id_h and id_h <= h-1 and 0 <= id_w and id_w <= w-1):
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


  ## 1. flag of neigbourhoods pixel.
  ## --> write TRUE if neigbourhoods pixel is exist, FALSE otherwise.
  ## 2. dictionary for omega's index
  ngb_flag = []
  omega_yx = np.zeros((h, w), dtype=np.int32)
  for index in range(omega_list.shape[0]):

    ## pixel location
    i, j = omega_list[index]

    ## create neigbourhoods flag
    ngb_flag.append([check_existence(mask, i, j+1),
                     check_existence(mask, i, j-1),
                     check_existence(mask, i+1, j),
                     check_existence(mask, i-1, j),])

    ## store index to dictionary
    omega_yx[i][j] = index



  return omega_list, np.array(ngb_flag, dtype=bool), omega_yx





def index4omega(omega, id_h, id_w):

  '''
  input: omega, point(id_h, id_w)
  return: index of point in omega
  '''

  p = np.array([id_h, id_w])
  match = np.all(omega==p, axis=1)
  index = np.where(match)[0][0]

  return index





def lap_at_index(source, index, contuor, ngb_flag):

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

  ## take laplacian
  N = np.sum(ngb_flag == True)
  val = (N * source[i, j]
         - (float(ngb_flag[0]==True) * source[i, j+1])
         - (float(ngb_flag[1]==True) * source[i, j-1])
         - (float(ngb_flag[2]==True) * source[i+1, j])
         - (float(ngb_flag[3]==True) * source[i-1, j]))

  return val






def lap_at_index_mixing(source, target, index, contuor, ngb_flag):

  '''
  Function to calculate gradient with respect given index.

  input; src, tar --> one channel, same size
         index    --> omega's coordinate[i, j]
         contour  --> coutour mask(mask.shape[0], mask.shape[1])
         ngb_flag --> neigbourhood's flag at [i, j], (4,), bool


  return grad(source) with Dirichlet boundary condition


                grad_up
            o-----o-----o
            |     A     |
            |     |     |
  grad_left o<----o---->o grad_right
            |     |     |
            |     v     |
            o-----o-----o
                 grad_bottom

  '''
  
  ## current location
  i, j = index


  ## gradient for source image
  grad_right_src = float(ngb_flag[0]==True) * (source[i, j] - source[i, j+1])
  grad_left_src = float(ngb_flag[1]==True) * (source[i, j] - source[i, j-1])
  grad_bottom_src = float(ngb_flag[2]==True) * (source[i, j] - source[i+1, j])
  grad_up_src = float(ngb_flag[3]==True) * (source[i, j] - source[i-1, j])

  ## gradient for target image
  grad_right_tar = float(ngb_flag[0]==True) * (target[i, j] - target[i, j+1])
  grad_left_tar = float(ngb_flag[1]==True) * (target[i, j] - target[i, j-1])
  grad_bottom_tar = float(ngb_flag[2]==True) * (target[i, j] - target[i+1, j])
  grad_up_tar = float(ngb_flag[3]==True) * (target[i, j] - target[i-1, j])


  val = [grad_right_src, grad_left_src, grad_bottom_src, grad_up_src]

  if(abs(grad_right_src) < abs(grad_right_tar)):
    val[0] = grad_right_tar

  if(abs(grad_left_src) < abs(grad_left_tar)):
    val[1] = grad_left_tar

  if(abs(grad_bottom_src) < abs(grad_bottom_tar)):
    val[2] = grad_bottom_tar

  if(abs(grad_up_src) < abs(grad_up_tar)):
    val[3] = grad_up_tar

  return val[0] + val[1] + val[2] + val[3]




def lap_at_index_faltten(source, index, contuor, ngb_flag, edge_mask):

  '''
  Function to calculate gradient with respect given index.

  input; src, tar --> one channel, same size
         index    --> omega's coordinate[i, j]
         contour  --> coutour mask(mask.shape[0], mask.shape[1])
         ngb_flag --> neigbourhood's flag at [i, j], (4,), bool


  return grad(source) with Dirichlet boundary condition


                grad_up
            o-----o-----o
            |     A     |
            |     |     |
  grad_left o<----o---->o grad_right
            |     |     |
            |     v     |
            o-----o-----o
                 grad_bottom

  '''
  
  ## current location
  i, j = index

  ## gradient for source image
  grad_right_src = float(ngb_flag[0]==True) * (source[i, j] - source[i, j+1]) * float(edge_mask[i][j])
  grad_left_src = float(ngb_flag[1]==True) * (source[i, j] - source[i, j-1]) * float(edge_mask[i][j-1])
  grad_bottom_src = float(ngb_flag[2]==True) * (source[i, j] - source[i+1, j]) * float(edge_mask[i][j])
  grad_up_src = float(ngb_flag[3]==True) * (source[i, j] - source[i-1, j]) * float(edge_mask[i-1][j])


  val = grad_right_src + grad_left_src + grad_bottom_src + grad_up_src


  return val




def coefficient_matrix(omega_list, mask, ngb_flag, omega_yx):

  '''
    Create poisson matrix
    --> A

    input: index omega, binary mask image, neigbourhoods flag
    return: Laplacian matrix

  '''  

  ## create empty sparse matrix
  N = omega_list.shape[0]
  A = sp.lil_matrix((N, N), dtype=np.float32)


  for i in range(N):

    ## progress
    progress_bar(i, N-1)


    ## fill 4 or -1
    ## center
    A[i, i] = 4
    id_h, id_w = omega_list[i]

    ## right
    if(ngb_flag[i][0]):
      j = omega_yx[id_h][id_w+1]
      A[i, j] = -1

    ## left
    if(ngb_flag[i][1]):
      j = omega_yx[id_h][id_w-1]
      A[i, j] = -1

    ## bottom
    if(ngb_flag[i][2]):
      j = omega_yx[id_h+1][id_w]
      A[i, j] = -1

    ## up
    if(ngb_flag[i][3]):
      j = omega_yx[id_h-1][id_w]
      A[i, j] = -1

  return A






def importing_gradients(src, tar, omega, contour, ngb_flag):


  '''
    Create gradient matrix
    --> u

    input: source, target image --> 3 channel
           omega                --> index of valid pixel
           contour              --> coutour mask(mask.shape[0], mask.shape[1])
           ngb_flag             --> neigbourhood's flag at [i, j], (4,), bool

    return: laplacian(src)[channel]

  '''  

  ### output array
  u_b = np.zeros(omega.shape[0])
  u_g = np.zeros(omega.shape[0])
  u_r = np.zeros(omega.shape[0])


  ### take laplacian
  for index in range(omega.shape[0]):

    ## progress
    progress_bar(index, omega.shape[0]-1)

    ## apply each color channel
    u_b[index] = lap_at_index(src[:, :, 0], omega[index], contour, ngb_flag[index]) \
                  + constrain(tar[:, :, 0], omega[index], contour, ngb_flag[index])
    u_g[index] = lap_at_index(src[:, :, 1], omega[index], contour, ngb_flag[index]) \
                  + constrain(tar[:, :, 1], omega[index], contour, ngb_flag[index])
    u_r[index] = lap_at_index(src[:, :, 2], omega[index], contour, ngb_flag[index]) \
                  + constrain(tar[:, :, 2], omega[index], contour, ngb_flag[index])


  return u_b, u_g, u_r 




def mixing_gradients(src, tar, omega, contour, ngb_flag):


  '''
    Create gradient matrix
    --> b

    input: source, target image --> 3 channel
           omega                --> index of valid pixel
           contour              --> coutour mask(mask.shape[0], mask.shape[1])
           ngb_flag             --> neigbourhood's flag at [i, j], (4,), bool

    return: laplacian(src)[channel]

  '''  

  ### output array
  u_b = np.zeros(omega.shape[0])
  u_g = np.zeros(omega.shape[0])
  u_r = np.zeros(omega.shape[0])


  ### take laplacian
  for index in range(omega.shape[0]):

    ## progress
    progress_bar(index, omega.shape[0]-1)

    ## apply each color channel
    u_b[index] = lap_at_index_mixing(src[:, :, 0], tar[:, :, 0], omega[index], contour, ngb_flag[index]) \
                + constrain(tar[:, :, 0], omega[index], contour, ngb_flag[index])
    u_g[index] = lap_at_index_mixing(src[:, :, 1], tar[:, :, 1], omega[index], contour, ngb_flag[index]) \
                + constrain(tar[:, :, 1], omega[index], contour, ngb_flag[index])
    u_r[index] = lap_at_index_mixing(src[:, :, 2], tar[:, :, 2], omega[index], contour, ngb_flag[index]) \
                + constrain(tar[:, :, 2], omega[index], contour, ngb_flag[index])


  return u_b, u_g, u_r






def 
texture_flatten(src, tar, omega, contour, ngb_flag):
  
  '''
    Create gradient matrix
    --> u

    input: source, target image --> 3 channel
           omega                --> index of valid pixel
           contour              --> coutour mask(mask.shape[0], mask.shape[1])
           ngb_flag             --> neigbourhood's flag at [i, j], (4,), bool

    return: laplacian(src)[channel]

  '''  

  ### output array
  u_b = np.zeros(omega.shape[0])
  u_g = np.zeros(omega.shape[0])
  u_r = np.zeros(omega.shape[0])


  ### binary mask turned on at a few locations of interest
  ### --> edge image
  gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
  # edge_mask = cv2.Canny(np.array(gray*255, dtype=np.uint8), 100, 200)
  edge_mask = get_edge(np.array(gray*255, dtype=np.uint8))
  cv2.imshow("edge", edge_mask*255)
  cv2.waitKey(0)

  ### take laplacian
  for index in range(omega.shape[0]):

    ## progress
    progress_bar(index, omega.shape[0]-1)

    ## apply each color channel
    i, j = omega[index]

    u_b[index] = lap_at_index_faltten(src[:, :, 0], omega[index], contour, ngb_flag[index], edge_mask) \
                  + constrain(tar[:, :, 0], omega[index], contour, ngb_flag[index])
    u_g[index] = lap_at_index_faltten(src[:, :, 1], omega[index], contour, ngb_flag[index], edge_mask) \
                  + constrain(tar[:, :, 1], omega[index], contour, ngb_flag[index])
    u_r[index] = lap_at_index_faltten(src[:, :, 2], omega[index], contour, ngb_flag[index], edge_mask) \
                  + constrain(tar[:, :, 2], omega[index], contour, ngb_flag[index])


  return u_b, u_g, u_r 




def average_gradients(src, tar, omega, contour, ngb_flag):


  '''
    Create gradient matrix
    --> b

    input: source, target image --> 3 channel
           omega                --> index of valid pixel
           contour              --> coutour mask(mask.shape[0], mask.shape[1])
           ngb_flag             --> neigbourhood's flag at [i, j], (4,), bool

    return: laplacian(src)[channel]

  '''  

  ### output array
  u_b = np.zeros(omega.shape[0])
  u_g = np.zeros(omega.shape[0])
  u_r = np.zeros(omega.shape[0])


  ### take laplacian
  for index in range(omega.shape[0]):

    ## progress
    progress_bar(index, omega.shape[0]-1)

    ## apply each color channel
    u_b_src = lap_at_index(src[:, :, 0], omega[index], contour, ngb_flag[index])
    u_g_src = lap_at_index(src[:, :, 1], omega[index], contour, ngb_flag[index])
    u_r_src = lap_at_index(src[:, :, 2], omega[index], contour, ngb_flag[index])

    u_b_tar = lap_at_index(tar[:, :, 0], omega[index], contour, ngb_flag[index])
    u_g_tar = lap_at_index(tar[:, :, 1], omega[index], contour, ngb_flag[index])
    u_r_tar = lap_at_index(tar[:, :, 2], omega[index], contour, ngb_flag[index])

    u_b[index] = (u_b_tar + u_b_src) / 2.0 \
                  + constrain(tar[:, :, 0], omega[index], contour, ngb_flag[index])
    u_g[index] = (u_g_tar + u_g_src) / 2.0 \
                  + constrain(tar[:, :, 1], omega[index], contour, ngb_flag[index])
    u_r[index] = (u_r_tar + u_r_src) / 2.0 \
                  + constrain(tar[:, :, 2], omega[index], contour, ngb_flag[index])




  return u_b, u_g, u_r





def constrain(target, index, contuor, ngb_flag):

  '''
  Function to set constraint grad(source) = target at boundary

  input; tar      --> one channel, same size
         index    --> omega's coordinate[i, j]
         contour  --> coutour mask(mask.shape[0], mask.shape[1])
         ngb_flag --> neigbourhood's flag at [i, j], (4,), bool


  return Dirichlet boundary condition for index
  '''
  
  ## current location
  i, j = index


  ## In order to use "Dirichlet boundary condition",
  ## if on boundry, add in target intensity --> set constraint grad(source) = target at boundary
  if(contuor[i][j]==1):
    val = (float(ngb_flag[0]==False) * target[i, j+1]
           + float(ngb_flag[1]==False) * target[i, j-1]
           + float(ngb_flag[2]==False) * target[i+1, j]
           + float(ngb_flag[3]==False) * target[i-1, j])
    return val

  ## If not on boundry, just take grad.
  else:
    val = 0.0
    return val






def progress_bar(n, N):

  '''
  print current progress
  '''

  step = 2
  percent = float(n) / float(N) * 100

  ## convert percent to bar
  current = "#" * int(percent//step)
  remain = " " * int(100/step-int(percent//step))
  bar = "|{}{}|".format(current, remain)
  print("\r{}:{:3.0f}[%]".format(bar, percent), end="", flush=True)
  





def poisson_blend(src, mask, tar, method, output_dir):

  '''
  solve Au = b

  -> A: poisson matrix
     b: (g)
     u: final pixel value

  '''


  ### create contour mask
  contour = get_contour(mask) # uint8
  mask = np.array(mask, dtype=np.uint8)




  ### get omega, neigbourhoods flag
  omega, ngb_flag, yx_omega = indicies(mask)




  ### fill A
  print("step1: filling coefficient matrix: A")
  A = sp.lil_matrix((omega.shape[0], omega.shape[0]), dtype=np.float32) 

  if(os.path.isfile("{0}/A.mat".format(output_dir))):
    A = scipy.io.loadmat("{}/A".format(output_dir))["A"]
    print("load coefficient matrix: A from .mat file\n")
  else:
    A = coefficient_matrix(omega, mask, ngb_flag, yx_omega)
    scipy.io.savemat("{}/A".format(output_dir), {"A":A}) 
    print("\n")



  ### fill u
  ### --> each color channel
  print("step2: filling gradient matrix: b")
  u_b = np.zeros(omega.shape[0])
  u_g = np.zeros(omega.shape[0])
  u_r = np.zeros(omega.shape[0])

  ## select process type
  if(method == "import"):
    u_b, u_g, u_r = importing_gradients(src, tar, omega, contour, ngb_flag)
    print("\n")
  if(method == "mix"):
    u_b, u_g, u_r =  mixing_gradients(src, tar, omega, contour, ngb_flag)
    print("\n")
  if(method == "average"):
    u_b, u_g, u_r =  average_gradients(src, tar, omega, contour, ngb_flag)
    print("\n")
  if(method == "flatten"):
    u_b, u_g, u_r = texture_flatten(src, tar, omega, contour, ngb_flag)
    print("\n")



  ### solve
  print("step3: solve Au = b")
  x_b, info_b = sp.linalg.cg(A, u_b)
  x_g, info_g = sp.linalg.cg(A, u_g)
  x_r, info_r = sp.linalg.cg(A, u_r)
  print("done!\n")



  ### create output by using x
  blended = tar.copy()
  overlapped = tar.copy()

  for index in range(omega.shape[0]):

    i, j = omega[index]
  
    ## normal
    blended[i][j][0] = np.clip(x_b[index], 0.0, 1.0)
    blended[i][j][1] = np.clip(x_g[index], 0.0, 1.0)
    blended[i][j][2] = np.clip(x_r[index], 0.0, 1.0)

    ## overlapping
    overlapped[i][j][0] = src[i][j][0]
    overlapped[i][j][1] = src[i][j][1]
    overlapped[i][j][2] = src[i][j][2]


  return (np.array(blended*255, dtype=np.uint8), 
          np.array(overlapped*255, dtype=np.uint8))
