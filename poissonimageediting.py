"""
 ----------------------------------------------------
  @Author: rinsa318
  @Affiliation: Waseda University
  @Email: rinsa@suou.waseda.jp
  @Date: 2019-03-14 16:51:51
  @Last Modified by:   rinsa318
  @Last Modified time: 2019-03-19 01:53:31
 ----------------------------------------------------

[original paper]
[Perez, Patrick](https://ptrckprz.github.io/), Michel Gangnet, and Andrew Blake. 
"Poisson image editing." 
ACM Transactions on graphics (TOG) 22.3 (2003): 313-318. 
[[Paper](http://www.irisa.fr/vista/Papers/2003_siggraph_perez.pdf "Paper")]

[textbook]
https://www.cs.unc.edu/~lazebnik/research/fall08/
https://www.cs.unc.edu/~lazebnik/research/fall08/jia_pan.pdf


[referenced code]
https://github.com/roadfromatoz/poissonImageEditing/blob/master/GradientDomainClone.py
https://github.com/peihaowang/PoissonImageEditing
https://github.com/datawine/poisson_image_editing

"""


import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import cv2
import sys




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





def laplacian_at_index(source, index, contuor, ngb_flag):

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
         - (int(ngb_flag[0]==True) * source[i, j+1])
         - (int(ngb_flag[1]==True) * source[i, j-1])
         - (int(ngb_flag[2]==True) * source[i+1, j])
         - (int(ngb_flag[3]==True) * source[i-1, j]))

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
  A = sp.lil_matrix((N, N), dtype=np.float32)


  for i in range(N):

    ## progress
    progress(i, N-1)

    A[i, i] = 4
    # A[i, i] = np.sum(ngb_flag[i] == True)
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






def gradient_matrix(src, tar, omega, contour, ngb_flag):


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
  b_b = np.zeros(omega.shape[0])
  b_g = np.zeros(omega.shape[0])
  b_r = np.zeros(omega.shape[0])


  ### take laplacian
  for index in range(omega.shape[0]):

    ## progress
    progress(index, omega.shape[0]-1)

    ## apply each color channel
    y, x = omega[index]
    b_b[index] = laplacian_at_index(src[:, :, 0], omega[index], contour, ngb_flag[index]) + constrain(tar[:, :, 0], omega[index], contour, ngb_flag[index])
    b_g[index] = laplacian_at_index(src[:, :, 1], omega[index], contour, ngb_flag[index]) + constrain(tar[:, :, 1], omega[index], contour, ngb_flag[index])
    b_r[index] = laplacian_at_index(src[:, :, 2], omega[index], contour, ngb_flag[index]) + constrain(tar[:, :, 2], omega[index], contour, ngb_flag[index])


  return b_b, b_g, b_r 




def mixing_gradient_matrix(src, tar, omega, contour, ngb_flag):


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
  b_b = np.zeros(omega.shape[0])
  b_g = np.zeros(omega.shape[0])
  b_r = np.zeros(omega.shape[0])


  ### take laplacian
  for index in range(omega.shape[0]):

    ## progress
    progress(index, omega.shape[0]-1)

    ## apply each color channel
    y, x = omega[index]
    b_b_s = laplacian_at_index(src[:, :, 0], omega[index], contour, ngb_flag[index])
    b_g_s = laplacian_at_index(src[:, :, 1], omega[index], contour, ngb_flag[index])
    b_r_s = laplacian_at_index(src[:, :, 2], omega[index], contour, ngb_flag[index])

    b_b_t = laplacian_at_index(tar[:, :, 0], omega[index], contour, ngb_flag[index])
    b_g_t = laplacian_at_index(tar[:, :, 1], omega[index], contour, ngb_flag[index])
    b_r_t = laplacian_at_index(tar[:, :, 2], omega[index], contour, ngb_flag[index])

    # if(abs(b_b_s) + abs(b_g_s) + abs(b_r_s) < abs(b_b_t) + abs(b_g_t) + abs(b_r_t)):
    #   b_b[index] = b_b_t + constrain(tar[:, :, 0], omega[index], contour, ngb_flag[index])
    #   b_g[index] = b_g_t + constrain(tar[:, :, 1], omega[index], contour, ngb_flag[index])
    #   b_r[index] = b_r_t + constrain(tar[:, :, 2], omega[index], contour, ngb_flag[index])
    # else:
    #   b_b[index] = b_b_s + constrain(tar[:, :, 0], omega[index], contour, ngb_flag[index])
    #   b_g[index] = b_g_s + constrain(tar[:, :, 1], omega[index], contour, ngb_flag[index])
    #   b_r[index] = b_r_s + constrain(tar[:, :, 2], omega[index], contour, ngb_flag[index])

    # print(abs(b_b_s), abs(b_b_t))
    if(abs(b_b_s) < abs(b_b_t)):
      b_b[index] = b_b_t + constrain(tar[:, :, 0], omega[index], contour, ngb_flag[index])
    else:
      b_b[index] = b_b_s + constrain(tar[:, :, 0], omega[index], contour, ngb_flag[index])

    if(abs(b_g_s) < abs(b_g_t)):
      b_g[index] = b_g_t + constrain(tar[:, :, 1], omega[index], contour, ngb_flag[index])
    else:
      b_g[index] = b_g_s + constrain(tar[:, :, 1], omega[index], contour, ngb_flag[index])

    if(abs(b_r_s) < abs(b_r_t)):
      b_r[index] = b_r_t + constrain(tar[:, :, 2], omega[index], contour, ngb_flag[index])
    else:
      b_r[index] = b_r_s + constrain(tar[:, :, 2], omega[index], contour, ngb_flag[index])



  return b_b, b_g, b_r




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






def progress(n, N):

  '''
  print current progress
  '''

  percent = float(n) / float(N) * 100

  ## convert percent to bar
  current = "#" * int(percent//2)
  # current = "=" * int(percent//2)
  remain = " " * int(100/2-int(percent//2))
  bar = "|{}{}|".format(current, remain)# + "#" * int(percent//2) + " " * int(100/2-int(percent//2)) + "|"
  print("\r{}: {:3.0f}[%]".format(bar, percent), end="", flush=True)
  





def poisson_blend(src, mask, tar):

  '''
  solve Ax = b

  -> A: poisson matrix
     b: gradient(g)
     x: final pixel value

  '''

  ### output
  blended = tar.copy()
  blended_mixing = tar.copy()
  overlapped = tar.copy()


  ### create contour mask
  contour = get_contour(mask) # uint8
  mask = np.array(mask, dtype=np.uint8)

  ### get omega, neigbourhoods flag
  omega, ngb_flag = indicies(mask)


  ### fill A
  print("step1: filling coefficient matrix: A")
  A = coefficient_matrix(omega, mask, ngb_flag)
  print("\ndone!\n")
  # print(A.shape)
  # print(A.dtype)


  ### fill b
  ### --> each color channel
  print("step2: filling gradient matrix: b")
  b_b, b_g, b_r = gradient_matrix(src, tar, omega, contour, ngb_flag)
  b_b2, b_g2, b_r2 =  mixing_gradient_matrix(src, tar, omega, contour, ngb_flag)
  print("\ndone!\n")


  ### solve
  print("step3: solve Ax = b")
  x_b, info_b = sp.linalg.cg(A, b_b)
  x_g, info_g = sp.linalg.cg(A, b_g)
  x_r, info_r = sp.linalg.cg(A, b_r)
  x_b2, info_b2 = sp.linalg.cg(A, b_b2)
  x_g2, info_g2 = sp.linalg.cg(A, b_g2)
  x_r2, info_r2 = sp.linalg.cg(A, b_r2)
  print("done!\n")



  ### create output by using x
  for index in range(omega.shape[0]):

    i, j = omega[index]
    blended[i][j][0] = np.clip(x_b[index], 0.0, 1.0)
    blended[i][j][1] = np.clip(x_g[index], 0.0, 1.0)
    blended[i][j][2] = np.clip(x_r[index], 0.0, 1.0)

    blended_mixing[i][j][0] = np.clip(x_b2[index], 0.0, 1.0)
    blended_mixing[i][j][1] = np.clip(x_g2[index], 0.0, 1.0)
    blended_mixing[i][j][2] = np.clip(x_r2[index], 0.0, 1.0)

    overlapped[i][j][0] = src[i][j][0]
    overlapped[i][j][1] = src[i][j][1]
    overlapped[i][j][2] = src[i][j][2]


  return np.array(blended*255, dtype=np.uint8), np.array(blended_mixing*255, dtype=np.uint8), np.array(overlapped*255, dtype=np.uint8)
