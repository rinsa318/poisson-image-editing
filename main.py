"""
 ----------------------------------------------------
  @Author: rinsa318
  @Affiliation: Waseda University
  @Email: rinsa@suou.waseda.jp
  @Date: 2019-03-14 16:51:01
  @Last Modified by:   rinsa318
  @Last Modified time: 2019-03-17 19:54:45
 ----------------------------------------------------

  Usage:
   python main.py argvs[1] argvs[2] argvs[3]...
  
   argvs[1]  :  source image
   argvs[2]  :  source image's mask
   argvs[3]  :  target


"""


import os
import sys
import cv2
import numpy as np


## my function
import poissonimageediting as poisson



### 1. prepare config
argvs = sys.argv
src_path = argvs[1]
src_mask_path = argvs[2]
tar_path = argvs[3]
filename_src, ext_src = os.path.splitext( os.path.basename(src_path) )
filename_tar, ext_tar = os.path.splitext( os.path.basename(tar_path) )
src_dir, filefullname_src = os.path.split( src_path )
tar_dir, filefullname_tar = os.path.split( tar_path )
print("source image --> " + filefullname_src)
print("target image --> " + filefullname_tar)



### 2. for output
output_dir = "{0}/result".format(tar_dir)
if(not(os.path.exists(output_dir))):
  os.mkdir(output_dir)

outname_blend = "{0}/blend_src_{1}_tar_{2}{3}".format(output_dir, filename_src, filename_tar, ext_tar)
outname_simplecopy = "{0}/simplecopy_src_{1}_tar_{2}{3}".format(output_dir, filename_src, filename_tar, ext_tar)
outname_merge = "{0}/merge_result{1}".format(output_dir, ext_tar)
outfile = [outname_blend, outname_simplecopy, outname_merge]
# print(output_path)



### 3. load images
src = np.array(cv2.imread(src_path, 1), dtype=np.uint8)
tar = np.array(cv2.imread(tar_path, 1), dtype=np.uint8)
mask = np.array(cv2.imread(src_mask_path, 0), dtype=np.uint8)
ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)



### 4. apply poisson image editing
blend, simple_copy = poisson.poisson_blend(src, mask/255.0, tar)




### 5. save result
print("save blended image as --> {}".format(outfile))
temp1 = np.hstack((src, tar))
temp2 = np.hstack((simple_copy, blend))
result = np.hstack((temp1, temp2))
cv2.imwrite(outname_merge, result)
cv2.imwrite(outname_blend, blend)
cv2.imwrite(outname_simplecopy, simple_copy)
cv2.imshow("out", result)
cv2.waitKey(0)

