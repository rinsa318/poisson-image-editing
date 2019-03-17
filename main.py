"""
 ----------------------------------------------------
  @Author: rinsa318
  @Affiliation: Waseda University
  @Email: rinsa@suou.waseda.jp
  @Date: 2019-03-14 16:51:01
  @Last Modified by:   rinsa318
  @Last Modified time: 2019-03-18 00:57:46
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

outname_blended = "{0}/blended_src_{1}_tar_{2}{3}".format(output_dir, filename_src, filename_tar, ext_tar)
outname_overlapped = "{0}/overlapped_src_{1}_tar_{2}{3}".format(output_dir, filename_src, filename_tar, ext_tar)
outname_merged = "{0}/merged_result{1}".format(output_dir, ext_tar)
outfile = [outname_blended, outname_overlapped, outname_merged]
# print(output_path)


### 3. load images
src = np.array(cv2.imread(src_path, 1), dtype=np.uint8)
tar = np.array(cv2.imread(tar_path, 1), dtype=np.uint8)
mask = np.array(cv2.imread(src_mask_path, 0), dtype=np.uint8)
ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)
# print(src.shape)
# print(mask.shape)
# print(tar.shape)


### 4. apply poisson image editing
blended, overlapped = poisson.poisson_blend(src, mask/255.0, tar)




### 5. save result
print("save blended image as --> {}".format(outfile))
merged_result = np.hstack((src, cv2.merge((mask, mask, mask)), tar, overlapped, blended))
cv2.imwrite(outname_merged, merged_result)
cv2.imwrite(outname_blended, blended)
cv2.imwrite(outname_overlapped, overlapped)
# cv2.imshow("out", merged_result)
# cv2.waitKey(0)

