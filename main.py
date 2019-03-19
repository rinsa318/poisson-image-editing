"""
 ----------------------------------------------------
  @Author: rinsa318
  @Affiliation: Waseda University
  @Email: rinsa@suou.waseda.jp
  @Date: 2019-03-14 16:51:01
  @Last Modified by:   rinsa318
  @Last Modified time: 2019-03-19 23:56:00
 ----------------------------------------------------

  Usage:
   python main.py argvs[1] argvs[2] argvs[3] argvs[4]
  
   argvs[1]  :  source image
   argvs[2]  :  source image's mask
   argvs[3]  :  target image
   argvs[4]  :  method name --> import, mix

"""


import os
import sys
import cv2
import numpy as np


## my function
import poissonimageediting as poisson

print(__doc__)





### 1. prepare config
argvs = sys.argv
src_path = argvs[1]
src_mask_path = argvs[2]
tar_path = argvs[3]
method = argvs[4]
filename_src, ext_src = os.path.splitext( os.path.basename(src_path) )
filename_tar, ext_tar = os.path.splitext( os.path.basename(tar_path) )
src_dir, filefullname_src = os.path.split( src_path )
tar_dir, filefullname_tar = os.path.split( tar_path )
print("source image --> {0}".format(filefullname_src))
print("target image --> {0}\n".format(filefullname_tar))



### 2. for output
output_dir = "{0}/result".format(tar_dir)
if(not(os.path.exists(output_dir))):
  os.mkdir(output_dir)

outname = "{0}/result_{1}{2}".format(output_dir, method, ext_tar)
outname_overlapped = "{0}/overlapped{1}".format(output_dir, ext_tar)
outname_merged = "{0}/merged_result_{1}{2}".format(output_dir, method, ext_tar)
outfile = [outname, outname_overlapped, outname_merged]




### 3. load images
src = np.array(cv2.imread(src_path, 1)/255.0, dtype=np.float32)
tar = np.array(cv2.imread(tar_path, 1)/255.0, dtype=np.float32)
mask = np.array(cv2.imread(src_mask_path, 0), dtype=np.uint8)
ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)



### 4. apply poisson image editing
blended, overlapped = poisson.poisson_blend(src, mask/255.0, tar, method, output_dir)




### 5. save result
print("save blended image as \n--> \n{0}\n{1}\n{2}".format(outfile[0], outfile[1], outfile[2]))
merged_result = np.hstack((np.array(src*255, dtype=np.uint8), cv2.merge((mask, mask, mask)), np.array(tar*255, dtype=np.uint8), overlapped, blended))
cv2.imwrite(outname_merged, merged_result)
cv2.imwrite(outname_overlapped, overlapped)
cv2.imwrite(outname, blended)
# cv2.imshow("output", merged_result)
# cv2.waitKey(0)

