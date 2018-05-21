import numpy as np
import matplotlib.pyplot as plt
import cv2
from monkeydetector import MonkeyDetector

'''
Intrinsic Camera Parameters
'''
width,height = 512,424
d1,d2 = 2000,10000
focal = -365.456
cx,cy = 256,212

# monkey detector library
md = MonkeyDetector(-focal,-focal,cx,cy,[800,800,1200],200,10000)

# get the joints in 3d coordinates
jnts_xyz = np.loadtxt('../monkey_tracker/joints_000000.txt')
jnts_xyz = jnts_xyz[[100, 97, 57, 60, 79, 61, 80, 62, 81, 69, 91, 71, 93, 38, 19, 39, 20, 40, 21, 41, 22, 50, 31]]
# get the depth image
depth = np.array(cv2.imread('../monkey_tracker/depth_000000.png',cv2.IMREAD_UNCHANGED))
# convert to image coordinates
jnts_uvd = md.xyztouvd(jnts_xyz)
# get the center of mass in image coordinates
com_uvd = md.calcCoMRenders(jnts_uvd)
# get the cropped patch
dpt, M, com = md.cropArea3D(depth,com=com_uvd)
# get the relative coordinates
rel_jnts_xyz, rel_jnts_uvd = md.getRelativeCoordinates(jnts_xyz,jnts_uvd,com_uvd,M)
# plot the relative coordinates
plt.imshow(dpt)
plt.scatter(rel_jnts_uvd[:,0],rel_jnts_uvd[:,1],c='r')
plt.show()


### testing run
# we only have the predicted CoM and correspondingly calculated M
# and 3D joints in the relative coordinate system
retrieved_jnts_xyz, retrieved_jnts_uvd = md.getAbsoluteCoordinates(rel_jnts_xyz,com_uvd)
plt.imshow(depth)
plt.scatter(retrieved_jnts_uvd [:,0],retrieved_jnts_uvd [:,1],c='r')
plt.show()