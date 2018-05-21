"""
A Monkey Detector Interface -- there are two different types of input to handle here:
Synthetic Renders & Kinect Data. This API offers the following functions

@init                       initialize the module with camera parameters
@calculateCoM               given all the joints of the renders, return the center of mass
@xyztouvd                   project 3D coordinates into the image plane
@uvdtoxyz                   convert image coordinates to 3D coordinates
@getRelativeCoordinates     given the 2D center of mass, and groundtruths get the relative coordinates (for training)
@transformPoint2D           apply a transformation
"""

import numpy
import os
import cv2
from scipy import stats, ndimage
import matplotlib.pyplot as plt
import matplotlib as mpl


class MonkeyDetector(object):
    """
    Detect monkey based on simple heuristic, centered at Center of Mass
    """

    RESIZE_BILINEAR = 0
    RESIZE_CV2_NN = 1
    RESIZE_CV2_LINEAR = 2

    def __init__(self, fx, fy, ux, uy, cube, d1, d2, importer=None):
        """
        Constructor
        :param dpt: depth image
        :param fx: camera focal length
        :param fy: camera focal length
        :param ux: principal x-point (half width)
        :param uy: principal y-point (half height)
        :param d1: near plane of the camera
        :param d2: far plane of the camera
        """
        #self.dpt = dpt
        #self.maxDepth = min(d2, dpt.max())
        #self.minDepth = max(d1, dpt.min())

        self.maxDepth = d2
        self.minDepth = d1

        # set values out of range to 0
        #self.dpt[self.dpt > self.maxDepth] = 0.
        #self.dpt[self.dpt < self.minDepth] = 0.

        # camera settings
        self.fx = fx
        self.fy = fy
        self.ux = ux
        self.uy = uy

        if len(cube) != 3:
            raise ValueError("Volume must be 3D")
        # the 3d volume to crop
        self.cube = cube

        # depth resize method
        self.resizeMethod = self.RESIZE_CV2_NN

    def calculateCoM(self, dpt):
        """
        Calculate the center of mass
        :param dpt: depth image
        :return: (x,y,z) center of mass
        """

        dc = dpt.copy()
        dc[dc < self.minDepth] = 0
        dc[dc > self.maxDepth] = 0
        cc = ndimage.measurements.center_of_mass(dc > 0)
        num = numpy.count_nonzero(dc)
        com = numpy.array((cc[1]*num, cc[0]*num, dc.sum()), numpy.float)

        if num == 0:
            return numpy.array((0, 0, 0), numpy.float)
        else:
            return com/num

    def xyztouvd(self, jnts_xyz):
        """
        convert from 3d coordinates to image coordinates
        :return:
        """
        if jnts_xyz.ndim == 1:
            jnt_uvd = numpy.zeros((3,), numpy.float32)
            if jnts_xyz[2] == 0.:
                jnt_uvd[0] = self.ux
                jnt_uvd[1] = self.uy
                return jnt_uvd
            jnt_uvd[0] = self.ux - jnts_xyz[0] / jnts_xyz[2] * self.fx
            jnt_uvd[1] = jnts_xyz[1] / jnts_xyz[2] * self.fy + self.uy
            jnt_uvd[2] = -jnts_xyz[2]

        elif jnts_xyz.ndim == 2:
            jnt_uvd = numpy.zeros((jnts_xyz.shape[0], 3), numpy.float32)
            for i in range(jnts_xyz.shape[0]):
                if jnts_xyz[i,2] == 0.:
                    jnt_uvd[i,0] = self.ux
                    jnt_uvd[i,1] = self.uy
                    continue
                jnt_uvd[i,0] = self.ux - jnts_xyz[i,0] / jnts_xyz[i,2] * self.fx
                jnt_uvd[i,1] = jnts_xyz[i,1] / jnts_xyz[i,2] * self.fy + self.uy
                jnt_uvd[i,2] = -jnts_xyz[i,2]
        return jnt_uvd

    def uvdtoxyz(self, jnts_uvd):
        """
        convert from image coordinates to camera coordinates
        :param jnts_uvd: single joint in (u,v,d) with u,v in image coordinates and d in mm
        NOTE: 'd', 'fx' and 'fy' are positive quantities here
        :return: normalized joints in mm
        """
        if jnts_uvd.ndim == 1:
            jnt_xyz = numpy.zeros((3,), numpy.float32)
            jnt_xyz[0] = (self.ux - jnts_uvd[0]) * jnts_uvd[2] / (-self.fx)
            jnt_xyz[1] = (jnts_uvd[1] - self.uy) * jnts_uvd[2] / (-self.fy)
            jnt_xyz[2] = -jnts_uvd[2]
        elif jnts_uvd.ndim == 2:
            jnt_xyz = numpy.zeros((jnts_uvd.shape[0], 3), numpy.float32)
            for i in range(jnts_uvd.shape[0]):
                jnt_xyz[i,0] = (self.ux - jnts_uvd[i,0]) * jnts_uvd[i,2] / (-self.fx)
                jnt_xyz[i,1] = (jnts_uvd[i,1] - self.uy) * jnts_uvd[i,2] / (-self.fy)
                jnt_xyz[i,2] = -jnts_uvd[i,2]
        return jnt_xyz

    def checkImage(self, tol):
        """
        Check if there is some content in the image
        :param tol: tolerance
        :return:True if image is contentful, otherwise false
        """
        if numpy.std(self.dpt) < tol:
            return False
        else:
            return True

    def getNDValue(self):
        """
        Get value of not defined depth value distances
        :return:value of not defined depth value
        """
        import ipdb; ipdb.set_trace()
        if self.dpt[self.dpt < self.minDepth].shape[0] > self.dpt[self.dpt > self.maxDepth].shape[0]:
            return stats.mode(self.dpt[self.dpt < self.minDepth])[0][0]
        else:
            return stats.mode(self.dpt[self.dpt > self.maxDepth])[0][0]

    def calcCoMRenders(self,jnts):
        """
        :param jnts: joints in 3D coordinates
        :return: center of mass in 3D coordinates
        """
        assert jnts.ndim == 2, 'input must be the 3D coordinates of all monkey joints'
        return numpy.sum(jnts,axis=0)/jnts.shape[0]

    def comToBounds(self, com, size):
        """
        Calculate boundaries, project to 3D, then add offset and backproject to 2D (ux, uy are canceled)
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :return: xstart, xend, ystart, yend, zstart, zend
        """
        zstart = com[2] - size[2] / 2.
        zend = com[2] + size[2] / 2.
        xstart = int(numpy.floor((com[0] * com[2] / self.fx - size[0] / 2.) / com[2]*self.fx))
        xend = int(numpy.floor((com[0] * com[2] / self.fx + size[0] / 2.) / com[2]*self.fx))
        ystart = int(numpy.floor((com[1] * com[2] / self.fy - size[1] / 2.) / com[2]*self.fy))
        yend = int(numpy.floor((com[1] * com[2] / self.fy + size[1] / 2.) / com[2]*self.fy))
        return xstart, xend, ystart, yend, zstart, zend

    def getCrop(self, dpt, xstart, xend, ystart, yend, zstart, zend, thresh_z=True):
        """
        Crop patch from image
        :param dpt: depth image to crop from
        :param xstart: start x
        :param xend: end x
        :param ystart: start y
        :param yend: end y
        :param zstart: start z
        :param zend: end z
        :param thresh_z: threshold z values
        :return: cropped image
        """
        if len(dpt.shape) == 2:
            cropped = dpt[max(ystart, 0):min(yend, dpt.shape[0]), max(xstart, 0):min(xend, dpt.shape[1])].copy()
            # add pixels that are out of the image in order to keep aspect ratio
            cropped = numpy.pad(cropped, ((abs(ystart)-max(ystart, 0),
                                           abs(yend)-min(yend, dpt.shape[0])),
                                          (abs(xstart)-max(xstart, 0),
                                           abs(xend)-min(xend, dpt.shape[1]))), mode='constant', constant_values=0)
        elif len(dpt.shape) == 3:
            cropped = dpt[max(ystart, 0):min(yend, dpt.shape[0]), max(xstart, 0):min(xend, dpt.shape[1]), :].copy()
            # add pixels that are out of the image in order to keep aspect ratio
            cropped = numpy.pad(cropped, ((abs(ystart)-max(ystart, 0),
                                           abs(yend)-min(yend, dpt.shape[0])),
                                          (abs(xstart)-max(xstart, 0),
                                           abs(xend)-min(xend, dpt.shape[1])),
                                          (0, 0)), mode='constant', constant_values=0)
        else:
            raise NotImplementedError()

        if thresh_z is True:
            msk1 = numpy.bitwise_and(cropped < zstart, cropped != 0)
            msk2 = numpy.bitwise_and(cropped > zend, cropped != 0)
            cropped[msk1] = zstart
            cropped[msk2] = 0.  # backface is at 0, it is set later
        return cropped

    def resizeCrop(self, crop, sz):
        """
        Resize cropped image
        :param crop: crop
        :param sz: size
        :return: resized image
        """
        if self.resizeMethod == self.RESIZE_CV2_NN:
            rz = cv2.resize(crop, sz, interpolation=cv2.INTER_NEAREST)
        elif self.resizeMethod == self.RESIZE_BILINEAR:
            rz = self.bilinearResize(crop, sz, self.getNDValue())
        elif self.resizeMethod == self.RESIZE_CV2_LINEAR:
            rz = cv2.resize(crop, sz, interpolation=cv2.INTER_LINEAR)
        else:
            raise NotImplementedError("Unknown resize method!")
        return rz

    def applyCrop3D(self, dpt, com, size, dsize, thresh_z=True, background=None):

        # calculate boundaries
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size)

        # crop patch from source
        cropped = self.getCrop(dpt, xstart, xend, ystart, yend, zstart, zend, thresh_z)

        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            sz = (dsize[0], hb * dsize[0] / wb)
        else:
            sz = (wb * dsize[1] / hb, dsize[1])

        # depth resize
        rz = self.resizeCrop(cropped, sz)

        if background is None:
            background = self.getNDValue()  # use background as filler
        ret = numpy.ones(dsize, numpy.float32) * background
        xstart = int(numpy.floor(dsize[0] / 2. - rz.shape[1] / 2.))
        xend = int(xstart + rz.shape[1])
        ystart = int(numpy.floor(dsize[1] / 2. - rz.shape[0] / 2.))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz

        return ret

    def cropArea3D(self, dpt, com=None, dsize=(128, 128), docom=False):
        """
        Crop area of monkey in 3D volumina, scales inverse to the distance of monkey to camera
        :param com: center of mass, in image coordinates (u,v,d), d in mm
        :param dsize: (x,y) extent of the destination size
        :return: cropped image, transformation matrix for joints, CoM in image coordinates
        """
        if len(dsize) != 2:
            raise ValueError("dsize must be a 2D bounding box")

        """
        during training, this parameter will be obtained from the labels,
        during testing, the attention part of the network will provide this as a prediction
        If this is not provided, just calculate the center of mass of the background subtracted image!
        """
        if com is None:
            com = self.calculateCoM(dpt)

        # calculate boundaries
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, self.cube)

        # crop patch from source image
        cropped = self.getCrop(dpt, xstart, xend, ystart, yend, zstart, zend)

        #############
        # a second refinement
        if docom is True:
            com = self.calculateCoM(cropped)
            if numpy.allclose(com, 0.):
                com[2] = cropped[cropped.shape[0]//2, cropped.shape[1]//2]
                if numpy.isclose(com[2], 0):
                    com[2] = 300.
            com[0] += xstart
            com[1] += ystart

            # calculate boundaries
            xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, self.cube)

            # crop patch from source
            cropped = self.getCrop(dpt, xstart, xend, ystart, yend, zstart, zend)
        #############

        wb = (xend - xstart)
        hb = (yend - ystart)
        trans = numpy.asmatrix(numpy.eye(3, dtype=float))
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart
        if wb > hb:
            sz = (dsize[0], hb * dsize[0] / wb)
        else:
            sz = (wb * dsize[1] / hb, dsize[1])

        # print com, sz, cropped.shape, xstart, xend, ystart, yend, hb, wb, zstart, zend
        if cropped.shape[0] > cropped.shape[1]:
            scale = numpy.asmatrix(numpy.eye(3, dtype=float) * sz[1] / float(cropped.shape[0]))
        else:
            scale = numpy.asmatrix(numpy.eye(3, dtype=float) * sz[0] / float(cropped.shape[1]))
        scale[2, 2] = 1
        # depth resize
        rz = self.resizeCrop(cropped, sz)

        ret = numpy.ones(dsize, numpy.float32) * self.maxDepth

        xstart = int(numpy.floor(dsize[0] / 2. - rz.shape[1] / 2.))
        xend = int(xstart + rz.shape[1])
        ystart = int(numpy.floor(dsize[1] / 2. - rz.shape[0] / 2.))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz
        # print rz.shape
        off = numpy.asmatrix(numpy.eye(3, dtype=float))
        off[0, 2] = xstart
        off[1, 2] = ystart

        return ret, off * scale * trans, com

    def transformPoint2D(self,pt, M):

        pt2 = numpy.asmatrix(M.reshape((3, 3))) * numpy.matrix([pt[0], pt[1], 1]).T
        return numpy.array([pt2[0] / pt2[2], pt2[1] / pt2[2]])

    def getRelativeCoordinates(self,jnts_xyz,jnts_uvd,com_uvd,M):
        # convert to 3D coordinates
        com_xyz = self.uvdtoxyz(com_uvd)
        # normalize 3D to center of mass
        rel_jnts_xyz = jnts_xyz - com_xyz

        rel_jnts_uvd = numpy.zeros((jnts_uvd.shape[0], 3), numpy.float32)
        for joint in range(jnts_uvd.shape[0]):
            t = self.transformPoint2D(jnts_uvd[joint], M)
            rel_jnts_uvd[joint, 0] = t[0]
            rel_jnts_uvd[joint, 1] = t[1]
            rel_jnts_uvd[joint, 2] = jnts_uvd[joint, 2]

        return rel_jnts_xyz, rel_jnts_uvd

    def getAbsoluteCoordinates(self,rel_jnts_xyz,com_uvd):
        com_xyz = self.uvdtoxyz(com_uvd)
        jnts_xyz = rel_jnts_xyz + com_xyz
        jnts_uvd = self.xyztouvd(jnts_xyz)
        return jnts_xyz, jnts_uvd