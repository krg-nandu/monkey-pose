'''
A set of utility functions for manipulation of annotated hand data
'''
import numpy as np
import cv2

'''
These are camera parameter for the NYU Hand dataset.
TODO: Import these values from the dataset class
'''
fx = 588.03
fy = 587.07
ux = 320.
uy = 240.


def resizeCrop(crop, sz):
    rz = cv2.resize(crop, sz, interpolation=cv2.INTER_NEAREST)
    return rz

def comToBounds(com, size):
    """
    Calculate boundaries, project to 3D, then add offset and backproject to 2D (ux, uy are canceled)
    :param com: center of mass, in image coordinates (x,y,z), z in mm
    :param size: (x,y,z) extent of the source crop volume in mm
    :return: xstart, xend, ystart, yend, zstart, zend
    """
    zstart = com[2] - size[2] / 2.
    zend = com[2] + size[2] / 2.
    xstart = int(np.floor((com[0] * com[2] / fx - size[0] / 2.) / com[2]*fx))
    xend = int(np.floor((com[0] * com[2] / fx + size[0] / 2.) / com[2]*fx))
    ystart = int(np.floor((com[1] * com[2] / fy - size[1] / 2.) / com[2]*fy))
    yend = int(np.floor((com[1] * com[2] / fy + size[1] / 2.) / com[2]*fy))
    return xstart, xend, ystart, yend, zstart, zend

def jointsImgTo3D(sample):
    '''
    U-V-D coordinate system to X-Y-Z coordinate system
    :param sample: uvd of a set of joints
    :return: xyz of the corresponding joints
    '''
    ret = np.zeros((sample.shape[0], 3), np.float32)
    for i in range(sample.shape[0]):
        ret[i] = jointImgTo3D(sample[i])
    return ret


def jointImgTo3D(sample):
    '''
    U-V-D coordinate system to X-Y-Z coordinate system
    :param sample: uvd of a single joint
    :return: xyz of the joint
    '''
    ret = np.zeros((3,), np.float32)
    # convert to metric using f, see Thomson et al.
    ret[0] = (sample[0] - ux) * sample[2] / fx
    ret[1] = (uy - sample[1]) * sample[2] / fy
    ret[2] = sample[2]
    return ret


def joints3DToImg(sample):
    '''
    X-Y-Z coordinate system to U-V-D coordinate system
    :param sample: xyz of a group of joints
    :return: uvd of the corresponding joints
    '''
    ret = np.zeros((sample.shape[0], 3), np.float32)
    for i in range(sample.shape[0]):
        ret[i] = joint3DToImg(sample[i])
    return ret


def joint3DToImg(sample):
    '''
    X-Y-Z coordinate system to U-V-D coordinate system
    :param sample: xyz of a single joint
    :return: uvd of the specified joint
    '''
    ret = np.zeros((3,), np.float32)
    # convert to metric using f, see Thomson et.al.
    if sample[2] == 0.:
        ret[0] = ux
        ret[1] = uy
        return ret
    ret[0] = sample[0] / sample[2] * fx + ux
    ret[1] = uy - sample[1] / sample[2] * fy
    ret[2] = sample[2]
    return ret

def rotatePoint2D(p1, center, angle):
        """
        Rotate a point in 2D around center
        :param p1: point in 2D (u,v,d)
        :param center: 2D center of rotation
        :param angle: angle in deg
        :return: rotated point
        """
        alpha = angle * np.pi / 180.
        pp = p1.copy()
        pp[0:2] -= center[0:2]
        pr = np.zeros_like(pp)
        pr[0] = pp[0] * np.cos(alpha) - pp[1] * np.sin(alpha)
        pr[1] = pp[0] * np.sin(alpha) + pp[1] * np.cos(alpha)
        pr[2] = pp[2]
        ps = pr
        ps[0:2] += center[0:2]
        return ps


def transHand(dpt, cube, comXYZ, joints3D, M, pad_value=1.):
    """
    Adjust already cropped image such that a moving CoM normalization is simulated
    :param dpt: cropped depth image with different CoM
    :param cube: metric cube of size (sx,sy,sz)
    :param com: original center of mass, in ***3D*** coordinates (x,y,z)
    :param off: offset to center of mass (dx,dy,dz) in image coordinates
    :param joints3D: 3D joint coordinates, cropped to old CoM
    :param pad_value: value of padding
    :return: adjusted image, new 3D joint coordinates, new center of mass in image coordinates
    """
    sigma_com = 5.
    off = np.random.randn(3) * sigma_com
    joints3D = joints3D * (cube[2] / 2.0)
    # get the uvd coordinates of the center of mass
    com = joint3DToImg(comXYZ)

    # if offset is 0, nothing to do
    if np.allclose(off, 0.):
        joints3D = np.clip(np.asarray(joints3D, dtype='float32') / (cube[2] / 2.0), -1, 1)
        return dpt, joints3D, comXYZ, M

    # new method, correction for shift due to z change:
    # [(x-u)*(z-v)/f - cube/2] * f/(z-v) = (x-u) - cube/2*f/(z-v)    with u,v random offsets
    new_com = com + off
    co = np.zeros((3,), dtype='int')

    # check for 1/0.
    if not np.allclose(com[2], 0.):
        # calculate offsets
        co[0] = off[0] + cube[0] / 2. * fx / com[2] * (off[2] / (com[2] + off[2]))
        co[1] = off[1] + cube[1] / 2. * fy / com[2] * (off[2] / (com[2] + off[2]))
        co[2] = off[2]

        # offset scale
        xstart, xend, _, _, _, _ = comToBounds(com, cube)
        scoff = dpt.shape[0] / float(xend - xstart)
        xstart2, xend2, ystart2, yend2, _, _ = comToBounds(new_com, cube)
        scoff2 = dpt.shape[0] / float(xend2 - xstart2)

        co = np.round(co).astype('int')

        # normalization
        scalePreX = 1. / scoff
        scalePreY = 1. / scoff
        scalePostX = scoff2
        scalePostY = scoff2
    else:
        # calculate offsets
        co[0] = off[0]
        co[1] = off[1]
        co[2] = off[2]

        co = np.round(co).astype('int')

        # normalization
        scalePreX = 1.
        scalePreY = 1.
        scalePostX = 1.
        scalePostY = 1.

    # print com, scoff, xend, xstart, dpt.shape, new_com, scalePreX, scalePostX, scalePreY, scalePostY

    new_dpt = np.asarray(dpt.copy(), 'float32')
    new_dpt = resizeCrop(new_dpt, (int(round(scalePreX * new_dpt.shape[1])),
                                        int(round(scalePreY * new_dpt.shape[0]))))

    # shift by padding
    if co[0] > 0:
        new_dpt = np.pad(new_dpt, ((0, 0), (0, co[0])), mode='constant', constant_values=pad_value)[:, co[0]:]
    elif co[0] == 0:
        pass
    else:
        new_dpt = np.pad(new_dpt, ((0, 0), (-co[0], 0)), mode='constant', constant_values=pad_value)[:, :co[0]]

    if co[1] > 0:
        new_dpt = np.pad(new_dpt, ((0, co[1]), (0, 0)), mode='constant', constant_values=pad_value)[co[1]:, :]
    elif co[1] == 0:
        pass
    else:
        new_dpt = np.pad(new_dpt, ((-co[1], 0), (0, 0)), mode='constant', constant_values=pad_value)[:co[1], :]

    rz = resizeCrop(new_dpt, (int(round(scalePostX * new_dpt.shape[1])),
                                   int(round(scalePostY * new_dpt.shape[0]))))
    new_dpt = np.zeros_like(dpt)
    new_dpt[0:rz.shape[0], 0:rz.shape[1],0] = rz[0:128, 0:128]


    # adjust joint positions to new CoM
    new_joints3D = joints3D + jointImgTo3D(com) - jointImgTo3D(new_com)

    # recalculate transformation matrix
    trans = np.eye(3)
    trans[0, 2] = -xstart2
    trans[1, 2] = -ystart2
    scale = np.eye(3) * scalePostX
    scale[2, 2] = 1

    off = np.eye(3)
    off[0, 2] = 0
    off[1, 2] = 0
    new_joints3D = np.clip(np.asarray(new_joints3D, dtype='float32') / (cube[2] / 2.0), -1, 1)

    return new_dpt, new_joints3D, np.asarray(new_com,dtype='float32'), np.asarray(np.dot(off, np.dot(scale, trans)),dtype='float32')

def scaleHand(dpt, cube, comXYZ, joints3D, M, pad_value=1):
        """
        Virtually scale the hand by applying different cube
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in 3d coordinates (x,y,z)
        :param sc: scale factor for cube
        :param joints3D: 3D joint coordinates, cropped to old CoM
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, new center of mass in image coordinates
        """
        sigma_sc = 0.03
        sc = abs(1. + np.random.randn() * sigma_sc)

        joints3D = joints3D * (cube[2] / 2.0)
        # get the uvd coordinates of the center of mass
        com = joint3DToImg(comXYZ)

        # if scale is 1, nothing to do
        if np.allclose(sc, 1.):
            joints3D = np.clip(np.asarray(joints3D, dtype='float32') / (cube[2] / 2.0), -1, 1)
            return dpt, joints3D, cube, M

        new_cube = [s*sc for s in cube]

        # check for 1/0.
        if not np.allclose(com[2], 0.):
            # scale to original size
            xstart, xend, ystart, yend, _, _ = comToBounds(com, cube)
            scoff = dpt.shape[0] / float(xend-xstart)
            xstart2, xend2, ystart2, yend2, _, _ = comToBounds(com, new_cube)
            scoff2 = dpt.shape[0] / float(xend2-xstart2)

            # normalization
            scalePreX = 1./scoff
            scalePreY = 1./scoff
            scalePostX = scoff2
            scalePostY = scoff2
        else:
            xstart = xstart2 = 0
            ystart = ystart2 = 0
            xend = xend2 = 0
            yend = yend2 = 0

            # normalization
            scalePreX = 1.
            scalePreY = 1.
            scalePostX = 1.
            scalePostY = 1.

        # print com, scoff, xend, xstart, dpt.shape, com2, scalePreX, scalePostX, scalePreY, scalePostY
        new_dpt = np.asarray(dpt.copy(), 'float32')
        #new_dpt = resizeCrop(new_dpt, (int(round(scalePreX*new_dpt.shape[1])),
        #                                    int(round(scalePreY*new_dpt.shape[0]))))
        new_dpt = cv2.resize(new_dpt, (int(round(scalePreX*new_dpt.shape[1])),
                                            int(round(scalePreY*new_dpt.shape[0]))), interpolation=cv2.INTER_NEAREST)
        # enlarge cube by padding, or cropping image
        xdelta = abs(xstart-xstart2)
        ydelta = abs(ystart-ystart2)
        if sc > 1.:
            new_dpt = np.pad(new_dpt, ((xdelta, xdelta), (ydelta, ydelta)), mode='constant', constant_values=pad_value)
        elif np.allclose(sc, 1.):
            pass
        else:
            new_dpt = new_dpt[xdelta:-(xdelta+1), ydelta:-(ydelta+1)]

        rz = resizeCrop(new_dpt, (int(round(scalePostX*new_dpt.shape[1])),
                                       int(round(scalePostY*new_dpt.shape[0]))))
        new_dpt = np.zeros_like(dpt)
        new_dpt[0:rz.shape[0], 0:rz.shape[1],0] = rz[0:128, 0:128]

        new_joints3D = joints3D

        # recalculate transformation matrix
        trans = np.eye(3)
        trans[0, 2] = -xstart2
        trans[1, 2] = -ystart2
        scale = np.eye(3) * scalePostX
        scale[2, 2] = 1

        off = np.eye(3)
        off[0, 2] = 0
        off[1, 2] = 0

        new_joints3D = np.clip(np.asarray(new_joints3D, dtype='float32') / (cube[2] / 2.0), -1, 1)

        return new_dpt, new_joints3D, np.asarray(new_cube,dtype='float32'), np.asarray(np.dot(off, np.dot(scale, trans)),dtype='float32')


'''
TODO: Cube should be a parameter
currenty we assume cube to be [300,300,300]
'''
def rotateHand(image, com, joints3D, dims):
    """
    Please do note that this function is different from the namesake in handdetector.py!
    Rotate hand virtually in the image plane by a given angle
    :param com: original center of mass, in **3D** coordinates (x,y,z)
    :param rot: rotation angle in deg
    :param joints3D: original joint coordinates, in 3D coordinates (x,y,z)
    :param dims: dimensions of the image
    :return: new 3D joint coordinates
    """
    rot = np.random.uniform(0, 360)

    cubez = 300
    # rescale joints
    joints3D = joints3D*(cubez/2.0)
    # get the uvd coordinates of the center of mass
    comUVD = joint3DToImg(com)
    # if rot is 0, nothing to do
    if np.allclose(rot, 0.):
        joints3D = np.clip(np.asarray(joints3D, dtype='float32') / (cubez/2.0), -1, 1)
        return image, joints3D

    # For a non-zero rotation!
    rot = np.mod(rot, 360)
    # get the 2D rotation matrix
    M = cv2.getRotationMatrix2D((dims[1] // 2, dims[0] // 2), -rot, 1)

    image = cv2.warpAffine(image, M, (dims[1], dims[0]), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=1)

    # translate to COM and project on to the image
    joint_2D = joints3DToImg(joints3D + com)
    # rotate every joint in plane
    data_2D = np.zeros_like(joint_2D)
    for k in xrange(data_2D.shape[0]):
        data_2D[k] = rotatePoint2D(joint_2D[k], comUVD[0:2], rot)
    # inverse translate
    new_joints3D = (jointsImgTo3D(data_2D) - com)
    # clip the limits of the joints
    new_joints3D = np.clip(np.asarray(new_joints3D, dtype='float32') / (cubez / 2.0), -1, 1)

    return image,new_joints3D

def augment_sample(label,image,com3D,M,dims):
    config = {'cube': (300, 300, 300)}
    # possible augmentation modes
    aug_modes = ['rot', 'scale', 'trans']
    # pick an augmentation method
    mode = np.random.randint(0, len(aug_modes))
    # choose to skip augmenting once in a while
    if (np.random.randint(0, 64) == 0):
        return label, image, com3D, M

    if aug_modes[mode] == 'rot':
        image, label = rotateHand(image, com3D, label, dims)
    elif aug_modes[mode] == 'scale':
        image, label, _, _ = scaleHand(image.astype('float32'), config['cube'], com3D, label, M)
    elif aug_modes[mode] == 'trans':
        image, label, _, _ = transHand(image.astype('float32'),config['cube'],com3D,label,M)
    else:
        print('Not recognized. Augmentation skipped!')

    return label,image,com3D,M