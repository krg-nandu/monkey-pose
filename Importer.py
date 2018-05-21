import os
import progressbar as pb
import cPickle
import numpy as np
from basetype import ICVLFrame,LabelledFrame, NamedImgSequence
from handdetector import MonkeyDetector
from transformations import transformPoint2D
import scipy.io
from PIL import Image
from check_fun import showAnnotatedDepth, showdepth,showImageLable,trans3DToImg,trans3DsToImg
import glob
import tqdm

class Importer(object):
    def __init__(self,fx,fy,ux,uy):
        """""
        Initialize object
        :param fx: focal length in x direction
        :param fy: focal length in y direction
        :param ux: principal point in x direction
        :param uy: principal point in y direction
        """""
        self.fx = fx
        self.fy = fy
        self.ux = ux
        self.uy = uy

    def jointsImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])
        return ret

    def jointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        ret[0] = (self.ux - sample[0]) * sample[2] / (-self.fx)
        ret[1] = (sample[1]-self.uy) * sample[2] / (-self.fy)
        ret[2] = -sample[2]
        return ret

    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.joint3DToImg(sample[i])
        return ret

    def joint3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3,),np.float32)
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = self.ux - sample[0] / sample[2] * self.fx
        ret[1] = sample[1] / sample[2] * self.fy + self.uy
        ret[2] = -sample[2]
        return ret

class MonkeyRendersImporter(Importer):
    def __init__(self,path,useCache = True,cacheDir = '/media/data_cifs/lakshmi/cache'):

        # also setting the focal length etc. here
        super(MonkeyRendersImporter,self).__init__(365.456,365.456,256.,212.)

        self.path = path
        self.useCache = useCache
        self.cacheDir = cacheDir
        self.numJoints = 36
        self.scales = {'train': 1., 'test_1': 1., 'test_2': 0.83, 'test': 1., 'train_synth': 1.,
                       'test_synth_1': 1., 'test_synth_2': 0.83, 'test_synth': 1.}
        self.restrictedJoints = [100, 97, 57, 60, 79, 61, 80, 62, 81, 69, 91, 71, 93, 38, 19, 39, 20, 40, 21, 41, 22, 50, 31]

    def loadDepthMap(self,filename):
        """
        Read a depth-map
        :param filename: file name to load
        :return: image data of depth image
        """
        img = Image.open(filename)
        imgdata = np.asarray(img, np.float32)
        return imgdata


    def loadSequence(self,seqName, cfg=None, Nmax = float('inf'),shuffle = False, rng = None, docom = False,allJoints=False):

        config = {'cube':(800,800,1200)}
        config['cube'] = [s*self.scales[seqName] for s in config['cube']]

        if Nmax is float('inf'):
            pickleCache = '{}/{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__, seqName,allJoints)
        else:
            pickleCache = '{}/{}_{}_{}_cache_{}.pkl'.format(self.cacheDir, self.__class__.__name__, seqName, allJoints,Nmax)
        print(pickleCache)

        if self.useCache:
            if os.path.isfile(pickleCache):
                print("Loading cache data from {}".format(pickleCache))
                f = open(pickleCache,'rb')
                (seqName,data,config) = cPickle.load(f)
                f.close()

                #shuffle data
                if shuffle and rng is not None:
                    print("shuffling")
                    rng.shuffle(data)
                if not(np.isinf(Nmax)):
                    return NamedImgSequence(seqName,data[0:Nmax],config)
                else:
                    return NamedImgSequence(seqName,data,config)

        #load the dataset
        objdir = '{}/{}/'.format(cfg.base_dir,cfg.data_dirs[seqName])
        names = glob.glob(os.path.join(objdir,'*.txt'))

        joints3D = np.empty([len(names),cfg.num_joints, cfg.num_dims])
        joints2D = np.empty([len(names),cfg.num_joints, cfg.num_dims])

        # load the groundtruth data here
        cnt = 0
        for name in tqdm.tqdm(names,total=len(names)):
            all_jnts = np.loadtxt(name)
            # get the proper subset
            joints3D[cnt] = all_jnts[self.restrictedJoints]
            # get 2D projections
            joints2D[cnt] = self.joints3DToImg(joints3D[cnt])
            cnt+=1

        if allJoints:
            eval_idxs = np.arange(cfg.num_joints)
        else:
            eval_idxs = self.restrictedJoints
        self.numJoints = len(eval_idxs)

        txt= 'Loading {}'.format(seqName)
        pbar = pb.ProgressBar(maxval=joints3D.shape[0],widgets=[txt,pb.Percentage(),pb.Bar()])
        pbar.start()

        data = []
        i=0
        for line in range(joints3D.shape[0]):
            imgid = names[line].split('/')[-1].split('.')[0].split('_')[-1]
            dptFileName = os.path.join(cfg.base_dir,
                                       cfg.data_dirs[seqName],
                                       'depth_%s.png'%imgid)

            if not os.path.isfile(dptFileName):
                print("File {} does not exist!").format(dptFileName)
                i += 1
                continue
            dpt = self.loadDepthMap(dptFileName)

            gtorig = joints2D[line,eval_idxs,:]
            gt3Dorig = joints3D[line,eval_idxs,:]

            data.append(LabelledFrame(dpt.astype(np.float32),gtorig,gt3Dorig,dptFileName,''))
            pbar.update(i)
            i+=1

            #early stop
            if len(data)>=Nmax:
                break

        pbar.finish()
        print("loaded {} samples.".format(len(data)))

        if self.useCache:
            print("Save cache data to {}".format(pickleCache))
            f = open(pickleCache,'wb')
            cPickle.dump((seqName,data,config), f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

        #shuffle data
        if shuffle and rng is not None:
            print("shuffling")
            rng.shuffle(data)
        
        return NamedImgSequence(seqName,data,config)

    def jntsXYZtoUVD(self, jnts_xyz):
        jnts_uvd = np.zeros((jnts_xyz.shape[0], 3), np.float32)
        for j in range(jnts_xyz.shape[0]):
            jnts_uvd[j] = self.joint3DToImg(jnts_xyz[j])
        return jnts_uvd
