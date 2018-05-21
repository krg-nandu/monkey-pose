from os.path import join as pjoin

class monkeyConfig(object):
    def __init__(self):
        #dir setting
        self.base_dir = '/media/data_cifs/monkey_tracking'
        self.data_dirs ={'train':'monkey_renders_v3', 'test':'simple_monkey_renders_v2'}

        self.real_data_dir = '/media/data_cifs/monkey_tracking/extracted_kinect_depth/201710091108-Freely_Moving_Recording_depth_0/DepthFrame_npys/'

        self.results_dir = '/media/data_cifs/lakshmi/monkey_pose_engine_e2e/results/'
        #self.results_dir = '/home/lakshmi/monkey_pose_results/'
        self.model_output = '/media/data_cifs/lakshmi/monkey_pose_engine_e2e/models/'
        self.model_input = ''
        self.train_summaries = '/media/data_cifs/lakshmi/monkey_pose_engine_e2e/summaries/'
        self.tfrecord_dir = '/media/data_cifs/lakshmi/monkey_pose_engine/tfrecords/'
        self.train_tfrecords = 'traine2e.tfrecords'
        self.val_tfrecords = 'vale2e.tfrecords'
        self.test_tfrecords = 'teste2e.tfrecords'
        self.vgg16_weight_path = pjoin(
            '/media/data_cifs/clicktionary/',
            'pretrained_weights',
            'vgg16.npy')

        # attention window
        self.window_size = {'cube': (800, 800, 1200)}

        # model settings
        self.model_type = 'vgg_regression_model_4fc'
        self.epochs = 300
        self.image_target_size = [128,128,1]
        self.image_orig_size = [424,512,1]
        self.image_max_depth = 10000.
        # self.label_shape = 36
        self.train_batch = 16
        self.val_batch= 16
        self.test_batch = 1
        self.initialize_layers = ['fc6', 'fc7', 'pre_fc8', 'fc8']
        self.fine_tune_layers = ['fc6', 'fc7', 'pre_fc8', 'fc8']
        self.batch_norm = ['conv1','fc1','fc2']
        self.wd_layers = ['fc6', 'fc7', 'pre_fc8']
        self.wd_penalty = 0.005
        self.optimizier = 'adam'
        self.lr = 1e-4  # Tune this -- also try SGD instead of ADAm
        self.hold_lr = self.lr / 2
        self.keep_checkpoints = 100

        # training setting
        self.num_joints = 23
        self.num_dims = 3
        self.num_classes = self.num_joints * self.num_dims  # there are 36 * 3 (x/y/z) joint coors
        self.num_attn_steps = 5000
