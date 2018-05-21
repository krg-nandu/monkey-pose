from config import monkeyConfig
from Datareader import Datareader
from train_models import test_input_full
from train_cnn_networks import train_model, test_model, eval_model_on_real_data

__DEBUG__ = False
__TFREC__ = True
__TRAIN__ = True

def main():
    config = monkeyConfig()
    if __TFREC__:
        seqconfig = Datareader(config)
    else:
        seqconfig = config.window_size

    if __DEBUG__:
        # this is just a function to check if the tfrecords were written correctly
        test_input_full(config,seqconfig)

    if __TRAIN__:
        train_model(config,seqconfig)
        test_model(config,seqconfig)
        # use this only if we want to eval on an unlabelled video, and not from a tf record
        eval_model_on_real_data(config,seqconfig)

    return 0

if __name__ == '__main__':
    main()
