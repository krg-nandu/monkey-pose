import tensorflow as tf
import numpy as numpy
import argparse
import os
import cPickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

def getMeanError_np(labels,results):
    """
    get average error over all joints, averaged over sequence
    :return: mean error
    """
    return numpy.nanmean(numpy.nanmean(numpy.sqrt(numpy.square(labels - results).sum(axis=2)), axis=1))


def getMaxError_np(labels,results):
    """
    get max error over all joints
    :return: maximum error
    """

    return numpy.nanmax(numpy.sqrt(numpy.square(labels - results).sum(axis=2)))

def getMean_np(labels,results):

    return numpy.nanmean(numpy.sqrt(numpy.square(labels - results).sum(axis=1)), axis=0)

def getMeanError_train(labels,results):
    """
    get average error over all joints, averaged over sequence
    :return: mean error
    """
    assert(labels.shape==results.shape)
    return tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(labels-results), 2)),1))

def getMeanError(labels,results):
    """
    get average error over all joints, averaged over sequence
    :return: mean error
    """
    assert(labels.shape==results.shape)
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(labels-results), 1)),0)

def getMeanErrors_N(labels,results):
    """
    get average error over all joints, averaged over sequence
    :return: mean error
    """
    assert(labels.shape==results.shape)
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(labels-results), 2)),1)

def getMaxError(labels,results):
    """
    get max error over all joints
    :return: maximum error
    """
    assert(labels.shape==results.shape)
    return tf.reduce_max(tf.sqrt(tf.reduce_sum(tf.square(labels-results), 2)))


def getNumFramesWithinMaxDist(labels,results, dist):
    """
    calculate the number of frames where the maximum difference of a joint is within dist mm
    :param dist: distance between joint and GT
    :return: number of frames
    """
    return (numpy.nanmax(numpy.sqrt(numpy.square(labels - results).sum(axis=2)), axis=1) <= dist).sum()


def getNumFramesWithinMeanDist(labels,results, dist):
    """
    calculate the number of frames where the mean difference over all joints of a hand are within dist mm
    :param dist: distance between joint and GT
    :return: number of frames
    """
    return (numpy.nanmean(numpy.sqrt(numpy.square(labels - results).sum(axis=2)), axis=1) <= dist).sum()


def getJointMeanError(labels,results,jointID):
    """
    get error of one joint, averaged over sequence
    :param jointID: joint ID
    :return: mean joint error
    """

    return numpy.nanmean(numpy.sqrt(numpy.square(labels[:, jointID, :] - results[:, jointID, :]).sum(axis=1)))





def main(model,Joints):
    if Joints == 'all':
        jointNames = ('P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'M1', 'M2', 'M3',
                      'M4', 'M5', 'M6', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'T1', 'T2', 'T3', 'T4',
                      'T5', 'C1', 'C2', 'C3','C4','C5','C6','C7')
    else:
        jointNames = ('P1', 'P2', 'R1', 'R2', 'M1', 'M2', 'I1', 'I2', 'T1', 'T2', 'T3', 'W1', 'W2', 'C')
    # imagedir = 'results_xyz/{}/results'.format(model)
    # files=os.listdir(imagedir)
    # img1=mpimg.imread(os.path.join(imagedir,files[0]))
    # img2=mpimg.imread(os.path.join(imagedir,files[1]))
    # plt.subplot(1,2,1)
    # plt.imshow(img1)
    # plt.subplot(1,2,2)
    # plt.imshow(img2)
    # plt.show()


    # numimg=len(files)
    # numrow=math.ceil(numimg/5)
    # num=1
    # manyfig = plt.figure()
    # for img in files:
    #     image = mpimg.imread(os.path.join(imagedir,img))
    #     plt.subplot(numrow,5,num)
    #     plt.imshow(image)
    #     plt.axis('off')
    #     num +=1
    # plt.subplots_adjust(wspace=0.01, hspace=0.01)
    # plt.suptitle(model)
    # #manyfig.savefig("results/{}/combination.png".format(model),bbox_inches='tight')
    # manyfig.savefig('results/{}/combination.eps'.format(model), format='eps', dpi=1000)
    # plt.close(manyfig)

    data = 'results_com/{}/cnn_result_cache.pkl'.format(model,model)
    if os.path.isfile(data) is False:
        print('{} does not exits!'.format(data))
        return 0
    print("Loading cache data from {}".format(data))
    f = open(data, 'rb')
    (joint_labels, joint_results) = cPickle.load(f)
    f.close()
    # the mean error and max error
    labels = numpy.asarray(joint_labels)
    results= numpy.asarray(joint_results)
    mean_error = getMeanError_np(labels,results)
    max_error = getMaxError_np(labels,results)
    print("There are {} test data".format(results.shape[0]))
    print ("The mean error is {} mm. The max error is {} mm".format(mean_error,max_error))

    # plot number of frames within max distance
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([getNumFramesWithinMaxDist(labels,results,j) / float(labels.shape[0]) * 100. for j in
             range(0, 100)],
            label=model, c='blue', linestyle='-')
    plt.suptitle("Max error of {} is :{} mm".format(model,max_error))
    plt.xlabel('Max Distance threshold / mm')
    plt.ylabel('Fraction of frames within distance / %')
    plt.ylim([0.0, 100.0])
    ax.grid(True)
    # Put a legend below current axis
    handles, axlabels = ax.get_legend_handles_labels()
    # lgd = ax.legend(handles, labels, loc='lower right', ncol=1) #, bbox_to_anchor=(0.5,-0.1)
    ax.legend(handles, axlabels, loc='upper center', bbox_to_anchor=(0.7, 0.1), ncol=3)
    fig.savefig('results_com/{}/fraction_max_error.png'.format(model))
    plt.close(fig)

    # plot number of frames within mean distance
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([getNumFramesWithinMeanDist(labels,results,j) / float(labels.shape[0]) * 100. for j in
             range(0, 100)],
            label=model, c='blue', linestyle='-')
    plt.suptitle("Mean error of {} is :{} mm".format(model,mean_error))
    plt.xlabel('Mean Distance threshold / mm')
    plt.ylabel('Fraction of frames within distance / %')
    plt.ylim([0.0, 100.0])
    ax.grid(True)
    # Put a legend below current axis
    handles, axlabels = ax.get_legend_handles_labels()
    # lgd = ax.legend(handles, labels, loc='lower right', ncol=1) #, bbox_to_anchor=(0.5,-0.1)
    ax.legend(handles, axlabels, loc='upper center', bbox_to_anchor=(0.7, 0.1), ncol=3)
    fig.savefig('results_com/{}/fraction_mean_error.png'.format(model))
    plt.close(fig)

    # plot mean error for each joint
    ind = numpy.arange(results.shape[1] + 1)  # the x locations for the groups, +1 for mean
    width = 0.67
    fig, ax = plt.subplots()
    mean = [getJointMeanError(labels,results,j) for j in range(results.shape[1])]
    mean.append(mean_error)
    ax.bar(ind, numpy.array(mean), width, label=model, color="blue")  # , yerr=std)
    bs_idx = 1
    ax.set_xticks(ind + width)
    ll = list(jointNames)
    ll.append('Avg')
    label = tuple(ll)
    ax.set_xticklabels(label)
    plt.suptitle('{}'.format(model))
    plt.ylabel('Mean error of joint / mm')
    # plt.ylim([0.0,50.0])
    # Put a legend below current axis
    handles, axlabels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, axlabels)
    plt.show()
    fig.savefig('results_com/{}/Joint_mean_error.png'.format(model),bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(fig)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str,default="cnn",help='the model name')
    parser.add_argument('--Joints',type=str,default = "all",help="use all joints or not")#no/all
    args = parser.parse_args()
    main(**vars(args))