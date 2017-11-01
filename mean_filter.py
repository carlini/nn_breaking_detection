## mean_filter.py -- break the mean filter defense
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import sys
import time
import tensorflow as tf
import numpy as np
import random
import scipy.ndimage

import sklearn.decomposition

from setup_cifar import CIFARModel, CIFAR
from setup_mnist import MNISTModel, MNIST

from nn_robust_attacks.l2_attack import CarliniL2
from fast_gradient_sign import FGS

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D
from keras.optimizers import SGD

import matplotlib
import matplotlib.pyplot as plt


def run_filter(Data, Model, path):
    K.set_learning_phase(False)
    data = Data()
    model = Model(path)
    model2 = Model(path)

    def new_predict(xs):
        print(xs.get_shape())
        if 'mnist' in path:
            xs = tf.nn.conv2d(xs, tf.constant(np.ones((3,3,1,1))/9,dtype=tf.float32),
                              [1,1,1,1], "SAME")
        else:
            xs = tf.nn.conv2d(xs, tf.constant(np.ones((3,3,3,3))/9,dtype=tf.float32),
                              [1,1,1,1], "SAME")
        return model2.model(xs)
    model2.predict = new_predict

    sess = K.get_session()
    #dist 1.45976

    attack = CarliniL2(sess, model2, batch_size=100, max_iterations=3000,
                       binary_search_steps=4, targeted=False, confidence=0,
                       initial_const=10)

    N = 100

    test_adv = attack.attack(data.test_data[:N], data.test_labels[:N])

    print('accuracy of original model',np.mean(np.argmax(sess.run(model.predict(tf.constant(data.test_data,dtype=np.float32))),axis=1)==np.argmax(data.test_labels,axis=1)))
    print('accuracy of blurred model',np.mean(np.argmax(sess.run(model.predict(tf.constant(data.test_data,dtype=np.float32))),axis=1)==np.argmax(data.test_labels,axis=1)))

    print('dist',np.mean(np.sum((test_adv-data.test_data[:N])**2,axis=(1,2,3))**.5))

    #it = np.argmax(sess.run(model.predict(tf.constant(test_adv))),axis=1)
    #print('success of unblured',np.mean(it==np.argmax(data.test_labels,axis=1)[:N]))
    it = np.argmax(sess.run(model2.predict(tf.constant(test_adv))),axis=1)
    print('success of blured',np.mean(it==np.argmax(data.test_labels,axis=1)[:N]))
    

run_filter(MNIST, MNISTModel, "models/mnist")
run_filter(CIFAR, CIFARModel, "models/cifar")
