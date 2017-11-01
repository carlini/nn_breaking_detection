# Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>
# All rights reserved.

import sys
import time
import tensorflow as tf
import numpy as np
import random
import re

import sklearn.decomposition

from setup_cifar import CIFARModel, CIFAR
from setup_mnist import MNISTModel, MNIST

from nn_robust_attacks.l2_attack import CarliniL2
from fast_gradient_sign import FGS

from keras import backend as K

#import matplotlib
#import matplotlib.pyplot as plt

from kernel_two_sample_test import *
from sklearn.metrics import pairwise_distances


def run_test(Data, Model, path):
    sess = K.get_session()
    K.set_learning_phase(False)
    data = Data()
    model = Model(path)

    N = 1000
    X = data.train_data[np.random.choice(np.arange(len(data.train_data)), N, replace=False)].reshape((N,-1))
    #Y = data.train_data[np.random.choice(np.arange(len(data.train_data)), N, replace=False)].reshape((N,-1))
    Y = data.test_data[np.random.choice(np.arange(len(data.test_data)), N, replace=False)].reshape((N,-1))

    #attack = FGS(sess, model, N, .275)
    attack = CarliniL2(sess, model, batch_size=100, binary_search_steps=2, initial_const=1,  targeted=False, max_iterations=500)
    

    idx = np.random.choice(np.arange(len(data.test_data)), N, replace=False)
    Y = attack.attack(data.test_data[idx], data.test_labels[idx]).reshape((N,-1))

    
    iterations = 1000
    
    sigma2 = 100
    mmd2u, mmd2u_null, p_value = kernel_two_sample_test(X, Y, iterations=iterations,
                                                        kernel_function='rbf',
                                                        gamma=1.0/sigma2,
                                                        verbose=True)

#run_test(MNIST, MNISTModel, "models/mnist")
run_test(CIFAR, CIFARModel, "models/cifar")
