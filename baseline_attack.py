## baseline_attack.py -- perform a baseline attack on MNIST and CIFAR models
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import time
import os
import tensorflow as tf
import numpy as np
import random
import time

from setup_cifar import CIFARModel, CIFAR
from setup_mnist import MNISTModel, MNIST

from utils import *

sys.path.append("../..")
from nn_robust_attacks.l2_attack import CarliniL2

from keras import backend as K

def run(Data, Model, path):
    sess = K.get_session()
    K.set_learning_phase(False)
    data, model = Data(), Model(path)

    if Data == MNIST:
        attack = CarliniL2(sess, model, batch_size=100, max_iterations=2000,
                           binary_search_steps=5, initial_const=1., learning_rate=1e-1,
                           targeted=False)
    else:
        attack = CarliniL2(sess, model, batch_size=100, max_iterations=200,
                           binary_search_steps=3, initial_const=.01, learning_rate=1e-2,
                           targeted=True, confidence=2)

    now = time.time()

    for name,X,y in [["test",data.test_data, data.test_labels]]:
        print("OKAY",name)
        for k in range(0,len(y),5000):
            #if os.path.exists("tmp/"+path.split("/")[1]+"."+name+".adv.X."+str(k)+".npy"):
            #    print('skip',k)
            #    continue
            now = time.time()
            adv = attack.attack(X[k:k+100], y[k:k+100])
            #print('time',time.time()-now)
            #print('accuracy',np.mean(np.argmax(model.model.predict(adv),axis=1)==np.argmax(y[k:k+5000],axis=1)))
            #print('mean distortion',np.mean(np.sum((adv-X[k:k+5000])**2,axis=(1,2,3))**.5))
            np.save("/tmp/"+path.split("/")[1]+"."+name+".adv.X."+str(k),adv)
    

run(CIFAR, CIFARModel, "models/cifar")
#run(MNIST, MNISTModel, "models/mnist")
