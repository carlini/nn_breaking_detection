## pca_detect.py -- break inner-layer pca-based detection
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

import sklearn.decomposition
from sklearn.svm import LinearSVC

from setup_cifar import CIFARModel, CIFAR
from setup_mnist import MNISTModel, MNIST

from nn_robust_attacks.l2_attack import CarliniL2
from fast_gradient_sign import FGS

from keras import backend as K
import pickle

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def pop(model):
    '''Removes a layer instance on top of the layer stack.
    This code is thanks to @joelthchao https://github.com/fchollet/keras/issues/2371#issuecomment-211734276
    '''
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')
    else:
        model.layers.pop()
        if not model.layers:
            model.outputs = []
            model.inbound_nodes = []
            model.outbound_nodes = []
        else:
            model.layers[-1].outbound_nodes = []
            model.outputs = [model.layers[-1].output]
        model.built = False

    return model


def run_hidden_pca(Data, Model, path=None):
    sess = K.get_session()
    K.set_learning_phase(False)

    data = Data()
    model = Model(path)
    model2 = Model(path)

    hidden_layer = pop(model2.model) # once to remove dense(10)
    hidden_layer = pop(hidden_layer) # once to remove ReLU
    train_hidden = hidden_layer.predict(data.test_data)
    #val_hidden = hidden_layer.predict(data.validation_data)
    test_hidden = hidden_layer.predict(data.test_data)
    
    pca = sklearn.decomposition.PCA(n_components=test_hidden.shape[1])
    
    pca.fit(train_hidden)

    #r_val = pca.transform(hidden_layer.predict(data.validation_data))
    r_test = pca.transform(hidden_layer.predict(data.test_data))

    attack = FGS(sess, model, eps=.2)
    #attack = CarliniL2(sess, model, batch_size=100, max_iterations=1000, 
    #                   binary_search_steps=2, targeted=False)

    N = 10000

    test_adv = attack.attack(data.test_data[:N], data.test_labels[:N])

    r_test_adv = pca.transform(hidden_layer.predict(test_adv[:N]))

    print(r_test_adv[0])

    show(test_adv[0])

    #compute_thresholds(r_val, r_val_adv)

    plt.figure(figsize=(4,3))
    plt.xlabel('Component Number')
    plt.ylabel('Mean Absolute Value (log scale)')

    plt.semilogy(range(r_test.shape[1]),np.mean(np.abs(r_test),axis=0))
    plt.semilogy(range(r_test_adv.shape[1]),np.mean(np.abs(r_test_adv),axis=0))
    
    plt.show()

def run_pca(Data, Model, path=None):
    sess = K.get_session()
    K.set_learning_phase(False)

    data = Data()
    model = Model(path)

    shape = (-1, model.num_channels*model.image_size**2)
    
    pca = sklearn.decomposition.PCA(n_components=shape[1])

    pca.fit(data.train_data.reshape(shape))

    print(pca.explained_variance_ratio_)

    r_test = pca.transform(data.test_data.reshape(shape))

    #attack = FGS(sess, model, eps=.3)
    attack = CarliniL2(sess, model, batch_size=100, max_iterations=1000, 
                       binary_search_steps=2, targeted=False,
                       initial_const=10)

    N = 10000

    #test_adv = attack.attack(data.test_data[:N], data.test_labels[:N])
    test_adv = np.load("tmp/outlieradvtest.npy")

    r_test_adv = pca.transform(test_adv[:N].reshape(shape))

    fig = plt.figure(figsize=(4,3))
    fig.subplots_adjust(bottom=0.17,left=.19)
    
    plt.xlabel('Component Number')
    plt.ylabel('Mean Absolute Value (log scale)')

    plt.semilogy(range(r_test.shape[1]),np.mean(np.abs(r_test),axis=0),label='Valid')
    plt.semilogy(range(r_test_adv.shape[1]),np.mean(np.abs(r_test_adv),axis=0), label='Adversarial')

    plt.legend()
    
    pp = PdfPages('/tmp/a.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()
    plt.show()

def run_convolution_pca(Data, Model, path):
    sess = K.get_session()
    K.set_learning_phase(False)

    data = Data()
    model = Model(path)

    """
    for i in range(4):
        model2 = Model(path)

        layer = i
        hidden_layer = model2.model
        while True:
            hidden_layer = pop(hidden_layer)
            if 'conv2d' in str(hidden_layer.outputs):
                if layer == 0: 
                    shape = hidden_layer.outputs[0].get_shape().as_list()
                    shape = tuple([-1]+list(shape[1:]))
                    flatshape = (-1, shape[1]*shape[2]*shape[3])
                    break
                layer -= 1
    
        pca = sklearn.decomposition.PCA(n_components=flatshape[-1])

        print('fitting',flatshape)
        pca.fit(hidden_layer.predict(data.train_data[::5]).reshape(flatshape))
        print('done')
        open("tmp/pcalayer%d.p"%i,"wb").write(pickle.dumps(pca))
    #"""

    pcas = []
    for i in range(2):
        layer = i
        model2 = Model(path)
        hidden_layer = model2.model
        while True:
            hidden_layer = pop(hidden_layer)
            if 'conv2d' in str(hidden_layer.outputs):
                if layer == 0: 
                    shape = hidden_layer.outputs[0].get_shape().as_list()
                    shape = tuple([-1]+list(shape[1:]))
                    flatshape = (-1, shape[1]*shape[2]*shape[3])
                    break
                layer -= 1

        print("shape",shape,flatshape)

        pca = pickle.load(open("tmp/pcalayer%d.p"%i,"rb"))

        print('loaded')
    
        test_adv = np.load("tmp/outlieradvtest.npy")
        hidden_adv = pca.transform(hidden_layer.predict(data.test_data).reshape(flatshape))
        hidden = pca.transform(hidden_layer.predict(test_adv).reshape(flatshape))

        print(hidden_adv.shape)

        np.save("/tmp/hidden_adv", hidden_adv)
        np.save("/tmp/hidden", hidden)
        print('complete')
    
        hidden_adv = np.load("/tmp/hidden_adv.npy").reshape(shape)
        hidden = np.load("/tmp/hidden.npy").reshape(shape)
    
        stdev = np.std(hidden,axis=0)
    
        hidden = np.mean(np.abs(hidden/stdev),axis=(1,2))
        hidden_adv = np.mean(np.abs(hidden_adv/stdev),axis=(1,2))
    
        print('fit model')
        svm = LinearSVC()
        svm.fit(np.concatenate([hidden_adv[:1000],hidden[:1000]],axis=0),[1]*1000+[0]*1000)
        print(np.mean(svm.predict(hidden)))
        print(np.mean(svm.predict(hidden_adv)))
    
        print(hidden.shape)
        


def show(img):
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))

#run_pca(MNIST, MNISTModel, "models/mnist")
#run_hidden_pca(CIFAR, CIFARModel, "models/cifar")

run_convolution_pca(MNIST, MNISTModel, "models/mnist")
