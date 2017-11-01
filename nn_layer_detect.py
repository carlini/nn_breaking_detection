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
from resnet import ResnetBuilder

from nn_robust_attacks.l2_attack import CarliniL2
from fast_gradient_sign import FGS

from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, Input, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Lambda
from keras.optimizers import SGD, Adam

#import matplotlib
#import matplotlib.pyplot as plt

class RobustModel:
    def __init__(self, model_with_detector):
        self.model_with_detector = model_with_detector
        self.num_channels = 3
        self.num_labels = 11
        self.image_size = 32
        
    def predict(self, data):
        print('here',data)
        predicted, is_bad = self.model_with_detector(data)

        padded = tf.pad(predicted, [[0, 0], [0, 1]], "CONSTANT")
        maximum = tf.reshape(tf.reduce_max(padded,axis=1),(-1,1))
        padded = padded + 2*maximum*tf.pad(1+is_bad, [[0, 0], [self.num_labels-1, 0]], "CONSTANT")
        print(padded)
        
        return padded

def train(sess, model, train_data, actual_train_labels, train_labels, file_name, 
          LEARNING_RATE=0.01, MOMENTUM=0.9, OPTIMIZER='sgd', NUM_EPOCHS=20, BATCH_SIZE=256):
    print('train')


    # there appears to be a bug in Keras that batchnorm will still update
    # even if the layer isn't trainable.
    # this is an ugly hack to fix that.
    for layer in model.layers:
        if not layer.trainable:
            layer.updates = []
            layer.params = []
            #print([x.name for x in layer.weights])
            
    if OPTIMIZER == 'sgd':
        OPTIMIZER = SGD(lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=False)
    elif OPTIMIZER == 'adam':
        OPTIMIZER = Adam(lr=LEARNING_RATE)

    model.compile(loss='binary_crossentropy',
                  optimizer=OPTIMIZER,
                  metrics=['accuracy'])

    if True:
        print(file_name, OPTIMIZER, LEARNING_RATE, MOMENTUM, NUM_EPOCHS, BATCH_SIZE)

        best_acc = 0
        

        idxs = list(range(len(train_data)))
        random.shuffle(idxs)

        train_data = train_data[idxs]
        train_labels = train_labels[idxs]
        actual_train_labels = actual_train_labels[idxs]

        for epoch in range(NUM_EPOCHS):
            weights = model.get_weights()

            model.fit(train_data[1000:], [actual_train_labels[1000:], train_labels[1000:]],
                      batch_size=BATCH_SIZE*4,
                      epochs=1)

            a = model.predict(train_data[:1000])[1].flatten()>.5
            b = train_labels[:1000].flatten()
            acc=np.mean(a == b)
            print('acc',acc)
            if acc > best_acc:
                print('improved')
                best_acc = acc
                model.save_weights(file_name)

    else:
        model.load_weights(file_name)

    return model

class Wrap:
    image_size = 32
    num_channels = 3
    num_labels = 10

    def __init__(self, model):
        self.model = model

    def predict(self, xs):
        return self.model(xs)

def train_nn_detection(Data, Model, path=None, num=0):
    data = Data()

    sess = K.get_session()
    K.set_learning_phase(False)

    """#Uncomment this ugly block of code to set up the weights of the detector

    with tf.variable_scope("model"):
        model = ResnetBuilder.build_resnet_32((3, 32, 32), 10)
        model.load_weights("models/cifar-resnet")
    with tf.variable_scope("model_with_detector"):
        model_with_detector = ResnetBuilder.build_resnet_32((3, 32, 32), 10, with_detector=2)

    layers1 = [(i,x) for i,x in enumerate(model.layers) if len(x.weights)]
    layers2 = model_with_detector.layers
    layers2 = [(i,x) for i,x in enumerate(layers2) if all('/detector' not in y.name for y in x.weights) and len(x.weights)]

    set_layers = []

    for (i,e),(j,f) in zip(layers1,layers2):
        ee = [re.sub("_[0-9]+","",x.name) for x in e.weights]
        ff = [re.sub("_[0-9]+","",x.name).replace("model_with_detector","model") for x in f.weights]
        set_layers.append(j)

        assert ee==ff
        f.set_weights(e.get_weights())

    print(set_layers)

    model_with_detector.save_weights("models/cifar-resnet-detector-base")

    #"""

    model_with_detector = ResnetBuilder.build_resnet_32((3, 32, 32), 10, with_detector=2)
    model_with_detector.load_weights("models/cifar-resnet-detector-base")
    

    set_layers = [1, 2, 4, 5, 7, 9, 11, 12, 14, 16, 18, 19, 21, 23, 25, 26, 28, 
                  30, 32, 33, 35, 37, 39, 40, 42, 43, 45, 47, 48, 50, 52, 54, 
                  55, 57, 59, 61, 62, 64, 66, 68, 69, 71, 73, 75, 76, 78, 79, 
                  81, 83, 84, 86, 88, 90, 91, 93, 95, 97, 98, 102, 106, 110, 
                  112, 116, 120, 128]

    for e in set_layers:
        model_with_detector.layers[e].trainable = False

    #print('Test accuracy',np.mean(np.argmax(model_with_detector.predict(data.test_data)[0],axis=1)==np.argmax(data.test_labels,axis=1)))


    train_data, train_labels = data.train_data, data.train_labels
    N = len(train_data)
    """ # uncomment to create the adversarial training data
    model = ResnetBuilder.build_resnet_32((3, 32, 32), 10, activation=False)
    model.load_weights("models/cifar-resnet")
    model = Wrap(model)
    
    
    #attack = FGS(sess, model)
    attack = CarliniL2(sess, model, batch_size=100, binary_search_steps=3,
                       initial_const=0.1, max_iterations=3000, learning_rate=0.005,
                       confidence=0, targeted=False)

    for i in range(0,N,1000):
        now=time.time()
        train_adv = attack.attack(data.train_data[i:i+1000], data.train_labels[i:i+1000])
        print(time.time()-now)
        print('accuracy',np.mean(np.argmax(model.model.predict(train_adv),axis=1)==np.argmax(data.train_labels[i:i+1000],axis=1)))
        np.save("tmp/adv"+path.split("/")[1]+str(i), train_adv)
    print('Accuracy on valid training',np.mean(np.argmax(model.model.predict(data.train_data),axis=1)==np.argmax(data.train_labels,axis=1)))
    print('Accuracy on adversarial training',np.mean(np.argmax(model.model.predict(train_adv),axis=1)==np.argmax(data.train_labels,axis=1)))
    #"""

    train_adv = []
    for i in range(0,N,1000):
        train_adv.extend(np.load("tmp/adv"+path.split("/")[1]+str(i)+".npy"))
    train_adv = np.array(train_adv)

    newX = np.zeros((train_data.shape[0]*2,)+train_data.shape[1:])
    newX[:train_data.shape[0]] = train_data
    newX[train_data.shape[0]:] = train_adv
    newY = np.zeros((train_data.shape[0]*2,1))
    newY[:train_data.shape[0]] = 0
    newY[train_data.shape[0]:] = 1

    for i in range(100):
        train(sess, model_with_detector, newX, np.concatenate([train_labels]*2,axis=0), newY, 
              path+"-layerdetect-"+str(i)+"-"+str(num),
              LEARNING_RATE=10**random.randint(-5,-1), MOMENTUM=random.random(), 
              OPTIMIZER=random.choice(['sgd', 'sgd', 'sgd', 'sgd', 'sgd', 'adam', 'rmsprop']),
              NUM_EPOCHS=20,
              BATCH_SIZE=2**random.randint(4,9))

def run_nn_detection(Data, path):
    
    data = Data()
    sess = K.get_session()
    K.set_learning_phase(False)
    
    model_with_detector = ResnetBuilder.build_resnet_32((3, 32, 32), 10, 
                                                        with_detector=2, activation=False)
    model_with_detector.save_weights("/tmp/q")
    

    model_with_detector.load_weights("models/cifar-layerdetect-37-0")

    N = 10#len(data.test_data)//100
    """ # uncomment to generate adversarial testing data
    model = ResnetBuilder.build_resnet_32((3, 32, 32), 10, activation=False)
    model.load_weights("models/cifar-resnet")
    model = Wrap(model)

    #attack = FGS(sess, model)
    attack = CarliniL2(sess, model, batch_size=100, binary_search_steps=3,
                       initial_const=0.1, max_iterations=3000, learning_rate=0.005,
                       confidence=0, targeted=False)

    for i in range(0,N,1000):
        test_adv = attack.attack(data.test_data[i:i+100], data.test_labels[i:i+100])
        np.save("tmp/testadv"+path.split("/")[1]+str(i), test_adv)
    #"""

    test_adv = []
    for i in range(0,N,1000):
        test_adv.extend(np.load("tmp/testadv"+path.split("/")[1]+str(i)+".npy"))
    test_adv = np.array(test_adv)

    print('Accuracy of model on test set',np.mean(np.argmax(model_with_detector.predict(data.test_data)[0],axis=1)==np.argmax(data.test_labels,axis=1)))
    print('Accuracy of model on adversarial data',np.mean(np.argmax(model_with_detector.predict(test_adv)[0],axis=1)==np.argmax(data.test_labels,axis=1)))

    print('Probaility detects valid data as valid',np.mean(model_with_detector.predict(data.test_data)[1]<=0))
    print('Probability detects adversarail data as adversarial',np.mean(model_with_detector.predict(test_adv)[1]>0))

    xs = tf.placeholder(tf.float32, [None, 32, 32, 3])
    rmodel = RobustModel(model_with_detector)
    preds = rmodel.predict(xs)

    y1 = np.argmax(sess.run(preds, {xs: data.test_data[:N]}),axis=1)
    print('Robust model accuracy on test dat',np.mean(y1==np.argmax(data.test_labels[:N],axis=1)))
    print('Probability robust model detects valid data as adversarial', np.mean(y1==10))

    y2 = np.argmax(sess.run(preds, {xs: test_adv}),axis=1)
    print('Probability robust model detects adversarial data as adversarial', np.mean(y2==10))

    attack = CarliniL2(sess, rmodel, batch_size=10, binary_search_steps=3,
                       initial_const=0.1, max_iterations=300, learning_rate=0.01,
                       confidence=0, targeted=True)

    targets = np.argmax(model_with_detector.predict(test_adv[:N])[0],axis=1)
    realtargets = np.zeros((N, 11))
    realtargets[np.arange(N),targets] = 1

    np.save("tmp/adaptiveattack",attack.attack(data.test_data[:N], realtargets))
    adv = np.load("tmp/adaptiveattack.npy")

    print('Accuracy on adversarial data',np.mean(np.argmax(model_with_detector.predict(adv)[0],axis=1)==np.argmax(data.test_labels,axis=1)))

    print('Probability detector detects adversarial data as adversarial',np.mean(model_with_detector.predict(adv)[1]>0))

    d=np.sum((adv-data.test_data[:N])**2,axis=(1,2,3))**.5
    print("mean distortion attacking robust model", np.mean(d))

    d=np.sum((test_adv[:N]-data.test_data[:N])**2,axis=(1,2,3))**.5
    print("mean distortion attacking unsecurred model", np.mean(d))
    

    model_with_detector_2 = ResnetBuilder.build_resnet_32((3, 32, 32), 10, 
                                                        with_detector=2, activation=False)
    model_with_detector_2.load_weights("models/cifar-layerdetect-42-0")


    print('Accuracy on adversarial data',np.mean(np.argmax(model_with_detector_2.predict(adv)[0],axis=1)==np.argmax(data.test_labels,axis=1)))

    print('Probability detector detects adversarial data as adversarial',np.mean(model_with_detector_2.predict(adv)[1]>0))
    
#for i in range(100):
#    train_nn_detection(CIFAR, CIFARModel, "models/cifar")

#run_nn_detection(CIFAR, "models/cifar")
