## outlier_class_training.py -- break an outlier class detector
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

import utils
import random
import tensorflow as tf
from setup_mnist import MNIST, MNISTModel
from setup_cifar import CIFAR, CIFARModel
import os

from keras import backend as K

from fast_gradient_sign import FGS
from nn_robust_attacks.l2_attack import CarliniL2

def train(Model, data, num_labels, file_name, num_epochs=50, batch_size=128):
    """
    Standard neural network training procedure.
    """

    model = Model(num_labels=num_labels).model
    
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              epochs=num_epochs,
              shuffle=True)

    acc = np.mean(np.argmax(model.predict(data.test_data),axis=1)==np.argmax(data.test_labels,axis=1))
    print("Overall accuracy on test set:", acc)

    if file_name != None:
        model.save(file_name)

    return model

class Wrap:
    def __init__(self, model, image_size, num_channels, num_labels):
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.model = model

    def predict(self, xs):
        return self.model(xs)

def run_evaluation(Data, Model, path, num_epochs, name):
    data = Data()

    #train(Model, data, 10, path, num_epochs=num_epochs)

    sess = K.get_session()
    K.set_learning_phase(False)

    model = Model(path)
    #attack = FGS(sess, model)
    attack = CarliniL2(sess, model, batch_size=100, max_iterations=3000,
                       binary_search_steps=3, targeted=True, initial_const=10, learning_rate=1e-2)
    
    """ # uncomment to run the training phase
    
    train_adv = attack.attack(data.train_data, data.train_labels)
    np.save("tmp/"+name+"outlieradvtrain",train_adv)
    train_adv = np.load("tmp/"+name+"outlieradvtrain.npy")
    data.train_data = np.concatenate((data.train_data, train_adv))
    data.train_labels = np.concatenate((data.train_labels, np.zeros(data.train_labels.shape, dtype=np.float32)))
    data.train_labels = np.pad(data.train_labels, [[0, 0], [0, 1]], mode='constant')
    data.train_labels[data.train_labels.shape[0]//2:,10] = 1

    validation_adv = attack.attack(data.validation_data, data.validation_labels)
    np.save("tmp/"+name+"outlieradvvalidation",validation_adv)
    validation_adv = np.load("tmp/"+name+"outlieradvvalidation.npy")
    data.validation_data = np.concatenate((data.validation_data, validation_adv))
    data.validation_labels = np.concatenate((data.validation_labels, np.zeros(data.validation_labels.shape, dtype=np.float32)))
    data.validation_labels = np.pad(data.validation_labels, [[0, 0], [0, 1]], mode='constant')
    data.validation_labels[data.validation_labels.shape[0]//2:,10] = 1

    test_adv = attack.attack(data.test_data, data.test_labels)
    np.save("tmp/"+name+"outlieradvtest",test_adv)
    test_adv = np.load("tmp/"+name+"outlieradvtest.npy")
    data.test_data = np.concatenate((data.test_data, test_adv))
    data.test_labels = np.concatenate((data.test_labels, np.zeros(data.test_labels.shape, dtype=np.float32)))
    data.test_labels = np.pad(data.test_labels, [[0, 0], [0, 1]], mode='constant')
    data.test_labels[data.test_labels.shape[0]//2:,10] = 1

    train(Model, data, 11, path+"_advtraining", num_epochs=num_epochs)

    data1 = Data() # just need a reference, this is a bit ugly to do
    data2 = Data() # just need a reference, this is a bit ugly to do

    idxs = list(range(len(data.train_data)))
    random.shuffle(idxs)

    data1.train_data = data.train_data[idxs[:len(idxs)//2]]
    data2.train_data = data.train_data[idxs[len(idxs)//2:]]
    data1.train_labels = data.train_labels[idxs[:len(idxs)//2],:]
    data2.train_labels = data.train_labels[idxs[len(idxs)//2:],:]

    idxs = list(range(len(data.validation_data)))
    random.shuffle(idxs)
    data1.validation_data = data.validation_data[idxs[:len(idxs)//2]]
    data2.validation_data = data.validation_data[idxs[len(idxs)//2:]]
    data1.validation_labels = data.validation_labels[idxs[:len(idxs)//2]]
    data2.validation_labels = data.validation_labels[idxs[len(idxs)//2:]]

    idxs = list(range(len(data.test_data)))
    random.shuffle(idxs)
    data1.test_data = data.test_data[idxs[:len(idxs)//2]]
    data2.test_data = data.test_data[idxs[len(idxs)//2:]]
    data1.test_labels = data.test_labels[idxs[:len(idxs)//2]]
    data2.test_labels = data.test_labels[idxs[len(idxs)//2:]]

    train(Model, data1, 11, path+"_advtraining-left", num_epochs=num_epochs)
    train(Model, data2, 11, path+"_advtraining-right", num_epochs=num_epochs)
    #"""


    K.set_learning_phase(False)

    rmodel = Model(num_labels=11).model
    rmodel.load_weights(path+"_advtraining")
    if name == "cifar":
        rmodel = Wrap(rmodel, 32, 3, 11)
    else:
        rmodel = Wrap(rmodel, 28, 1, 11)
        

    rmodel1 = Model(num_labels=11).model
    rmodel1.load_weights(path+"_advtraining-left")
    if name == "cifar":
        rmodel1 = Wrap(rmodel1, 32, 3, 11)
    else:
        rmodel1 = Wrap(rmodel1, 28, 1, 11)

    rmodel2 = Model(num_labels=11).model
    rmodel2.load_weights(path+"_advtraining-right")
    if name == "cifar":
        rmodel2 = Wrap(rmodel2, 32, 3, 11)
    else:
        rmodel2 = Wrap(rmodel2, 28, 1, 11)

    rmodel2.model.summary()

    attack2 = CarliniL2(sess, rmodel, batch_size=100, max_iterations=2000, confidence=.1,
                        binary_search_steps=3, targeted=True, initial_const=10, learning_rate=1e-2)


    #test_adv = np.load("tmp/outlieradvtest.npy")
    #print('qq',np.mean(rmodel.model.predict_classes(test_adv)==10))

    N = 100
    targets = utils.get_labs(data.test_labels[:100])
    #"""
    test_adv = attack.attack(data.test_data[:N], targets)
    print('mean distortion',np.mean(np.sum((test_adv-data.test_data[:N])**2,axis=(1,2,3))**.5))
    print('model predict',np.argmax(model.model.predict(test_adv),axis=1))
    print('rmodel predict',np.argmax(rmodel.model.predict(test_adv),axis=1))
    #"""


    targets2 = np.zeros((N, 11))
    targets2[:, :10] = targets
    test_adv = attack2.attack(data.test_data[:N], targets2)
    print(list(test_adv[0].flatten()))
    print('mean distortion',np.mean(np.sum((test_adv-data.test_data[:N])**2,axis=(1,2,3))**.5))

    a=(np.argmax(model.model.predict(test_adv),axis=1))
    #print(a)
    print('summary',np.mean(a==np.argmax(targets,axis=1)),np.mean(a==10))

    a=(np.argmax(rmodel.model.predict(test_adv),axis=1))
    #print(a)
    print('summary',np.mean(a==np.argmax(targets,axis=1)),np.mean(a==10))

    a=(np.argmax(rmodel1.model.predict(test_adv),axis=1))
    #print(a)
    print('summary',np.mean(a==np.argmax(targets,axis=1)),np.mean(a==10))

    a=(np.argmax(rmodel2.model.predict(test_adv),axis=1))
    #print(a)
    print('summary',np.mean(a==np.argmax(targets,axis=1)),np.mean(a==10))


    
    

run_evaluation(MNIST, MNISTModel, "models/mnist", num_epochs=30, name="mnist")
run_evaluation(CIFAR, CIFARModel, "models/cifar", num_epochs=100, name="cifar")
