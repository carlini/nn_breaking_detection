## dimensionality_reduction.py -- break PCA dim-reduction
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

from setup_cifar import CIFARModel, CIFAR
from setup_mnist import MNISTModel, MNIST

sys.path.append("../..")
from nn_robust_attacks.l2_attack import CarliniL2
from fast_gradient_sign import FGS

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D
from keras.optimizers import SGD

import matplotlib
import matplotlib.pyplot as plt

def make_model(size):
    model = Sequential()

    model.add(Dense(100, input_shape=(size,)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(100))
    model.add(Activation('relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(10))

    return model

def train(data, file_name, components=100, num_epochs=20, batch_size=256, pca=None, invert=False):
    """
    Standard neural network training procedure.
    """

    shape = (-1, data.train_data.shape[1]*data.train_data.shape[2]*data.train_data.shape[3])

    train_data = pca.transform(data.train_data.reshape(shape))[:,:components]
    validation_data = pca.transform(data.validation_data.reshape(shape))[:,:components]
    test_data = pca.transform(data.test_data.reshape(shape))[:,:components]

    print(train_data.shape)

    if invert:
        train_data = pca.inverse_transform(train_data).reshape((-1, 28, 28, 1))
        validation_data = pca.inverse_transform(validation_data).reshape((-1, 28, 28, 1))
        test_data = pca.inverse_transform(test_data).reshape((-1, 28, 28, 1))
        
        model = MNISTModel(None).model
    else:
        model = make_model(components)
    
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)

    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss=fn,
                  optimizer='adam',
                  metrics=['accuracy'])
    
    model.fit(train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(validation_data, data.validation_labels),
              nb_epoch=num_epochs,
              shuffle=True)

    acc = np.mean(np.argmax(model.predict(test_data),axis=1)==np.argmax(data.test_labels,axis=1))
    print("Overall accuracy on test set:", acc)

    if file_name != None:
        model.save(file_name)

    return model

class Wrap:
    def __init__(self, model, pca):
        self.image_size = 28
        self.num_channels = 1
        self.num_labels = 10
        self.model = model
        self.mean = tf.constant(pca.mean_,dtype=tf.float32)
        self.components = tf.constant(pca.components_.T,dtype=tf.float32)

    def predict(self, xs):
        xs = tf.reshape(xs,(-1,784))
        xs -= self.mean
        xs = tf.matmul(xs, self.components)
        return self.model(xs)

def run_pca(Data, num_components=10, invert=False):
    data = Data()

    sess = K.get_session()

    K.set_learning_phase(False)

    shape = (-1, 784)
    
    pca = sklearn.decomposition.PCA(n_components=num_components)

    pca.fit(data.train_data.reshape(shape)) # [:10000]

    if invert:
        model = MNISTModel("models/mnist-pca-cnn-top-"+str(num_components))
    else:
        model = make_model(num_components)
        model.load_weights("models/mnist-pca-top-"+str(num_components))
        model = Wrap(model,pca)

    tf_mean = tf.constant(pca.mean_,dtype=tf.float32)
    tf_components = tf.constant(pca.components_.T,dtype=tf.float32)

    def new_predict(xs):
        # map to PCA space
        xs = tf.reshape(xs,(-1,784))
        xs -= tf_mean
        xs = tf.matmul(xs, tf_components)
    
        # map back
        xs = tf.matmul(xs, tf.transpose(tf_components))
        xs += tf_mean
        xs = tf.reshape(xs, (-1, 28, 28, 1))
        return model.model(xs)

    if invert:
        model.predict = new_predict

    attack = CarliniL2(sess, model, batch_size=100, max_iterations=3000, 
                       binary_search_steps=6, targeted=False,
                       initial_const=1)

    N = 100

    test_adv = attack.attack(data.test_data[:N], data.test_labels[:N])

    print('accuracy',np.mean(np.argmax(sess.run(model.predict(tf.constant(data.test_data,dtype=np.float32))),axis=1)==np.argmax(data.test_labels,axis=1)))

    print(list(test_adv[0].flatten()))

    print('dist',np.mean(np.sum((test_adv-data.test_data[:N])**2,axis=(1,2,3))**.5))

    it = np.argmax(sess.run(model.predict(tf.constant(test_adv))),axis=1)
    print('success',np.mean(it==np.argmax(data.test_labels,axis=1)[:N]))
    

def show(img):
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))

def do_all(invert=False):
    if invert:
        name = "models/mnist-pca-cnn-top-"
    else:
        name = "models/mnist-pca-top-"
    """
    for n in range(3,28):
        n = n**2
        pca = sklearn.decomposition.PCA(n_components=n)
        pca.fit(MNIST().train_data.reshape([-1,784]))
        train(MNIST(), name+str(n), n, pca=pca, invert=invert)
    """
    for n in range(3,28):
        n = n**2
        run_pca(MNIST, n, invert)

do_all(False)

def compare_baseline():
    data = MNIST()
    model = MNISTModel("models/mnist")
    sess = K.get_session()

    attack = CarliniL2(sess, model, batch_size=100, max_iterations=3000, 
                       binary_search_steps=4, targeted=False,
                       initial_const=10)

    N = 100
    test_adv = attack.attack(data.test_data[:N], data.test_labels[:N])
    print('dist',np.mean(np.sum((test_adv-data.test_data[:N])**2,axis=(1,2,3))**.5))

#compare_baseline()
accs = []
dists = []
for line in open("logs/dimensionality_reduction"):
    if 'accuracy ' in line[:10]:
        accs.append(float(line.split()[1]))
    elif 'dist ' in line:
        dists.append(float(line.split()[1]))

dists2=[]
for line in open("logs/dimensionality_reduction_cnn"):
    if 'accuracy ' in line[:10]:
        accs.append(float(line.split()[1]))
    elif 'dist ' in line:
        dists2.append(float(line.split()[1]))

fig = plt.figure(figsize=(4,3))
fig.subplots_adjust(bottom=0.17,left=.14)

axes = plt.gca()
axes.set_ylim([0,2.3])

plt.xlabel('Number of Principle Components')
plt.ylabel('Mean Distance to Adversarial Example')
a=plt.plot([9,27**2],[1.32, 1.32],'r--', label='Baseline (CNN)')[0]
b=plt.plot(np.arange(3,28)**2, dists2, label='PCA Model (CNN)')[0]
c=plt.plot(np.arange(3,28)**2, dists, label='PCA Model (FC)')[0]

import matplotlib.patches as mpatches
plt.legend(handles=[a, b, c])

from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('/tmp/a.pdf')
plt.savefig(pp, format='pdf')
pp.close()

plt.show()
