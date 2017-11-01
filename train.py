## train.py -- train the MNIST and CIFAR models
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

from setup_mnist import MNIST, MNISTModel
from setup_cifar import CIFAR, CIFARModel
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

import tensorflow as tf

def train(data, Model, file_name, num_epochs=50, batch_size=128, init=None):
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)

    model = Model(None).model
    print(model.summary())

    def get_lr(epoch):
        return base_lr*(.5**(epoch/num_epochs*10))
    sgd = SGD(lr=0.00, momentum=0.9, nesterov=False)
    schedule= LearningRateScheduler(get_lr)

    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    if Model == MNISTModel:
        datagen = ImageDataGenerator(
            rotation_range=0,
            width_shift_range=0.0,
            height_shift_range=0.0,
            horizontal_flip=False)
        base_lr = 0.1
    else:
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)
        base_lr = 0.1


    datagen.fit(data.train_data)

    model.fit_generator(datagen.flow(data.train_data, data.train_labels,
                                     batch_size=batch_size),
                        steps_per_epoch=data.train_data.shape[0] // batch_size,
                        epochs=num_epochs,
                        verbose=1,
                        validation_data=(data.validation_data, data.validation_labels),
                        callbacks=[schedule])

    print('Test accuracy:', np.mean(np.argmax(model.predict(data.test_data),axis=1)==np.argmax(data.test_labels,axis=1)))

    if file_name != None:
        model.save_weights(file_name)

    return model

if __name__ == "__main__":
    train(MNIST(), MNISTModel, "models/mnist", num_epochs=30)
    train(CIFAR(), CIFARModel, "models/cifar", num_epochs=300)
