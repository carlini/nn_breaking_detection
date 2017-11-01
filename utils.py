## utils.py -- a collection of utilities
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import numpy as np

def get_labs(y):
    l = np.zeros((len(y),10))
    for i in range(len(y)):
        r = np.random.random_integers(0,9)
        while r == np.argmax(y[i]):
            r = np.random.random_integers(0,9)
        l[i,r] = 1
    return l
