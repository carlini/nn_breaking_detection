# Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>
# All rights reserved.

import tensorflow as tf
import numpy as np

class FGS:
    def __init__(self, sess, model, batch_size=100, eps=0.2):
        self.sess = sess
        self.model = model

        self.delta = tf.placeholder(tf.float32, (batch_size,model.image_size,model.image_size,model.num_channels))
        self.img = tf.placeholder(tf.float32, (batch_size,model.image_size,model.image_size,model.num_channels))
        self.lab = tf.placeholder(tf.float32, (batch_size,model.num_labels))
        
        self.out = model.predict(tf.clip_by_value(self.img+self.delta, -0.5, 0.5))
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, 
                                                                          labels=self.lab))

        self.grads = tf.gradients(self.loss,[self.delta])[0]
        
        self.batch_size = batch_size

        self.eps = eps

    def attack(self, imgs, labs):
        
        r = []
        
        for offset in range(0,len(imgs),self.batch_size):
            batch_imgs,batch_labs = imgs[offset:offset+self.batch_size],labs[offset:offset+self.batch_size]
        
            directions = np.sign(self.sess.run(self.grads, feed_dict={self.img: batch_imgs, 
                                                                      self.lab: batch_labs,
                                                                      self.delta: np.zeros(batch_imgs.shape)}))
            
            it = np.clip(batch_imgs+directions*self.eps, -.5, .5)

            r.extend(it)
        return np.array(r)
