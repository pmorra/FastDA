#!/usr/bin/python3
# -*- coding: utf-8 -*-

# DeepONet class

import os
import copy
import numpy as np
import tensorflow as tf
from   tensorflow import keras
import time

tf.keras.backend.set_floatx('float32')

class DeepONet:
    """
    DeepONet
    The class creates a keras.Model with a branch and trunk networks.
    The code assumes the data is given in batched tf.data.Dataset formats.

    Parameters
    ----------

    m : int
        Number of sensors (second dimension of branch network's input)
    dim_y : int
        Dimension of y (trunk network's input)
    depth_branch : int
        depth of branch network
    depth_trunk : int
        depth of branch network
    p : int
        Width of branch and trunk networks
    dim_out : int
        Dimension of output. Must be a true divisor of p. Default is 1. 
    dest : str [optional]
        Path for output files.
    activation : str [optional]
        Activation function to be used. Default is 'relu'.
    optimizer : keras.optimizer instance [optional]
        Optimizer to be used in the gradient descent. Default is Adam with
        fixed learning rate equal to 1e-3.
    norm_in : float or array [optional]
        If a number or an array of size din is supplied, the first layer of the
        network normalizes the inputs uniformly between -1 and 1. Default is
        False.
    norm_out : float or array [optional]
        If a number or an array of size dim_out is supplied, the layer layer of the
        network normalizes the outputs using z-score. Default is
        False.
    norm_out_type : str [optional]
        Type of output normalization to use. Default is 'z-score'.
    save_freq : int [optional]
        Save model frequency. Default is 1.
    restore : bool [optional]
        If True, it checks if a checkpoint exists in dest. If a checkpoint
        exists it restores the modelfrom there. Default is True.
    """
    # Initialize the class
    def __init__(self,
                 m,
                 dim_y,
                 depth_branch,
                 depth_trunk,
                 p,
                 dim_out=1,
                 dest='./',
                 regularizer=None,
                 p_drop=0.0,
                 activation='relu',
                 slope_recovery=False,
                 optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                 norm_in=False,
                 norm_out=False,
                 norm_out_type='z-score',
                 save_freq=1,
                 restore=True):

        # Numbers and dimensions
        self.m            = m
        self.dim_y        = dim_y
        self.dim_out      = dim_out
        self.depth_branch = depth_branch
        self.depth_trunk  = depth_trunk
        self.width        = p

        # Extras
        self.dest        = dest
        self.regu        = regularizer
        self.norm_in     = norm_in
        self.norm_out    = norm_out
        self.optimizer   = optimizer
        self.save_freq   = save_freq
        self.activation  = activation

        # Activation function
        if activation=='tanh':
            self.act_fn = keras.activations.tanh
            self.kinit  = 'glorot_normal'
        elif activation=='relu':
            self.act_fn = keras.activations.relu
            self.kinit  = 'he_normal'
        elif activation=='elu':
            self.act_fn = keras.activations.elu
            self.kinit  = 'glorot_normal'

        # Inputs definition
        funct = keras.layers.Input(m,     name='funct')
        point = keras.layers.Input(dim_y, name='point')

        # Normalize input
        if norm_in:
            fmin   = norm_in[0][0]
            fmax   = norm_in[0][1]
            pmin   = norm_in[1][0]
            pmax   = norm_in[1][1]
            norm_f   = lambda x: 2*(x-fmin)/(fmax-fmin) - 1
            norm_p   = lambda x: 2*(x-pmin)/(pmax-pmin) - 1
            hid_b = keras.layers.Lambda(norm_f)(funct)
            hid_t = keras.layers.Lambda(norm_p)(point)
        else:
            hid_b = funct
            hid_t = point

        # Branch network
        for ii in range(self.depth_branch-1):
            hid_b = keras.layers.Dense(self.width,
                                       kernel_regularizer=self.regu,
                                       kernel_initializer=self.kinit,
                                       activation=self.act_fn)(hid_b)
            if p_drop:
                hid_b = keras.layers.Dropout(p_drop)(hid_b)
        hid_b = keras.layers.Dense(self.width,
                                   kernel_initializer=self.kinit,
                                   kernel_regularizer=self.regu)(hid_b)

        # Trunk network
        for ii in range(self.depth_trunk):
            hid_t = keras.layers.Dense(self.width,
                                       kernel_regularizer=self.regu,
                                       kernel_initializer=self.kinit,
                                       activation=self.act_fn)(hid_t)
            if p_drop and ii<self.depth_trunk-1:
                hid_t = keras.layers.Dropout(p_drop)(hid_t)

        # Output definition
        if dim_out>0:
            hid_b = keras.layers.Reshape((dim_out, p//dim_out))(hid_b)
            hid_t = keras.layers.Reshape((dim_out, p//dim_out))(hid_t)
        output = keras.layers.Multiply()([hid_b, hid_t])
        output = tf.reduce_sum(output, axis=2)
        
        output = BiasLayer()(output)

        # Normalize output
        if norm_out:
            if norm_out_type=='z_score':
                mm = norm_out[0]
                sg = norm_out[1]
                out_norm = lambda x: sg*x + mm 
            elif norm_out_type=='min_max':
                ymin = norm_out[0]
                ymax = norm_out[1]
                out_norm = lambda x: 0.5*(x+1)*(ymax-ymin) + ymin
            output = keras.layers.Lambda(out_norm)(output)

        # Create model
        model = keras.Model(inputs=[funct, point], outputs=[output])
        self.model = model
        self.num_trainable_vars = np.sum([np.prod(v.shape)
                                          for v in self.model.trainable_variables])
        self.num_trainable_vars = tf.cast(self.num_trainable_vars, tf.float32)

        # Create save checkpoints, managers and callbacks
        self.ckpt    = tf.train.Checkpoint(step=tf.Variable(0),
                                           model=self.model,
                                           optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(self.ckpt,
                                                  self.dest + '/ckpt',
                                                  max_to_keep=5)
        if restore:
            self.ckpt.restore(self.manager.latest_checkpoint)

    def train(self,
              train_dataset,
              epochs=10,
              verbose=False,
              print_freq=1,
              valid_freq=0,
              early_stopping=False,
              val_threshold=np.inf,
              valid_dataset=None,
              save_freq=1):
        """
        Train function

        Loss functions written to loss.dat

        Parameters
        ----------
        train_dataset : tf.data.Dataset
            Training data. Shoud be of the form X, Y, W with X=(X_branch,
            X_trunk). W plays the role of loss function sample weights 
        epochs : int [optional]
            Number of epochs to train. Default is 10.
        verbose : bool [optional]
            Verbose output or not. Default is False.
        print_freq : int [optional]
            Print status frequency. Default is 1.
        early_stopping : bool [optional]
            If True only saves the model Checkpoint when the validation is
            decreasing. Default is False.
        valid_dataset : tf.data.Dataset
            Validation data. Same description as training_data, but does not
            use a data_mask.
        save_freq : int [optional]
            Save model frequency. Default is 1.
        """

        # Run epochs
        ep0 = int(self.ckpt.step)
        best_val = np.inf
        for ep in range(ep0, ep0+epochs):
            # Loop through batches
            for X, Y, W in train_dataset:
                (loss_data) = self.training_step(X, Y, W)
            # Get validation
            if valid_dataset is not None:
                valid = self.validation(valid_dataset)
            else:
                valid = loss_data
            # Print status
            if ep%print_freq==0:
                status = [loss_data.numpy()]
                if valid_dataset is not None:
                    status.append(valid.numpy())
                self.print_status(ep, status, verbose=verbose)
            # Save progress
            self.ckpt.step.assign_add(1)
            if ep%save_freq==0 and valid.numpy()<best_val:
                self.manager.save()
                if early_stopping and valid.numpy()<val_threshold:
                    best_val = valid.numpy()

    @tf.function
    def training_step(self, X, Y, W):
        with tf.GradientTape(persistent=True) as tape:
            # Data part
            Y_p = self.model(X, training=True)
            aux = [tf.reduce_mean(W*tf.square(Y[:,ii]-Y_p[:,ii])) for ii in range(self.dim_out)]
            loss_data = tf.add_n(aux)/self.dim_out
            # Add regularization
            loss_data = tf.add_n([loss_data] + self.model.losses)
        # Calculate gradients
        gradients_data = tape.gradient(loss_data,
                    self.model.trainable_variables,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO)
        gradients = gradients_data
        self.optimizer.apply_gradients(zip(gradients,
                    self.model.trainable_variables))
        return loss_data

    @tf.function
    def validation(self, valid_dataset):
        jj  = 0.0
        acc = 0.0
        for X, Y, W in valid_dataset:
            Y_p = self.model(X, training=True)
            aux = [tf.reduce_mean(W*tf.square(Y[:,ii]-Y_p[:,ii]))
                   for ii in range(self.dim_out)]
            acc += tf.add_n(aux)/self.dim_out
            jj  += 1.0
        return acc/jj

    def print_status(self, ep, status, verbose=False):
        """ Print status function """
        # Loss functions
        output_file = open(self.dest + 'loss.dat', 'a')
        print(ep, *status, file=output_file)
        output_file.close()
        if verbose:
            print(ep, *status)

class BiasLayer(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True)
    def call(self, x):
        return x + self.bias

