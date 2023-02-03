# Author: Mattia Silvestri

"""
    Implementation of the Euler method in Tensorflow and related utility methods.
"""
import sys

import tensorflow as tf
# import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import numpy as np
from typing import Tuple, List, Union
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

tf.executing_eagerly()
tf.config.run_functions_eagerly(True)

########################################################################################################################


def build_model(input_size: int,
                output_size: int,
                hidden: list = [],
                output_activation: str = 'exponential',
                name: str = None) -> tf.keras.Model:
    # Build all layers
    nn_in = tf.keras.Input(shape=(input_size,))
    nn_out = nn_in
    for h in hidden:
        nn_out = tf.keras.layers.Dense(h, activation='relu')(nn_out)
    nn_out = tf.keras.layers.Dense(output_size, activation=output_activation)(nn_out)
    # Build the model
    model = tf.keras.Model(inputs=nn_in, outputs=nn_out, name=name)
    return model

########################################################################################################################


def SIR(y, beta, gamma=1 / 10):
    """ function performing SIR simulation """
    S, I, R = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    S_to_I = beta * I * S
    I_to_R = gamma * I
    dS = - S_to_I
    dI = S_to_I - I_to_R
    dR = I_to_R

    return tf.concat([dS, dI, dR], axis=1)

########################################################################################################################


def SEIR(y, beta, epsilon=1 / 5, gamma=1 / 10):
    """ function performing SIR simulation """
    S, E, I, R = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]

    S_to_E = beta * I * S
    E_to_I = epsilon * E
    I_to_R = gamma * I
    dS = - E_to_I
    dE = S_to_E - E_to_I
    dI = E_to_I - I_to_R
    dR = I_to_R

    return tf.concat([dS, dE, dI, dR], axis=1)

########################################################################################################################


class PairedSIR(tf.keras.Model):
    """
        Method for fitting beta parametrization of a SIR
        covering a specific time window
    """

    def __init__(self,
                 steps: int,
                 gamma: float,
                 stochastic: bool = False):

        """
        Constructor of the class.
        :param T int; lenght of historical data (in days)
        :param steps int; the number of steps of the Euler method.
        :param gamma float; recovery rate of SIR model
        :param stochastic bool: flag to indicate stochastic training
        :param mu float: learning rate for lagrangian smoothing coefficient
        """

        super(PairedSIR, self).__init__()

        self.steps = steps
        self.gamma = gamma
        self.stochastic = stochastic

        # Parameters
        self.initializer = tf.keras.initializers.RandomUniform(minval=0, seed=42)
        # shape_betas = (self.T, 2) if self.stochastic else (self.T,)
        self.beta = tf.Variable(self.initializer(shape=(1,), dtype='float32'), trainable=True)

        if self.stochastic:
            self.std = tf.Variable(self.initializer(shape=(1,), dtype='float32'),
                                   trainable=True,
                                   constraint=lambda x: tf.clip_by_value(x, 0, 1.))

        # Trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, data, training=False):
        y0 = data
        S, I, R = y0[:, 0], y0[:, 1], y0[:, 2]
        h = 1 / self.steps  # update step euler method

        # if the model is stochastic we sample beta
        if self.stochastic:
            # mean, std = self.betas[:,0], self.betas[:,1]
            epsilon = tf.random.normal((1,), mean=0.5, stddev=0.5)
            beta = (self.beta + self.std * epsilon)
        else:
            beta = self.beta

        # SIR integration
        for _ in range(self.steps):
            S_to_I = h * (beta * I * S)
            I_to_R = h * self.gamma * I
            S = S - S_to_I
            I = I + S_to_I - I_to_R
            R = R + I_to_R
        # TODO - improve code here, there's a way to avoi
        # all these append and reshape
        S = tf.reshape(S, [-1, 1])
        I = tf.reshape(I, [-1, 1])
        R = tf.reshape(R, [-1, 1])
        return tf.concat([S, I, R], 1)

    def train_step(self, data):
        y0, yt = data

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([self.beta])
            if self.stochastic:
                tape.watch([self.std])
            y_pred = self(y0, training=True)
            loss = self.compiled_loss(y_pred, yt)

            # Compute Gradient
        grads = tape.gradient(loss, [self.beta])
        grads_and_vars = list(zip(grads, [self.beta]))
        self.optimizer.apply_gradients(grads_and_vars)

        if self.stochastic:
            grads_std = tape.gradient(loss, [self.std])
            grads_and_vars_std = list(zip(grads_std, [self.std]))
            self.optimizer.apply_gradients(grads_and_vars_std)

        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]

########################################################################################################################


class UnrolledSIR(tf.keras.Model):
    """
        Method for fitting beta parametrization of a SIR
        covering a specific time window
    """

    def __init__(self,
                 T: int,
                 steps: int,
                 gamma: float,
                 stochastic: bool = False,
                 mu: float = 1e-1):

        """
        Constructor of the class.
        :param T int; lenght of historical data (in days)
        :param steps int; the number of steps of the Euler method.
        :param gamma float; recovery rate of SIR model
        :param stochastic bool: flag to indicate stochastic training
        :param mu float: learning rate for lagrangian smoothing coefficient
        """

        super(UnrolledSIR, self).__init__()

        self.T = T
        self.steps = steps
        self.gamma = gamma
        self.stochastic = stochastic
        self.mu = mu

        # Parameters
        self.initializer = tf.keras.initializers.RandomUniform(minval=0.01, maxval=0.04, seed=42)
        # shape_betas = (self.T, 2) if self.stochastic else (self.T,)
        self.beta = tf.Variable(self.initializer(shape=(1,), dtype='float32'), trainable=True)

        if self.stochastic:
            self.std = tf.Variable(self.initializer(shape=(1,), dtype='float32'), trainable=True)

        # Trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, data, training=False):
        y0 = data
        S, I, R = [y0[0]], [y0[1]], [y0[2]]
        h = 1 / self.steps  # update step euler method

        # if the model is stochastic we sample beta
        if self.stochastic:
            # mean, std = self.betas[:,0], self.betas[:,1]
            epsilon = tf.random.normal(shape=(1,), mean=0.5, stddev=0.5)
            beta = (self.beta + self.std * epsilon)[0]
        else:
            beta = self.beta[-1]
        for _ in range(self.T - 1):
            S_t, I_t, R_t = S[-1], I[-1], R[-1]

            # SIR integration
            for _ in range(self.steps):
                S_to_I = h * (beta * I_t * S_t)
                I_to_R = h * self.gamma * I_t
                S_t = S_t - S_to_I
                I_t = I_t + S_to_I - I_to_R
                R_t = R_t + I_to_R

            # TODO - improve code here, there's a way to avoi
            # all these append and reshape
            S = tf.concat([S, [S_t]], 0)
            I = tf.concat([I, [I_t]], 0)
            R = tf.concat([R, [R_t]], 0)
        S = tf.reshape(S, [-1, 1])
        I = tf.reshape(I, [-1, 1])
        R = tf.reshape(R, [-1, 1])

        return tf.concat([S, I, R], 1)[1:]  # we remove the first examle y0

    def train_step(self, data):
        y0, yt = data[0], data[1:]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([self.beta])
            if self.stochastic:
                tape.watch([self.std])
            y_pred = self(y0, training=True)
            loss = self.compiled_loss(y_pred, yt)

            # Compute Gradient
        grads = tape.gradient(loss, [self.beta])
        grads_and_vars = list(zip(grads, [self.beta]))
        self.optimizer.apply_gradients(grads_and_vars)

        if self.stochastic:
            grads_std = tape.gradient(loss, [self.std])
            grads_and_vars_std = list(zip(grads_std, [self.std]))
            self.optimizer.apply_gradients(grads_and_vars_std)

        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]

########################################################################################################################


class EndtoEndIntervetionMap(tf.keras.Model):
    """
        Method to approximate a compartmental model given
        a set of interventions or NPIs
    """

    def __init__(self,
                 u_ode,
                 f_ode,
                 steps: int = 2,
                 window_size: int = 7,
                 params: dict = {'gamma': 1 / 10},
                 stochastic: bool = False,
                 mu: float = 1e-2):
        """
        Constructor of the class.
        :param u_ode; neural net mappoing interventions and beta
        :para f_ode; compartmental model function
        :param steps int; the number of steps of the Euler method.
        :param window_size int; lenght in days for each time step cosidered
        :param params dict; set of fixed parameters of the compartmental model
        :param stochastic bool: flag to indicate stochastic training
        :param mu float: learning rate for lagrangian smoothing coefficient
        """

        super(EndtoEndIntervetionMap, self).__init__()

        self.u_ode = u_ode
        self.f_ode = f_ode
        self.steps = steps
        self.window_size = window_size
        self.params = params
        self.stochastic = stochastic
        self.mu = mu

        # Trackers
        self.loss_tracker = tf.keras.metrics.Mean()
        self.val_tracker = tf.keras.metrics.Mean()

    def call(self, data, training=False):
        y, x = data
        for layer in self.layers:
            x = layer(x)
        # computing beta if stochastic
        self.betas = x
        self.x = x
        if self.stochastic:
            ## TODO !!! samples negative values
            self.mean, self.std = x[:, 0], x[:, 1]
            epsilon = tf.random.normal(shape=(len(self.mean),), mean=.5, stddev=.5)[0]
            self.betas = tf.reshape(self.mean + self.std * epsilon, [-1, 1])
        # integration of compartmental model
        for _ in range(self.window_size * self.steps):
            y += 1 / self.steps * self.f_ode(y, self.betas, **self.params)
        return tf.cast(y, dtype=tf.float64)

    def train_step(self, data):
        # Unpack the data
        (y0, x), yt = data

        with tf.GradientTape() as tape:
            # Watch variables
            y_pred = self((y0, x), training=True)  # Forward pass
            # Compute the loss value
            loss = self.compiled_loss(yt[:, 1], y_pred[:, 1])

            # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Compute losses
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        (y0, x), yt = data

        y_pred = self((y0, x))
        loss = self.compiled_loss(yt[:, 1], y_pred[:, 1])
        self.val_tracker.update_state(loss)

        return {'loss': self.val_tracker.result()}

    def _tensor(self, xs, dtype=tf.float64):
        return [tf.convert_to_tensor(x, dtype) for x in xs]

    @property
    def metrics(self):
        return [self.loss_tracker, self.val_tracker]

