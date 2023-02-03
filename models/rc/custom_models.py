# Author: Mattia Silvestri

"""
    Implementation of the Euler method in Tensorflow and related utility methods.
"""


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import numpy as np
import time
from typing import Tuple, List, Union

########################################################################################################################


class EulerMethod:
    """
    TensorFlow graph implementing the Euler method to solve first order differential equation learning some
    unknown parameters.

    Attributes:
        full_unrolling: bool; if True, the Euler method is unrolled for the full time horizon; otherwise, we
                                      expect a dataset with pairs of consecutive measurements.
        h_f: float; data sampling frequency, t_k - t_{k+1}.
        n_steps: int; number of Euler method steps.
        Vs_input_shape: tuple; if None, the Vs parameter of the ODE is approximated with single parameters; if
                               specified, input shape is used to build the linear regression model.
        tau_input_shape: tuple; if None, the \tau parameter of the ODE is approximated with single parameters; if
                                specified, input shape is used to build the linear regression model.
    """

    def __init__(self,
                 full_unrolling: bool,
                 h_f: float,
                 n_steps: int,
                 Vs_input_shape: Tuple[int],
                 tau_input_shape: Tuple[int],
                 trainable_Vs: bool,
                 trainable_tau: bool,
                 Vs_init_val: float,
                 tau_init_val: float,
                 kernel_init_val: int = 0,
                 bias_init_val: int = 0):
        """
        Constructor of the class.
        :param: full_unrolling: bool; if True, the Euler method is unrolled for the full time horizon; otherise, we
                                      expect a dataset with pairs of consecutive measurements.
        :param: h_f: float; data sampling frequency, t_k - t_{k+1}.
        :param: n_steps: int; number of Euler method steps.
        :param: Vs_input_shape: tuple; if None, the Vs parameter of the ODE is approximated with single parameters; if
                               specified, input shape is used to build the linear regression model.
        :param: tau_input_shape: tuple; if None, the \tau parameter of the ODE is approximated with single parameters; if
                                specified, input shape is used to build the linear regression model.
        :param: trainable_Vs: bool; if True, Vs parameter is trainable.
        :param: trainable_tau: bool; if True, tau parameter is trainable.
        :param: Vs_init_val: float; initial value for Vs.
        :param: tau_init_val: float; initial value for tau.
        :param: kernel_init_val: float; initial value for the kernel of the linear regression model (it is used only if Vs or
                                \tau are approximated with a linear regression model.
        :param: bias_init_val: float; initial value for the bias of the linear regression model (it is used only if Vs or
                                \tau are approximated with a linear regression model.
        """

        super(EulerMethod, self).__init__()
        self._full_unrolling = full_unrolling
        self._n_steps = n_steps
        self._h_f = tf.constant([h_f], dtype=tf.float32)
        self._Vs_input_shape = Vs_input_shape
        self._tau_input_shape = tau_input_shape
        self._create_trainable_params(Vs_input_shape=Vs_input_shape,
                                      tau_input_shape=tau_input_shape,
                                      trainable_Vs=trainable_Vs,
                                      trainable_tau=trainable_tau,
                                      Vs_init_val=Vs_init_val,
                                      tau_init_val=tau_init_val,
                                      kernel_init_val=kernel_init_val,
                                      bias_init_val=bias_init_val)
        assert isinstance(self._trainable_params, list), "trainable parameters must be a list"

    @property
    def full_unrolling(self) -> bool:
        return self._full_unrolling

    @full_unrolling.setter
    def full_unrolling(self, value: bool):
        self._full_unrolling = value

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @n_steps.setter
    def n_steps(self, value: int):
        self._n_steps = value

    @property
    def h_f(self) -> float:
        return self._h_f

    @h_f.setter
    def h_f(self, value: float):
        self._h_f = value

    def _create_params(self,
                       input_shape: Tuple[int],
                       trainable: bool,
                       init_val: float,
                       kernel_init_val: float,
                       bias_init_val: float):
        """
        Create trainable parameter of a linear regression model to approximate a parameter of the UODE.
        :param input_shape: tuple of int; the input shape of the linear regression model; if a trainable parameter is
                                          used then None is expected.
        :param trainable: bool; True if the parameter is trainable, False otherwise.
        :param init_val: float; the inital value of the parameter; this is only used if the parameter to estimate is a
                                single variable.
        kernel_init_val: float; initial value for the kernel of the linear regression model (the ODE parameter is
                                approximated with a linear regression model).
        bias_init_val: float; initial value for the bias of the linear regression model (the ODE parameter is
                                approximated with a linear regression model).
        :return:
        """

        # The parameter of the ODE is a single variable
        if input_shape is None:
            # The user must specify an initial value
            assert init_val is not None, "An initial value is required if the ODE parameter is a single parameter"

            # Create the variable
            param = tf.Variable([init_val], trainable=trainable, dtype=tf.float32)

            # If it is trainable, keep track of it
            if trainable:
                self._trainable_params.append(param)

            self._all_variables.append(param)
        # The parameter of the ODE is linear regression model
        else:

            # An initial value for both the kernel and the bias must initialized
            assert kernel_init_val is not None, \
                "A kernel initial value is required if the parameter is a linear regression model"
            assert bias_init_val is not None, \
                "A bias initial value is required if the parameter is a linear regression model"

            # Build the linear regression model
            param = build_linear_regressor(input_shape=input_shape,
                                           kernel=kernel_init_val,
                                           bias=bias_init_val,
                                           trainable=trainable)

            # Keep track of the trainable parameters
            self._trainable_params = self._trainable_params + param.trainable_variables

        return param

    def _create_trainable_params(self,
                                 Vs_input_shape: Tuple[int],
                                 tau_input_shape: Tuple[int],
                                 trainable_Vs: bool,
                                 trainable_tau: bool,
                                 Vs_init_val: float,
                                 tau_init_val: float,
                                 kernel_init_val: int,
                                 bias_init_val: int):
        """
        Create the trainable parameters.
        :param: Vs_input_shape: tuple; if None, the Vs parameter of the ODE is approximated with single parameters; if
                               specified, input shape is used to build the linear regression model.
        :param: tau_input_shape: tuple; if None, the \tau parameter of the ODE is approximated with single parameters;
                                        if specified, input shape is used to build the linear regression model.
        :param: trainable_Vs: bool; if True, Vs parameter is trainable.
        :param: trainable_tau: bool; if True, tau parameter is trainable.
        :param: Vs_init_val: float; initial value for Vs.
        :param: tau_init_val: float; initial value for tau.
        :param: kernel_init_val: float; initial value for the kernel of the linear regression model (it is used only if
                                        Vs or \tau are approximated with a linear regression model.
        :param: bias_init_val: float; initial value for the bias of the linear regression model (it is used only if Vs
                                      or \tau are approximated with a linear regression model.
        :return:
        """

        # Keep track of both the only trainable and all the variables
        self._trainable_params = list()
        self._all_variables = list()

        # Create the Vs parameter
        self.Vs = self._create_params(input_shape=Vs_input_shape,
                                      trainable=trainable_Vs,
                                      init_val=Vs_init_val,
                                      kernel_init_val=kernel_init_val,
                                      bias_init_val=bias_init_val)

        # Create the tau parameter
        self.tau = self._create_params(input_shape=tau_input_shape,
                                       trainable=trainable_tau,
                                       init_val=tau_init_val,
                                       kernel_init_val=kernel_init_val,
                                       bias_init_val=bias_init_val)

    def _differential_equation(self, current_val: tf.Tensor, inputs: tf.Tensor) -> tf.Tensor:
        """
        Formulation of the first order differential equation.
        :param current_val: tf.Tensor; current value of Euler method loop.
        :param inputs: tf.Tensor; input values for the parameters approximators.
        :return: tf.Tensor; next value of the Euler method loop.
        """

        if self._Vs_input_shape is not None:
            assert inputs is not None, "Vs is a linear regression model and requires the input tensors"
            Vs = self.Vs(inputs)
        else:
            Vs = self.Vs
        if self._tau_input_shape is not None:
            assert inputs is not None, "tau is a linear regression model and requires the input tensors"
            tau = self.tau(inputs)
        else:
            tau = self.tau

        dv_dt = (Vs - current_val) / tau
        return dv_dt

    def get_trainable_variables(self) -> List[tf.Variable]:
        """
        Get the list with the trainable parameters.
        :return:
        """
        return self._trainable_params

    def get_weights(self) -> List[tf.Variable]:
        """
        Get all the weights of the graph.
        """
        return self._all_variables

    def set_weights(self, weights: List[tf.Variable]):
        """
        Set the weights given as input.
        :param weights: list of tf.Variable; the new set of weights.
        :return:
        """
        assert len(weights) == len(self._all_variables), "The number of weights is different"

        for current_weight, new_weight in zip(self._all_variables, weights):
            assert current_weight.name == new_weight.name, "Name of the weights must be the same"
            assert current_weight.shape == new_weight.shape, "Shape of the weights must be the same"
            current_weight.assign(new_weight.value())

    def forward(self, init_v: np.array, inputs: np.array) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute the approximate solution of the first order differential equation with the Euler method.
        :param init_v: np.array; initial values.
        :param inputs: numpy.array; numpy array of shape (n_samples, input_dim) with the input features.
        :return: tf.Tensor, tf.Tensor; the approximated solution of the differential equation for the final and
                                         for all timesteps.
        """

        init_v = tf.cast(init_v, dtype=tf.float32)
        next_v = init_v
        values = next_v

        h_e = self.h_f / self._n_steps

        # Euler method loop
        for i in range(self._n_steps):
            next_v = next_v + self._differential_equation(next_v, inputs) * h_e
            values = tf.concat([values, next_v], axis=1)

        return next_v, values

    def get_tau(self, inputs: np.array) -> float:
        """
        Method to get the estimated tau.
        :param inputs; np.array of shape (1, number of features).
        :return: float; the estimated tau.
        """
        if isinstance(self.tau, tf.Variable):
            return self.tau.numpy().item()
        else:
            return self.tau(inputs).numpy()

    def get_Vs(self, inputs: np.array) -> float:
        """
        Method to get the estimated final temperature/voltage.
        :param inputs; np.array of shape (1, number of features).
        :return: float; the estimated Vs.
        """
        if isinstance(self.Vs, tf.Variable):
            return self.Vs.numpy().item()
        else:
            return self.Vs(inputs).numpy()

    def save(self, savepath: str):
        """
        Save the models of the differential equations parameters.
        :param savepath: string; where the model are saved to.
        :return:
        """
        raise NotImplementedError()

    def load(self, loadpath: str):
        raise NotImplementedError()

########################################################################################################################


class StochasticEulerMethod:
    """
    TensorFlow graph implementing the Euler method to solve first order differential equation learning some
    unknown stochastic parameters.
    """

    def __init__(self,
                 full_unrolling: bool,
                 h_f: float,
                 n_steps: int,
                 tau_input_shape: Tuple = None,
                 init_tau_mean: int = 1,
                 trainable_tau_mean: bool = True,
                 init_Vs_mean: int = 1,
                 trainable_Vs_mean: bool = True,
                 init_tau_sigma: int = 0,
                 trainable_tau_sigma: bool = True,
                 init_Vs_sigma: int = 0,
                 trainable_Vs_sigma: bool = True):
        """
        Constructor of the class.
        full_unrolling: bool; if True, the Euler method is unrolled for the full time horizon; otherise, we
                                      expect a dataset with pairs of consecutive measurements.
        h_f: float; data sampling frequency, t_k - t_{k+1}.
        n_steps: int; number of simulation steps.
        init_tau_mean: float; initial value for the mean of the time constant.
        trainable_tau_mean: bool; True if the mean of the time constant is trainable.
        init_Vs_mean: float; initial value for the mean of the final voltage.
        trainable_Vs_mean: bool; True if the mean of the final voltage is trainable.
        init_tau_sigma: float; initial value for the std dev of the time constant.
        trainable_tau_sigma: float; True if the std dev of the time constant is trainable.
        init_Vs_sigma: float; initial value for the std dev of the final voltage.
        trainable_Vs_sigma: bool; True if the std dev of the final voltage is trainable.
        """

        super(StochasticEulerMethod, self).__init__()
        self._full_unrolling = full_unrolling
        self.h_f = tf.constant([h_f], dtype=tf.float32)
        self.n_steps = n_steps
        self._create_trainable_params(init_tau_mean=init_tau_mean,
                                      tau_input_shape=tau_input_shape,
                                      trainable_tau_mean=trainable_tau_mean,
                                      init_Vs_mean=init_Vs_mean,
                                      trainable_Vs_mean=trainable_Vs_mean,
                                      init_tau_sigma=init_tau_sigma,
                                      trainable_tau_sigma=trainable_tau_sigma,
                                      init_Vs_sigma=init_Vs_sigma,
                                      trainable_Vs_sigma=trainable_Vs_sigma)
        assert isinstance(self._trainable_params, list), "trainable parameters must be a list"

    @property
    def full_unrolling(self) -> bool:
        return self._full_unrolling

    @full_unrolling.setter
    def full_unrolling(self, value: bool):
        self._full_unrolling = value

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @n_steps.setter
    def n_steps(self, value: int):
        self._n_steps = value

    @property
    def h_f(self) -> float:
        return self._h_f

    @h_f.setter
    def h_f(self, value: float):
        self._h_f = value

    def _create_trainable_params(self,
                                 tau_input_shape,
                                 init_tau_mean: float,
                                 trainable_tau_mean: bool,
                                 init_Vs_mean: float,
                                 trainable_Vs_mean: bool,
                                 init_tau_sigma: float,
                                 trainable_tau_sigma: bool,
                                 init_Vs_sigma: float,
                                 trainable_Vs_sigma: bool):
        """
        Create the trainable parameters.
        :param: full_unrolling: bool; if True, the Euler method is unrolled for the full time horizon; otherwise, we
                                      expect a dataset with pairs of consecutive measurements.
        init_tau_mean: float; initial value for the mean of the time constant.
        trainable_tau_mean: bool; True if the mean of the time constant is trainable.
        init_Vs_mean: float; initial value for the mean of the final voltage.
        trainable_Vs_mean: bool; True if the mean of the final voltage is trainable.
        init_tau_sigma: float; initial value for the std dev of the time constant.
        trainable_tau_sigma: float; True if the std dev of the time constant is trainable.
        init_Vs_sigma: float; initial value for the std dev of the final voltage.
        trainable_Vs_sigma: bool; True if the std dev of the final voltage is trainable.
        :return:
        """

        self._all_variables = list()

        if tau_input_shape is None:
            self.tau_mean = tf.Variable([float(init_tau_mean)],
                                        name="tau_mean",
                                        trainable=trainable_tau_mean)
            self._all_variables.append(self.tau_mean)
            self.tau_sigma = tf.Variable([float(init_tau_sigma)],
                                         name="tau_sigma",
                                         trainable=trainable_tau_sigma)
            self._all_variables.append(self.tau_sigma)
        else:
            self.tau_mean = build_linear_regressor(input_shape=tau_input_shape,
                                                   kernel=init_tau_mean,
                                                   bias=0,
                                                   trainable=trainable_tau_mean)

            self._all_variables = self._all_variables + self.tau_mean.trainable_variables

            self.tau_sigma = build_linear_regressor(input_shape=tau_input_shape,
                                                    kernel=init_tau_sigma,
                                                    bias=0,
                                                    trainable=trainable_tau_sigma)

            self._all_variables = self._all_variables + self.tau_sigma.trainable_variables

        self.Vs_mean = tf.Variable([float(init_Vs_mean)],
                                   name="Vs_mean",
                                   trainable=trainable_Vs_mean)
        self._all_variables.append(self.Vs_mean)
        self.Vs_sigma = tf.Variable([float(init_Vs_sigma)],
                                    name="Vs_sigma",
                                    trainable=trainable_Vs_sigma)
        self._all_variables.append(self.Vs_sigma)

        self._trainable_params = []
        for variable in self._all_variables:
            if variable.trainable:
                self._trainable_params.append(variable)

    @staticmethod
    def _differential_equation(current_val: tf.Tensor,
                               sampled_Vs: tf.Tensor,
                               sampled_tau: tf.Tensor) -> tf.Tensor:
        """
        Formulation of the first order differential equation.
        :param current_val: tf.Tensor; current value of Euler method loop.
        :return: tf.Tensor; next value of the Euler method loop.
        """

        dv_dt = (sampled_Vs - current_val) / sampled_tau

        return dv_dt

    def get_trainable_variables(self) -> List[tf.Variable]:
        """
        Get the list with the trainable parameters.
        :return:
        """
        return self._trainable_params

    def get_weights(self) -> List[tf.Variable]:
        """
        Get all the weights of the graph.
        """
        return self._all_variables

    def set_weights(self, weights: List[tf.Variable]):
        """
        Set the weights given as input.
        :param weights: list of tf.Variable; the new set of weights.
        :return:
        """
        assert len(weights) == len(self._all_variables), "The number of weights is different"

        for current_weight, new_weight in zip(self._all_variables, weights):
            assert current_weight.name == new_weight.name, "Name of the weights must be the same"
            assert current_weight.shape == new_weight.shape, "Shape of the weights must be the same"
            current_weight.assign(new_weight.value())

    # FIXME: the general version of the UODE is still not supported
    def forward(self, init_v: np.array,
                inputs: np.array = None,
                sample_every_step: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute the approximate solution of the first order differential equation with the Euler method.
        :param init_v: np.array; the initial values.
        :return: tf.Tensor, tf.Tensor; the approximated solution of the differential equation at each timestep and the
                                       final value.
        """

        init_v = tf.cast(init_v, dtype=tf.float32)
        next_v = init_v
        values = next_v

        h_e = self.h_f / self.n_steps

        # Euler method loop
        for i in range(self.n_steps):
            if i == 0 or sample_every_step:
                # FIXME: extend logprob to Vs
                epsilon = tf.random.normal(shape=self.Vs_mean.shape, mean=0, stddev=1)
                sampled_Vs = self.Vs_mean + self.Vs_sigma * epsilon

                if inputs is not None:
                    sampled_tau = tf.math.exp(self.tau_mean(inputs) + self.tau_sigma(inputs) * epsilon)
                else:
                    sampled_tau = tf.math.exp(self.tau_mean + self.tau_sigma * epsilon)

            next_v = next_v + self._differential_equation(current_val=next_v,
                                                          sampled_Vs=sampled_Vs,
                                                          sampled_tau=sampled_tau) * h_e
            values = tf.concat([values, next_v], axis=1)

        return next_v, values

    def get_tau(self, inputs=None) -> Tuple[float, float]:
        """
        Method to get the estimated mean and sigma of tau.
        :return: float, float; the estimated mean and sigma of tau.
        """

        if inputs is None:
            tau_mean = self.tau_mean.numpy().item()
            tau_sigma = self.tau_sigma.numpy().item()
        else:
            tau_mean = self.tau_mean(inputs).numpy()
            tau_sigma = self.tau_sigma(inputs).numpy()

        return tau_mean, tau_sigma

    # FIXME: support for inputs
    def get_Vs(self, inputs=None) -> Tuple[float, float]:
        """
        Method to get the estimated mean and sigma of Vs.
        :return: float, float; the estimated mean and sigma of Vs.
        """

        Vs_mean = self.Vs_mean.numpy().item()
        Vs_sigma = self.Vs_sigma.numpy().item()

        return Vs_mean, Vs_sigma

########################################################################################################################


# FIXME: the method should be updated to support Eager mode disabled
# @tf.function
def train_step(model: Union[EulerMethod, StochasticEulerMethod],
               init_val: float,
               t_vals: tf.Tensor,
               inputs: np.array,
               loss_fn: tf.keras.losses.Loss,
               optimizer: tf.keras.optimizers.Optimizer,
               train_metric: tf.keras.metrics.Metric,
               axis: int = None):
    """
    Single training step.
    :param model: EulerMethod; TensorFlow implementation of the Euler method.
    :param init_val: float; the initial value for the Euler method.
    :param t_vals: numpy.array; target values for each timestep.
    :param inputs: numpy.array; input features for each timestep.
    :param loss_fn: tf.keras.losses; loss function object.
    :param optimizer: tf.keras.optimizers; optimizer object.
    :param train_metric: tf.keras.metrics; metrics used to aggregate loss values.
    :param axis: int; if not None, when computing the loss we will consider the predicted and target values on the only
                      specified axis.
    :return:
    """

    with tf.GradientTape() as tape:
        next_val, all_vals = model.forward(init_v=init_val, inputs=inputs)

        if model.full_unrolling:
            preds = all_vals
            # FIXME: bad code here
            if axis is not None:
                preds = tf.experimental.numpy.swapaxes(preds, 1, 2)
        else:
            preds = next_val

        if axis is not None:
            # FIXME: this should not be hardcoded
            preds = tf.squeeze(preds)
            t_vals = t_vals[:, axis]
            preds = preds[:, axis]

        assert t_vals.shape == preds.shape, "Target values and predictions must have the same dimension"

        loss = loss_fn(t_vals, preds)

        train_metric(loss)

    # Parameters optimization
    trainable_vars = model.get_trainable_variables()
    watched_vars = [var for var in tape.watched_variables()]
    assert trainable_vars == watched_vars, "Missing gradient for some variables"
    gradients = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))

########################################################################################################################


# FIXME: the method should be updated to support Eager mode disabled
# @tf.function
def val_step(model: Union[EulerMethod, StochasticEulerMethod],
             init_val: float,
             t_vals: np.array,
             inputs: np.array,
             loss_fn: tf.keras.losses.Loss,
             axis: int):
    """
    Single training step.
    :param model: EulerMethod; TensorFlow implementation of the Euler method.
    :param t_vals: numpy.array; target values for each timestep.
    :param inputs: numpy.array; input features for each timestep.
    :param loss_fn: tf.keras.losses; loss function object.
    :return:
    """

    next_val, all_vals = model.forward(init_v=init_val, inputs=inputs)

    # FIXME: repeated code from train_step
    if model.full_unrolling:
        preds = all_vals
        # FIXME: bad code here
        if axis is not None:
            preds = tf.experimental.numpy.swapaxes(preds, 1, 2)
    else:
        preds = next_val

    if axis is not None:
        # FIXME: this should not be hardcoded
        preds = tf.squeeze(preds)
        t_vals = t_vals[:, axis]
        preds = preds[:, axis]

    assert t_vals.shape == preds.shape, "Target values and predictions must have the same dimension"

    loss = loss_fn(t_vals, preds)
    loss = loss.numpy()

    return loss

########################################################################################################################


def train(model: Union[EulerMethod, StochasticEulerMethod],
          init_vals: np.array,
          target_vals: np.array,
          inputs: np.array,
          epochs: int,
          loss_fn: tf.keras.losses.Loss,
          optimizer: tf.keras.optimizers.Optimizer,
          learning_rate: float,
          train_metric: tf.keras.metrics.Metric,
          patience: int,
          axis: int = None,
          batch_size: int = 8,
          delta_loss: float = 0.001,
          verbose: int = 0):
    """
    Custom training loop.
    :param model: EulerMethod; TensorFlow implementation of the Euler method.
    :param init_vals: numpy.array of shape (n_samples, 1); initial value of ODE.
    :param target_vals: numpy.array of shape (n_samples, 1); the available values for each step of the Euler method.
    :param inputs: numpy.array of shape (n_samples, n_features); the input features for the differential equation
                                                                 parameters approximator.
    :param epochs: int; number of training epochs.
    :param loss_fn: tf.keras.losses; loss function object.
    :param optimizer: optimizer: tf.keras.optimizers; optimizer object.
    :param learning_rate: float; the learning rate of the optimizer.
    :param train_metric: tf.keras.metrics; metrics used to aggregate loss values.
    :param patience: int; patience for early stopping.
    :param axis: int; if not None, when computing the loss we will consider the predicted and target values on the only
                      specified axis.
    :param batch_size: int; batch size.
    :param delta_loss: float; the minimum decrease of the loss required to continue the training.
    :param verbose: int; output verbosity.
    :return:
    """

    loss_fn = tf.keras.losses.get(loss_fn)
    optimizer = tf.keras.optimizers.get(optimizer)
    optimizer.lr.assign(learning_rate)
    train_metric = tf.keras.metrics.get(train_metric)

    assert isinstance(loss_fn, tf.keras.losses.Loss)
    assert isinstance(optimizer, tf.keras.optimizers.Optimizer)
    assert isinstance(train_metric, tf.keras.metrics.Metric)

    # Check that verbosity is allowed
    assert verbose >= 0, "verbose must be greater than or equal to 0"

    # Initialize early stopping variables
    best_loss = np.inf
    best_weights = model.get_weights()
    model.set_weights(best_weights)

    # Save history
    history = dict()
    history['Loss'] = list()
    history['Val loss'] = list()
    history['Epoch'] = list()
    history['Epoch time'] = list()

    if not model.full_unrolling:
        tf_dataset = tf.data.Dataset.from_tensor_slices((init_vals,
                                                         target_vals,
                                                         inputs)).shuffle(10000).batch(batch_size)

    # Keep track of the current time
    train_start = time.time()

    # Training loop
    for epoch in range(epochs):
        start = time.time()

        if not model.full_unrolling:
            for batch_init_v, batch_target_v, batch_inputs in tf_dataset:
                train_step(model=model,
                           init_val=batch_init_v,
                           t_vals=batch_target_v,
                           inputs=batch_inputs,
                           loss_fn=loss_fn,
                           optimizer=optimizer,
                           train_metric=train_metric,
                           axis=axis)
        else:
            train_step(model=model,
                       init_val=init_vals,
                       t_vals=target_vals,
                       inputs=inputs,
                       loss_fn=loss_fn,
                       optimizer=optimizer,
                       train_metric=train_metric,
                       axis=axis)
        end = time.time()

        # Early stopping
        current_epoch_loss = train_metric.result()

        # Compute the validation loss
        val_loss = val_step(model=model,
                            init_val=init_vals,
                            t_vals=target_vals,
                            inputs=inputs,
                            loss_fn=loss_fn,
                            axis=axis)

        if (epoch + 1) % patience == 0:
            if best_loss < val_loss:
                print('\nEarly stopping')
                print('Restoring best weights')
                model.set_weights(best_weights)
                break
            elif best_loss - val_loss < delta_loss:
                print('\nEarly stopping')
                break

        # Update early stopping variables
        best_loss = val_loss
        best_weights = model.get_weights()

        if verbose >= 1 and (epoch + 1) % verbose == 0:

            history['Loss'].append(current_epoch_loss.numpy())
            history['Val loss'].append(val_loss)
            history['Epoch'].append(epoch)
            history['Epoch time'].append(end - start)

            print_string = "Epoch: {} | Val loss: {}".format(epoch+1, val_loss)
            print(print_string)

        # Reset aggregated loss values at the end of each epoch
        train_metric.reset_states()

        if current_epoch_loss == np.nan:
            break

    train_stop = time.time()

    history['Training duration'] = train_stop - train_start
    history['Loss'].append(current_epoch_loss.numpy())
    history['Val loss'].append(val_loss)
    history['Epoch'].append(epoch)

    print_string = "Epoch: {} | Val loss: {}".format(epoch + 1, val_loss)
    print(print_string)

    return history


########################################################################################################################


def build_linear_regressor(input_shape: Tuple[int],
                           kernel: float,
                           bias: float,
                           trainable: bool) -> tf.keras.Model:
    """
    Method to build the function approximator (like MLP) to estimate the parameters of the ODE.
    :param input_shape: tuple of int; shape of the input.
    :param kernel: float; kernel value.
    :param bias: float; bias value.
    :return: tf.keras.Model; the model of the function approximator.
    """

    model = Sequential()
    model.add(InputLayer(input_shape))
    model.add(Dense(units=1,
                    activation=None,
                    kernel_initializer=tf.keras.initializers.Constant(kernel),
                    bias_initializer=tf.keras.initializers.Constant(bias),
                    use_bias=False,
                    trainable=trainable))

    return model


########################################################################################################################


