# Author: Mattia Silvestri

"""
    Utility methods.
"""

import pandas as pd
import numpy as np
import random
import tensorflow as tf
import pickle
import json
from typing import List, Callable, Tuple, Union, Dict

########################################################################################################################


def load_from_pickle(loadpath: str):
    """
    Load results from a .pkl file.
    :param loadpath: str; the loadpath of a .pkl file.
    :return: a general object that is pickle-serializable.
    """
    with open(loadpath, 'rb') as pkl_file:
        res = pickle.load(pkl_file)

    return res


########################################################################################################################


def load_from_json(loadpath: str) -> Dict:
    """

    :param loadpath: str; the loadpath of a JSON file.
    :return: dict; the content of the JSON file.
    """
    with open(loadpath, 'r') as json_file:
        res = json.load(json_file)

    return res

########################################################################################################################


def save_to_pickle(savepath: str, res):
    """
    Save a pickle-seriablize object to file.
    :param savepath: str; where the object is saved to.
    :param res: pickle serializable object; the object to save.
    :return:
    """
    with open(savepath, 'wb') as pkl_file:
        pickle.dump(res, pkl_file)

########################################################################################################################


def save_to_json(savepath: str, config: dict):
    """
    Save a configuration file in JSON format.
    :param savepath: str; where the configuration file is saved to.
    :param config: dict; the configuration file as dictionary.
    :return:
    """
    with open(savepath, 'w') as json_file:
        json.dump(config, json_file)

########################################################################################################################


def min_max_scaler(val: np.ndarray,
                   lower: float,
                   upper: float,
                   rescaled_max: float = 1,
                   rescaled_min: float = 0):
    """
    Rescale the array values in the range [0,1].
    :param val: np.ndarray; the value to rescale.
    :param lower: float; the lower bound of the original values interval.
    :param upper: float; the lower bound of the original values interval.
    :param rescaled_max: float; the new upper bound of the original values interval.
    :param rescaled_min: float; the new lower bound of the original values interval.
    :return:
    """
    val_std = (val - lower) / (upper - lower)
    val_scaled = val_std * (rescaled_max - rescaled_min) + rescaled_min

    return val_scaled

########################################################################################################################


def rc_circuit(timesteps: List[int],
               tau: List[float],
               init_v: float,
               Vs: float) -> List:
    """
    RC circuit function.
    :param timesteps: list of int; the timesteps at which the RC function is evaluated.
    :param tau: list of float; time constant of the circuit.
    :param init_v: float; initial voltage of the circuit.
    :param Vs: float; capacitor voltage (final voltage).
    :return: list; voltage value at each timestep.
    """

    values = list()

    values.append(init_v)

    for t, tau_t in zip(timesteps, tau):
        current_val = (init_v - Vs) * np.exp(-t / tau_t) + Vs
        values.append(current_val)

    return values


########################################################################################################################


def linear_rel_synthetic_data(linear_coeff: float,
                              offset: float,
                              Vs: float,
                              init_v: float,
                              n_steps: int,
                              n_samples: int = 100,
                              eoh: int = 5):
    """
    Create synthetic data in which tau is approximated by a linear relationship.
    :param linear_coeff: float; the linear coefficient of the relationship.
    :param offset: float; the offset of the relationship.
    :param Vs: float; the final capacitor voltage.
    :param init_v: float; the initial capacitor voltage.
    :param n_steps: int; Euler method number of steps used when generating data.
    :param n_samples: int; the number of samples to be generated.
    :param eoh: int; upper bound of the time interval (as unit of \tau).
    :return:
    """
    np.random.seed(1)

    # The input feature from which tau is generated
    x_t = np.zeros(shape=(n_samples+1,))
    x_val = 1

    # Periodically increase the input
    for i in range(n_samples):
        x_val += random.uniform(0, 1)
        x_t[i] = x_val

    tau_x_t = linear_coeff * x_t + offset

    # Generate n_samples data points from 0 to 5*tau (time required to reach the 98% of the final voltage value)
    # The sample step size must be updated accordingly
    max_tau = np.max(tau_x_t)
    eoh = eoh * max_tau

    timesteps, sampling_freq = np.linspace(start=0, stop=eoh, endpoint=True, num=n_samples+1, retstep=True)

    values = [init_v]
    current_v = init_v
    for t, current_tau in zip(timesteps[1:], tau_x_t):
        rc_fun = lambda y: (Vs - y) / current_tau
        current_v, _ = euler_method(f=rc_fun, f0=current_v, t_n=sampling_freq, n_steps=n_steps)
        values.append(current_v)

    values = np.asarray(values)

    train_data = pd.DataFrame(index=timesteps, columns=['x', 'Tau', 'Voltage', 'Time'])
    train_data['x'] = x_t
    train_data['Tau'] = tau_x_t
    train_data['Voltage'] = values
    train_data['Time'] = timesteps

    """axes = train_data.plot(subplots=True)
    axes[2].axhline(Vs, linestyle='--', color='red')
    plt.show()"""

    return train_data, sampling_freq

########################################################################################################################


def euler_method(f: Callable,
                 f0: [float, np.ndarray],
                 t_n: float,
                 n_steps: int) -> Tuple[float, np.array]:
    """
    Implementation of the euler method.
    :param f: function; function object representing the first order derivative.
    :param f0: float; initial value.
    :param t_n: float; end time.
    :param n_steps: int; the number of steps performed by the Euler method.
    :return: float, np.array; state value at the end of the simulation and all the intermediate results.
    """

    y = f0
    if isinstance(y, np.ndarray):
        y_t = [y.copy()]
    else:
        y_t = [y]

    # FIXME: find a more numerically stable approach
    h = t_n / n_steps

    for n in range(n_steps):
        if isinstance(f, tuple):
            current_y = y.copy()
            for idx in range(len(f)):
                y[idx] = y[idx] + f[idx](*current_y) * h
        else:
            y = y + f(y) * h

        y_t.append(y.copy())

    return y, np.asarray(y_t, dtype=np.float32)

########################################################################################################################


def rc_derivative(tau: float, Vs: float) -> Callable:
    """
    Lambda expression to get the first order derivative of the RC-circuit.
    :param tau: float; tau of the RC-circuit.
    :param Vs: float; capacitor voltage.
    :return: lambda expression to compute the first order derivative of the RC-circuit.
    """
    return lambda y: (Vs - y) / tau

########################################################################################################################


def exp_derivative():
    """
    Lambda expression to get the first order derivative of the exponential function.
    :return:
    """
    return lambda y: y


########################################################################################################################


def create_training_set_with_timesteps(loadpath: Union[str, pd.DataFrame],
                                       target_column: str,
                                       input_columns: List[str],
                                       param_column: str) -> pd.DataFrame:
    """
    Create the training set from a dynamic system curve.
    :param loadpath: string; the path where data are loaded from.
    :param target_column: string; the name of the target column.
    :param input_columns: list of string; the name of the input features.
    :param param_column: string; the name of the parameter of the differential equation.
    :return: a pandas.Dataframe, with the following columns: the parameter of the differential equation to estimate,
             the current and next value of the state variable and the input features.
    """

    assert isinstance(loadpath, str) or isinstance(loadpath, pd.DataFrame), \
        "loadpath must be a string or a pandas.Dataframe"

    if isinstance(loadpath, str):
        df = pd.read_csv(loadpath, index_col=0)
    else:
        df = loadpath

    ramps = df[target_column].values
    if not isinstance(target_column, list):
        columns = [target_column]
        additional_cols = f'Next_{target_column}'
        columns.append(additional_cols)
    else:
        columns = target_column
        additional_cols = [f'Next_{c}' for c in target_column]
        columns = columns + additional_cols

    if param_column is not None:
        columns.append(param_column)

    if input_columns is not None:
        columns = columns + input_columns

    tr_set = pd.DataFrame(index=df.index[:-1],
                          columns=columns,
                          dtype=np.float32)

    tr_set[target_column] = ramps[:-1]
    tr_set[additional_cols] = ramps[1:]

    if input_columns is not None:
        tr_set[input_columns] = df[input_columns].iloc[:-1].values

    if param_column is not None:
        tr_set[param_column] = df[param_column].iloc[:-1].values

    return tr_set

########################################################################################################################


def config_gpu(no_gpu):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        if no_gpu:
            tf.config.set_visible_devices([], 'GPU')
        else:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])





