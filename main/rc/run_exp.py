import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings
import pickle
import json
import tensorflow as tf
import seaborn as sns
import argparse
from tqdm import tqdm
from typing import List, Tuple

from helpers.rc.utility import rc_circuit, create_training_set_with_timesteps, linear_rel_synthetic_data
from models.rc.custom_models import EulerMethod, train
from helpers.rc.utility import save_to_json, save_to_pickle, config_gpu

########################################################################################################################

sns.set_style('darkgrid')
DEC_SEQ_STR = 'decomposed-seq'
FULL_UNROLL_STR = 'full-unrolling'

########################################################################################################################


def experiment_0(Vs_bounds: Tuple[int],
                 tau_bounds: Tuple[int],
                 num_runs: int,
                 n_samples: int,
                 savepath: str = None,
                 seed: int = 0):
    """
    In this experiment, we test whether training the UODE by decomposing a ramp in sequence of consecutive measurements
    affects the performance.
    :param Vs_bounds: tuple of 2 int; lower and upper bounds for Vs.
    :param tau_bounds: tuple of 2 int; lower and upper bounds for \tau.
    :param n_samples: int; number of point measurements of the dynamical system to generate.
    :param savepath: str; where the results are saved to; if None, nothing will be saved.
    :param seed: int; set the seed for reproducibility.
    :return:
    """

    # FIXME: the initial value is hardcoded to 1

    # Set the seed before creating the instances
    np.random.seed(seed)

    # Create the ground truth values
    Vs_list = np.random.randint(low=Vs_bounds[0], high=Vs_bounds[1], size=num_runs)
    tau_val_list = np.random.randint(low=tau_bounds[0], high=tau_bounds[1], size=num_runs)

    # Warn the user if nothing will be saved
    if savepath is None:
        warnings.warn("You did not specify a savepath; results will not be saved.")

    # Dictionary with the parameters for the Euler method object.
    euler_method_params = dict()
    euler_method_params['Vs_init_val'] = 1.
    euler_method_params['tau_init_val'] = 1.
    euler_method_params['trainable_Vs'] = True
    euler_method_params['trainable_tau'] = True

    # Dictionary with the parameters for the training routine
    train_params = dict()
    train_params['epochs'] = 10000
    train_params['loss_fn'] = 'MeanSquaredError'
    train_params['optimizer'] = 'Adam'
    train_params['learning_rate'] = 0.1
    train_params['train_metric'] = 'Mean'
    train_params['patience'] = 50
    train_params['batch_size'] = 8
    train_params['delta_loss'] = 0.0001

    # Dictionary with the data generation information
    data_gen_params = dict()
    data_gen_params['Vs_lower'] = Vs_bounds[0]
    data_gen_params['Vs_upper'] = Vs_bounds[1]
    data_gen_params['tau_lower'] = tau_bounds[0]
    data_gen_params['tau_upper'] = tau_bounds[1]
    data_gen_params['num_runs'] = num_runs

    # Keep track of the results
    full_unrolling_preds_list = list()
    decomposed_seq_preds_list = list()

    # True v_c(t)
    true_values_list = list()

    # Estimated V_s for the two approaches
    full_unrolling_Vs_list = list()
    decomposed_seq_Vs_list = list()

    # Estimated \tau for the two approaches
    full_unrolling_tau_list = list()
    decomposed_seq_tau_list = list()

    # True V_s and \tau
    true_Vs_list = list()
    true_tau_list = list()

    # Training history for the two approaches
    full_unrolling_history_list = list()
    decomposed_seq_history_list = list()

    timesteps_list = list()

    # For each pair V_s-\tau run a training routine
    for Vs, tau_val in tqdm(zip(Vs_list, tau_val_list), desc='Experiment runs', total=len(Vs_list)):

        # Keep track of the true values
        true_Vs_list.append(Vs)
        true_tau_list.append(tau_val)

        # FULL UNROLLING
        print('\nFull unrolling')

        # Data sampling frequency
        h_f = 5 * tau_val / n_samples
        tau = [tau_val] * n_samples

        # Generate one point measurement for each timestep
        timesteps = np.arange(0, 5 * tau_val + h_f, h_f)
        timesteps_list.append(timesteps)

        # Generate the true v_c(t)
        true_values = \
            rc_circuit(timesteps=timesteps[1:],
                       tau=tau,
                       init_v=0,
                       Vs=Vs)
        true_values_list.append(true_values)

        # v_c(0)
        init_val = np.zeros(shape=(1, 1))

        # Add a fake batch dimension to the target values and convert to tf.Tensor
        target_vals = np.expand_dims(true_values, axis=0)
        target_vals = tf.convert_to_tensor(target_vals, dtype=tf.float32)

        # Create the graph
        graph = EulerMethod(full_unrolling=True,
                            # When we use the full unrolling method, we iterate until the end of the time interval
                            h_f=5 * tau_val,
                            Vs_input_shape=None,
                            tau_input_shape=None,
                            # We unroll the Euler method for a number of steps equal to the number of sampling
                            # frequency interval
                            n_steps=n_samples,
                            **euler_method_params)

        # Fit the model
        history =\
            train(model=graph,
                  init_vals=init_val,
                  target_vals=target_vals,
                  inputs=None,
                  verbose=100,
                  **train_params)

        full_unrolling_history_list.append(history)

        # Keep track of the fitted params
        full_unrolling_tau = graph.get_tau(inputs=None)
        full_unrolling_Vs = graph.get_Vs(inputs=None)
        full_unrolling_tau_list.append(full_unrolling_tau)
        full_unrolling_Vs_list.append(full_unrolling_Vs)

        # Get predictions with the fitted parameters
        _, values = graph.forward(init_val, inputs=None)
        full_unrolling_preds = np.squeeze(values)

        # Sanity check
        true_values = np.asarray(true_values)
        assert true_values.shape == full_unrolling_preds.shape
        full_unrolling_preds_list.append(full_unrolling_preds)

        # Compute the RMSE
        full_unrolling_mse = mean_squared_error(true_values, full_unrolling_preds)
        full_unrolling_rmse = np.sqrt(full_unrolling_mse)

        ################################################################################################################

        # DECOMPOSED SEQUENCE TRAINING
        print('\nDecomposed sequence training')

        # Data sampling frequency
        h_f = 5 * tau_val / n_samples
        tau = [tau_val] * n_samples

        # Generate one point measurement for each timestep
        timesteps = np.arange(0, 5 * tau_val + h_f, h_f)
        decomposed_seq_true_values = \
            rc_circuit(timesteps=timesteps[1:],
                       tau=tau,
                       init_v=0,
                       Vs=Vs)

        # Generate a pd.Dataframe with time and v_c(t)
        dataframe = pd.DataFrame(columns=['Timestep', 'Value'])
        dataframe['Timestep'] = timesteps
        dataframe['Value'] = decomposed_seq_true_values

        # Create the training set with pairs of successive measurements
        tr_set = \
            create_training_set_with_timesteps(loadpath=dataframe,
                                               target_column='Value',
                                               input_columns=None,
                                               param_column=None)

        # Add a fake feature dimension to the initial and target values of each pair
        init_val = np.expand_dims(tr_set['Value'].values, axis=1)
        target_vals = np.expand_dims(tr_set['Next_Value'].values, axis=1)

        # Create the graph
        graph = EulerMethod(full_unrolling=False,
                            # When we use the decomposed sequence method, we iterate until the next sampled instant
                            h_f=h_f,
                            Vs_input_shape=None,
                            tau_input_shape=None,
                            n_steps=1,
                            **euler_method_params)

        # Fit the model
        history = \
            train(model=graph,
                  init_vals=init_val,
                  target_vals=target_vals,
                  inputs=None,
                  verbose=100,
                  **train_params)

        decomposed_seq_history_list.append(history)

        # At test time, we full unroll the Euler method otherwise the comparison is not fair
        graph.n_steps = n_samples
        graph.full_unrolling = True
        graph.h_f = 5 * tau_val

        # Keep track of the fitted params
        decomposed_seq_tau = graph.get_tau(inputs=None)
        decomposed_seq_Vs = graph.get_Vs(inputs=None)
        decomposed_seq_tau_list.append(decomposed_seq_tau)
        decomposed_seq_Vs_list.append(decomposed_seq_Vs)

        # Get predictions with the fitted parameters
        init_val = np.zeros(shape=(1, 1))
        _, values = graph.forward(init_val, inputs=None)
        decomposed_seq_preds = np.squeeze(values)

        # Sanity check
        true_values.shape == decomposed_seq_preds.shape
        decomposed_seq_preds_list.append(decomposed_seq_preds)
        np.testing.assert_almost_equal(true_values, decomposed_seq_true_values)

        # Compute the RMSE
        decomposed_seq_mse = mean_squared_error(decomposed_seq_true_values, decomposed_seq_preds)
        decomposed_seq_rmse = np.sqrt(decomposed_seq_mse)

        # Visualize results
        print_string = f'\nFull unrolling estimated tau: {full_unrolling_tau} | '
        print_string += f'Decomposed sequence estimated tau: {decomposed_seq_tau} | '
        print_string += f' | True tau: {tau_val}'
        print(print_string)

        print_string = f'Full unrolling estimated Vs: {full_unrolling_Vs} | '
        print_string += f'Decomposed sequence estimated Vs: {decomposed_seq_Vs} | '
        print_string += f'True Vs: {Vs}'
        print(print_string)

        print_string = f'MSE with full unrolling: {full_unrolling_mse} | '
        print_string += f'MSE with decomposed sequence: {decomposed_seq_mse}'
        print(print_string)

        print_string = f'RMSE with full unrolling: {full_unrolling_rmse} | '
        print_string += f'RMSE with decomposed sequence: {decomposed_seq_rmse}'
        print(print_string)
        print('\n' + '-' * len(print_string) + '\n')

    # Save results and experiments configuration if required
    if savepath is not None:

        # Make the results folder if it does not exist
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        save_to_pickle(savepath=os.path.join(savepath, 'timesteps.pkl'),
                       res=timesteps_list)

        save_to_pickle(savepath=os.path.join(savepath, 'true-Vs.pkl'),
                       res=true_Vs_list)

        true_tau_list = np.asarray(true_tau_list)
        save_to_pickle(savepath=os.path.join(savepath, 'true-tau.pkl'),
                       res=true_tau_list)

        save_to_pickle(savepath=os.path.join(savepath, f'{DEC_SEQ_STR}-Vs.pkl'),
                       res=decomposed_seq_Vs_list)

        save_to_pickle(savepath=os.path.join(savepath, f'{DEC_SEQ_STR}-tau.pkl'),
                       res=decomposed_seq_tau_list)

        save_to_pickle(savepath=os.path.join(savepath, f'{DEC_SEQ_STR}-preds.pkl'),
                       res=decomposed_seq_preds_list)

        save_to_pickle(savepath=os.path.join(savepath, f'true-values.pkl'),
                       res=true_values_list)

        save_to_pickle(savepath=os.path.join(savepath, f'{DEC_SEQ_STR}-history.pkl'),
                       res=decomposed_seq_history_list)

        save_to_pickle(savepath=os.path.join(savepath, f'{FULL_UNROLL_STR}-Vs.pkl'),
                       res=full_unrolling_Vs_list)

        save_to_pickle(savepath=os.path.join(savepath, f'{FULL_UNROLL_STR}-tau.pkl'),
                       res=full_unrolling_tau_list)

        save_to_pickle(savepath=os.path.join(savepath, f'{FULL_UNROLL_STR}-preds.pkl'),
                       res=full_unrolling_preds_list)

        save_to_pickle(savepath=os.path.join(savepath, F'{FULL_UNROLL_STR}-history.pkl'),
                       res=full_unrolling_history_list)

        save_to_json(savepath=os.path.join(savepath, 'euler-method-params.json'),
                     config=euler_method_params)

        save_to_json(savepath=os.path.join(savepath, 'train-params.json'),
                     config=train_params)

        save_to_json(savepath=os.path.join(savepath, 'data_gen_params.json'),
                     config=data_gen_params)

########################################################################################################################


def experiment_1(Vs_bounds: Tuple[int],
                 tau_bounds: Tuple[int],
                 num_runs: int,
                 n_samples: int,
                 savepath: str = None,
                 seed: int = 0):

    """
    With this experiment, we show how the predictions of the UODE with the fitted parameters are more accurate than the
    predictions we could achieve with the true parameters.
    :param Vs_bounds: tuple of 2 int; lower and upper bound for the V_s generation.
    :param tau_bounds: tuple of 2 int; lower and upper bound for the \tau generation.
    :param num_runs: int; number of experiments to run.
    :param n_samples: int; number of point measurements of the dynamical system to generate.
    :param savepath: str; where the results are saved to; if None, nothing will be saved.
    :param seed: int; set the seed for reproducibility.
    :return:
    """

    # Set the seed before creating the instances
    np.random.seed(seed)

    # Create the ground truth values
    Vs_list = np.random.randint(low=Vs_bounds[0], high=Vs_bounds[1], size=num_runs)
    tau_val_list = np.random.randint(low=tau_bounds[0], high=tau_bounds[1], size=num_runs)

    # Warn the user if nothing will be saved
    if savepath is None:
        warnings.warn("You did not specify a savepath; results will not be saved.")

    # Dictionary with the parameters for the Euler method object.
    euler_method_params = dict()
    euler_method_params['n_steps'] = 1
    euler_method_params['Vs_init_val'] = 1.
    euler_method_params['tau_init_val'] = 1.
    euler_method_params['full_unrolling'] = False

    # Keep track of results
    true_params_preds_list = list()
    true_values_list = list()

    # For each pair Vs-\tau run a training routine
    for Vs, tau_val in tqdm(zip(Vs_list, tau_val_list), desc='Experiment runs', total=len(Vs_list)):

        # Data sampling frequency
        h_f = 5 * tau_val / n_samples
        tau = [tau_val] * n_samples

        # Generate one point measurement for each timestep
        timesteps = np.arange(0, 5 * tau_val + h_f, h_f)
        true_values = rc_circuit(timesteps=timesteps[1:],
                                 tau=tau,
                                 init_v=0,
                                 Vs=Vs)

        # Create the fake feature dimension to the initial value
        init_val = np.zeros(shape=(1, 1))

        # Create a new graph using the true parameters values
        graph = EulerMethod(h_f=h_f,
                            full_unrolling=False,
                            Vs_input_shape=None,
                            tau_input_shape=None,
                            trainable_Vs=True,
                            trainable_tau=True,
                            Vs_init_val=Vs,
                            tau_init_val=tau_val,
                            n_steps=euler_method_params['n_steps'])

        true_values = np.asarray(true_values)

        # At test time, we full unroll the Euler method otherwise the comparison is not fair
        graph.n_steps = n_samples
        graph.full_unrolling = True
        graph.h_f = 5 * tau_val

        # Get predictions with the true parameters
        _, values = graph.forward(init_val, inputs=None)
        true_params_preds = np.squeeze(values)

        # Sanity check
        assert true_values.shape == true_params_preds.shape
        true_params_preds_list.append(true_params_preds)

        # Compute RMSE
        true_params_rmse = np.sqrt(np.mean(np.square(true_values - true_params_preds)))

        print_string = f'RMSE with true parameters: {true_params_rmse}'
        print(print_string)

    # Save results and experiments configuration if required
    if savepath is not None:

        # Make the results folder if it does not exist
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        Vs_list = np.asarray(Vs_list)
        save_to_pickle(savepath=os.path.join(savepath, 'true-Vs.pkl'), res=Vs_list)

        tau_val_list = np.asarray(tau_val_list)
        save_to_pickle(savepath=os.path.join(savepath, 'true-tau.pkl'), res=tau_val_list)

        true_params_preds_list = np.asarray(true_params_preds_list)
        save_to_pickle(savepath=os.path.join(savepath, 'true-params-preds.pkl'), res=true_params_preds_list)

        true_values_list = np.asarray(true_values_list)
        save_to_pickle(savepath=os.path.join(savepath, 'true-values.pkl'), res=true_values_list)

        save_to_json(savepath=os.path.join(savepath, 'euler-method-params.json'), config=euler_method_params)

########################################################################################################################


def experiment_2(Vs_bounds: Tuple[int],
                 tau_bounds: Tuple[int],
                 num_runs: int,
                 n_samples: int,
                 n_steps_list: List[int],
                 savepath: str = None,
                 seed: int = 0):
    """
    With this experiment, we show how the estimation of \tau parameters benefits from an increasing in the
    computational complexity of the Euler method. The Vs estimation is still accurate even when the computational
    complexity of the Euler method is low.
    :param Vs_bounds: tuple of 2 int; lower and upper bound for the V_s generation.
    :param tau_bounds: tuple of 2 int; lower and upper bound for the \tau generation.
    :param n_samples: int; number of point measurements of the dynamical system to generate.
    :param n_steps_list: list of int; the list of Euler steps values.
    :param savepath: str; where the results are saved to; if None, nothing will be saved.
    :param seed: int; set the seed for reproducibility.
    :return:
    """

    # Set the seed before creating the instances
    np.random.seed(seed)

    # Create the ground truth values
    Vs_list = np.random.randint(low=Vs_bounds[0], high=Vs_bounds[1], size=num_runs)
    tau_val_list = np.random.randint(low=tau_bounds[0], high=tau_bounds[1], size=num_runs)

    # Warn the user if nothing will be saved
    if savepath is None:
        warnings.warn("You did not specify a savepath; results will not be saved.")

    # Dictionary with the parameters for the Euler method object.
    euler_method_params = dict()
    euler_method_params['tau_init_val'] = 1.
    euler_method_params['full_unrolling'] = False
    euler_method_params['trainable_Vs'] = False
    euler_method_params['trainable_tau'] = True

    # Dictionary with the parameters for the training routine
    train_params = dict()
    train_params['epochs'] = 100000
    train_params['loss_fn'] = 'MeanSquaredError'
    train_params['optimizer'] = 'Adam'
    train_params['learning_rate'] = 0.1
    train_params['train_metric'] = 'Mean'
    train_params['patience'] = 50
    train_params['batch_size'] = 8
    train_params['delta_loss'] = 0.0001

    # Keep track of results for all the runs
    fitted_tau_exp = list()
    true_tau_exp = list()
    fitted_Vs_exp = list()
    true_Vs_exp = list()
    preds_exp = list()
    true_vals_exp = list()
    history_exp = list()

    # For each pair Vs-\tau run a training routine
    for Vs, tau_val in tqdm(zip(Vs_list, tau_val_list), desc='Experiment runs', total=len(Vs_list)):

        # Keep track of the true and fitted \tau for each number of steps value of the Euler method
        fitted_tau_steps = list()
        true_tau_steps = list()
        true_Vs_steps = list()
        fitted_Vs_steps = list()
        preds_steps = list()
        true_vals_steps = list()
        history_steps = list()

        # Compute the sampling frequency
        h_f = 5 * tau_val / n_samples

        tau = [tau_val] * n_samples

        # Generate one point measurement for each timestep
        timesteps = np.arange(0, 5 * tau_val + h_f, h_f)
        true_values = rc_circuit(timesteps=timesteps[1:],
                                 tau=tau,
                                 init_v=0,
                                 Vs=Vs)

        dataframe = pd.DataFrame(columns=['Timestep', 'Value'])
        dataframe['Timestep'] = timesteps
        dataframe['Value'] = true_values

        # Create the training set with pairs of successive measurements
        tr_set = create_training_set_with_timesteps(loadpath=dataframe,
                                                    target_column='Value',
                                                    input_columns=None,
                                                    param_column=None)

        # Add a fake feature dimension to the initial an target values
        init_val = np.expand_dims(tr_set['Value'].values, axis=1)
        target_vals = np.expand_dims(tr_set['Next_Value'].values, axis=1)

        # Fit a model for each number of steps of the Euler method
        for n_steps in n_steps_list:
            graph = EulerMethod(h_f=h_f,
                                n_steps=n_steps,
                                Vs_input_shape=None,
                                tau_input_shape=None,
                                Vs_init_val=Vs,
                                **euler_method_params)

            # Fit the model
            history = \
                train(model=graph,
                      init_vals=init_val,
                      target_vals=target_vals,
                      inputs=None,
                      verbose=100,
                      **train_params)
            history_steps.append(history)

            # Keep track of the fitted and true \tau
            fitted_tau = graph.get_tau(inputs=None)
            fitted_Vs = graph.get_Vs(inputs=None)

            fitted_tau_steps.append(fitted_tau)
            true_tau_steps.append(tau_val)

            fitted_Vs_steps.append(fitted_Vs)
            true_Vs_steps.append(Vs)

            # Get predictions with the fitted parameters
            values, _ = graph.forward(init_val, inputs=None)
            values = np.squeeze(values)
            fitted_params_preds = np.insert(values, 0, 0, axis=0)

            # Sanity check
            true_values = np.asarray(true_values)
            assert true_values.shape == fitted_params_preds.shape
            preds_steps.append(fitted_params_preds)
            true_vals_steps.append(true_values)

            # Visualize results
            print_string = f'\nNumber of steps: {n_steps} | Fitted Tau: {fitted_tau} | True tau: {tau_val}'
            print_string += f' | Fitted Vs: {fitted_Vs} | True Vs: {Vs}'
            print(print_string)
            print('-' * len(print_string))

        # Keep track of the results for each V_s and \tau pair
        fitted_tau_exp.append(fitted_tau_steps)
        true_tau_exp.append(true_tau_steps)
        fitted_Vs_exp.append(fitted_Vs_steps)
        true_Vs_exp.append(true_Vs_steps)
        preds_exp.append(preds_steps)
        true_vals_exp.append(true_vals_steps)
        history_exp.append(history_steps)

    # Convert to array
    fitted_tau_exp = np.asarray(fitted_tau_exp)
    true_tau_exp = np.asarray(true_tau_exp)
    fitted_Vs_exp = np.asarray(fitted_Vs_exp)
    true_Vs_exp = np.asarray(true_Vs_exp)
    preds_exp = np.asarray(preds_exp)
    true_vals_exp = np.asarray(true_vals_exp)

    # Save results if required
    if savepath is not None:

        # Make the results folder if it does not exist
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        with open(os.path.join(savepath, 'fitted-tau.pkl'), 'wb') as file:
            pickle.dump(fitted_tau_exp, file)

        with open(os.path.join(savepath, 'fitted-Vs.pkl'), 'wb') as file:
            pickle.dump(fitted_Vs_exp, file)

        with open(os.path.join(savepath, 'true-tau.pkl'), 'wb') as file:
            pickle.dump(true_tau_exp, file)

        with open(os.path.join(savepath, 'true-Vs.pkl'), 'wb') as file:
            pickle.dump(true_Vs_exp, file)

        with open(os.path.join(savepath, 'preds.pkl'), 'wb') as file:
            pickle.dump(preds_exp, file)

        with open(os.path.join(savepath, 'true-vals.pkl'), 'wb') as file:
            pickle.dump(true_vals_exp, file)

        with open(os.path.join(savepath, 'n-steps.pkl'), 'wb') as file:
            pickle.dump(n_steps_list, file)

        with open(os.path.join(savepath, 'history.pkl'), 'wb') as file:
            pickle.dump(history_exp, file)

        with open(os.path.join(savepath, 'euler-method-params.json'), 'w') as file:
            json.dump(euler_method_params, file)

        with open(os.path.join(savepath, 'train-params.json'), 'w') as file:
            json.dump(train_params, file)

########################################################################################################################


def experiment_3(true_a_list: List[float],
                 true_b_list: List[float],
                 n_samples: List[int],
                 n_steps: int,
                 eoh_list: List[float],
                 savepath: str = None):
    """
    With this experiment, we show that the variability of the dataset strongly affects the estimated parameters of the
    linear regression model.
    :param true_a_list: list of float; the linear coefficients of the linear regression models.
    :param true_b_list: list of float; the bias of the linear regression models.
    :param n_samples: int; number of generated measurements for each ramp.
    :param n_steps: int: number of steps of the Euler method.
    :param eoh_list: float; the upper bound of the time interval of the dataset as a multiple of \tau.
    :param savepath: str; where the results are saved to.
    :return:
    """

    # Warn the user if nothing will be saved
    if savepath is None:
        warnings.warn("You did not specify a savepath; results will not be saved.")

    # Dictionary with the parameters for the Euler method object.
    euler_method_params = dict()
    euler_method_params['kernel_init_val'] = 1
    euler_method_params['bias_init_val'] = 1
    euler_method_params['Vs_init_val'] = 1
    euler_method_params['tau_init_val'] = None
    euler_method_params['n_steps'] = n_steps

    # Dictionary with the parameters for the training routine
    train_params = dict()
    train_params['epochs'] = 100000
    train_params['loss_fn'] = 'MeanSquaredError'
    train_params['optimizer'] = 'Adam'
    train_params['learning_rate'] = 0.1
    train_params['train_metric'] = 'Mean'
    train_params['patience'] = 50
    train_params['batch_size'] = 8
    train_params['delta_loss'] = 0.0001

    # Keep track of results
    fitted_a_exp = list()
    true_a_exp = list()
    fitted_b_exp = list()
    true_b_exp = list()
    history_exp = list()
    preds_exp = list()
    true_vals_exp = list()
    true_tau_exp = list()
    pred_tau_exp = list()

    for true_a, true_b in tqdm(zip(true_a_list, true_b_list), desc='Experiment runs', total=len(true_a_list)):
        # FIXME: set a configurable bias
        true_b = 0

        # Keep track of the true and fitted \tau for each number of steps value of the Euler method
        fitted_a_eoh = list()
        true_a_eoh = list()
        true_b_eoh = list()
        fitted_b_eoh = list()
        history_eoh = list()
        preds_eoh = list()
        true_vals_eoh = list()
        preds_tau_eoh = list()
        true_tau_eoh = list()

        # Generated \tau accordingly to a linear relationship \tau = ax and then v_c(t)
        test_data, test_h_f = \
            linear_rel_synthetic_data(linear_coeff=true_a,
                                      offset=true_b,
                                      n_samples=n_samples,
                                      Vs=euler_method_params['Vs_init_val'],
                                      init_v=0,
                                      n_steps=10000,
                                      eoh=5)

        # Create the training set
        test_set = \
            create_training_set_with_timesteps(loadpath=test_data,
                                               target_column='Voltage',
                                               input_columns=['x'],
                                               param_column='Tau')

        # Add a fake feature dimension to the initial, input and target values
        init_val_test_set = np.expand_dims(test_set['Voltage'].values, axis=1)
        x_test_set = np.expand_dims(test_set['x'].values, axis=1)

        # Iterate for each value of EOH
        for eoh in eoh_list:

            # Generated \tau accordingly to a linear relationship \tau = ax and then v_c(t)
            train_data, h_f = \
                linear_rel_synthetic_data(linear_coeff=true_a,
                                          offset=true_b,
                                          n_samples=n_samples,
                                          Vs=euler_method_params['Vs_init_val'],
                                          init_v=0,
                                          n_steps=10000,
                                          eoh=eoh)
            # Create the training set
            tr_set = \
                create_training_set_with_timesteps(loadpath=train_data,
                                                   target_column='Voltage',
                                                   input_columns=['x'],
                                                   param_column='Tau')

            # Add a fake feature dimension to the initial, input and target values
            init_val = np.expand_dims(tr_set['Voltage'].values, axis=1)
            x = np.expand_dims(tr_set['x'].values, axis=1)
            target_vals = np.expand_dims(tr_set['Next_Voltage'].values, axis=1)

            # Create the graph
            graph = EulerMethod(full_unrolling=False,
                                h_f=h_f,
                                Vs_input_shape=None,
                                tau_input_shape=(1,),
                                trainable_Vs=False,
                                trainable_tau=True,
                                **euler_method_params)

            # Train the model
            history = train(model=graph,
                            init_vals=init_val,
                            target_vals=target_vals,
                            inputs=x,
                            verbose=10,
                            **train_params)

            graph.h_f = test_h_f

            # Get the kernel of the linear model
            fitted_kernel = graph.tau.trainable_variables[0].numpy().item()
            # FIXME: set a configurable bias
            # fitted_bias = graph.tau.trainable_variables[1].numpy().item()
            fitted_bias = None

            # Get the predicted \tau
            preds_tau = graph.get_tau(inputs=x_test_set)
            preds_tau = np.squeeze(preds_tau)
            preds_tau_eoh.append(preds_tau)

            # Get the true \tau
            true_tau_eoh.append(test_data['Tau'].values[:-1])

            # Get predictions with the fitted parameters
            fitted_params_preds, _ = graph.forward(init_val_test_set, inputs=x_test_set)
            fitted_params_preds = np.squeeze(fitted_params_preds)

            # Compute the RMSE of tau
            rmse_tau = np.sqrt(np.mean(np.square(test_data['Tau'].values[:-1] - preds_tau), axis=0))

            # Sanity check
            true_values = test_data['Voltage'].values[1:]
            assert true_values.shape == fitted_params_preds.shape

            preds_eoh.append(fitted_params_preds)
            true_vals_eoh.append(true_values)

            # Compute RMSE
            rmse_preds = np.sqrt(np.mean(np.square(fitted_params_preds - true_values)))

            # Keep track of results for the current EOH
            fitted_a_eoh.append(fitted_kernel)
            fitted_b_eoh.append(fitted_bias)
            true_a_eoh.append(true_a)
            true_b_eoh.append(true_b)
            history_eoh.append(history)

            # Visualize the results
            print_str = f'EOH: {eoh} | Predictions RMSE: {rmse_preds} |'
            print_str += f' Fitted kernel: {fitted_kernel} | Fitted bias: {fitted_bias}'
            print_str += f' | True kernel: {true_a} | True bias: {true_b}'
            print(print_str)
            print(f'RMSE tau: {rmse_tau}\n')
            print('-' * len(print_str) + '\n')

        # Keep track of the results for each experiment
        fitted_a_exp.append(fitted_a_eoh)
        true_a_exp.append(true_a_eoh)
        true_b_exp.append(true_b_eoh)
        fitted_b_exp.append(fitted_b_eoh)
        preds_exp.append(preds_eoh)
        true_vals_exp.append(true_vals_eoh)
        true_tau_exp.append(true_tau_eoh)
        pred_tau_exp.append(preds_tau_eoh)
        history_exp.append(history_eoh)

    # RMSE of kernel, bias and the predictions over the different experiments
    true_a_exp = np.asarray(true_a_exp)
    fitted_a_exp = np.asarray(fitted_a_exp)
    true_b_exp = np.asarray(true_b_exp)
    fitted_b_exp = np.asarray(fitted_b_exp)
    preds_exp = np.asarray(preds_exp)
    true_vals_exp = np.asarray(true_vals_exp)
    true_tau_exp = np.asarray(true_tau_exp)
    pred_tau_exp = np.asarray(pred_tau_exp)

    # Save results if required
    if savepath is not None:

        # Make the results folder if it does not exist
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        save_to_pickle(savepath=os.path.join(savepath, 'fitted-a.pkl'), res=fitted_a_exp)
        save_to_pickle(savepath=os.path.join(savepath, 'true-a.pkl'), res=true_a_exp)
        save_to_pickle(savepath=os.path.join(savepath, 'fitted-b.pkl'), res=fitted_b_exp)
        save_to_pickle(savepath=os.path.join(savepath, 'true-b.pkl'), res=true_b_exp)
        save_to_pickle(savepath=os.path.join(savepath, 'eoh.pkl'), res=eoh_list)
        save_to_pickle(savepath=os.path.join(savepath, 'fitted-params-preds.pkl'), res=preds_exp)
        save_to_pickle(savepath=os.path.join(savepath, 'true-values.pkl'), res=true_vals_exp)
        save_to_pickle(savepath=os.path.join(savepath, 'history.pkl'), res=history_exp)
        save_to_pickle(savepath=os.path.join(savepath, 'true-tau.pkl'), res=true_tau_exp)
        save_to_pickle(savepath=os.path.join(savepath, 'pred-tau.pkl'), res=pred_tau_exp)
        save_to_json(savepath=os.path.join(savepath, 'euler-method-params.json'), config=euler_method_params)
        save_to_json(savepath=os.path.join(savepath, 'train-params.json'), config=train_params)

########################################################################################################################


def main():
    parser = argparse.ArgumentParser()
    experiment_id_help_str = "The identifier of the experiment to run.\n"
    experiment_id_help_str += "0: test whether training the UODE by decomposing a ramp in sequence of consecutive " \
                              "measurements affects the performance.\n"
    experiment_id_help_str += "1: show how the predictions of the UODE with the fitted parameters are more " \
                              "accurate than the predictions we could achieve with the true parameters."
    experiment_id_help_str += "2: show how the estimation of the tau parameters benefits from an increasing in the " \
                              "computational complexity of the Euler method; the Vs estimation is still accurate " \
                              "even when the computational complexity of the Euler method is low."
    experiment_id_help_str += "3: show that the variability of the dataset strongly affects the estimated parameters " \
                              "of the linear regression model."

    parser.add_argument("--savepath", type=str, required=False, help="Where results are saved to", default=None)

    parser.add_argument("--exp-id",
                        type=int,
                        required=True,
                        help=experiment_id_help_str,
                        choices=[0, 1, 2, 3])

    parser.add_argument("--Vs-lower", type=int, help="Lower bound for the generated Vs", default=5)
    parser.add_argument("--Vs-upper", type=int, help="Upper bound for the generated Vs", default=10)
    parser.add_argument("--tau-lower", type=int, help="Lower bound for the generated Vs", default=2)
    parser.add_argument("--tau-upper", type=int, help="Upper bound for the generated Vs", default=6)
    parser.add_argument("--num-runs", type=int, help="Number of Vs-tau generated pairs", default=5)
    parser.add_argument("--num-samples", type=int, help="Number of samples for each ramp", default=10)
    parser.add_argument('--num-steps-list',
                        nargs='+',
                        type=int,
                        help="List of number of steps for the Euler method",
                        default=[1, 3, 5, 10, 20, 50])
    parser.add_argument('--num-steps', type=int, help="Single value of number of steps for the Euler method",
                        default=10)
    parser.add_argument("--kernel-upper-bound",
                        type=int,
                        help="Upper bound of the coefficient of the linear regression model",
                        default=6)
    parser.add_argument("--kernel-lower-bound",
                        type=int,
                        help="Lower bound coefficient of the linear regression model",
                        default=2)
    parser.add_argument("--bias-upper-bound",
                        type=int,
                        help="Upper bound of the offset of the linear regression model",
                        default=2)
    parser.add_argument("--bias-lower-bound",
                        type=int,
                        help="Lower bound of the offset of the linear regression model",
                        default=1)
    parser.add_argument('--eoh-list',
                        nargs='+',
                        type=float,
                        help="List of upper bounds of the dataset as a multiple of tau coefficient",
                        default=[0.5, 1.0, 1.5, 3.0, 5.0])
    parser.add_argument("--gpu",
                        action="store_true",
                        default=False,
                        help="Set this flag if you want to use the GPU; only the first GPU will be used")

    args = parser.parse_args()
    savepath = args.savepath
    exp_id = args.exp_id
    Vs_lower = args.Vs_lower
    Vs_upper = args.Vs_upper
    tau_lower = args.tau_lower
    tau_upper = args.tau_upper
    num_runs = args.num_runs
    n_samples = args.num_samples
    n_steps_list = args.num_steps_list
    n_steps = args.num_steps
    kernel_lower = args.kernel_lower_bound
    kernel_upper = args.kernel_upper_bound
    bias_lower = args.bias_lower_bound
    bias_upper = args.bias_upper_bound
    eoh_list = args.eoh_list
    no_gpu = not args.gpu

    config_gpu(no_gpu=no_gpu)

    true_a_list = np.random.uniform(low=kernel_lower, high=kernel_upper, size=num_runs)
    true_b_list = np.random.uniform(low=bias_lower, high=bias_upper, size=num_runs)

    if exp_id == 0:
        experiment_0(Vs_bounds=(Vs_lower, Vs_upper),
                     tau_bounds=(tau_lower, tau_upper),
                     num_runs=num_runs,
                     n_samples=n_samples,
                     savepath=savepath)
    elif exp_id == 1:
        experiment_1(Vs_bounds=(Vs_lower, Vs_upper),
                     tau_bounds=(tau_lower, tau_upper),
                     num_runs=num_runs,
                     n_samples=n_samples,
                     savepath=savepath)
    elif exp_id == 2:
        experiment_2(Vs_bounds=(Vs_lower, Vs_upper),
                     tau_bounds=(tau_lower, tau_upper),
                     num_runs=num_runs,
                     n_samples=n_samples,
                     n_steps_list=n_steps_list,
                     savepath=savepath)
    elif exp_id == 3:
        experiment_3(true_a_list=true_a_list,
                     true_b_list=true_b_list,
                     n_samples=n_samples,
                     n_steps=n_steps,
                     eoh_list=eoh_list,
                     savepath=savepath)
    else:
        raise Exception(f"Experiment {exp_id} does not exist")

