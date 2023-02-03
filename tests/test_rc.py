import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

from helpers.utility import rc_derivative, euler_method, rc_circuit, create_training_set_with_timesteps
from helpers.utility import linear_rel_synthetic_data
from models import EulerMethod, train

########################################################################################################################


class MyTestCase(unittest.TestCase):

    @staticmethod
    def test_01_euler_method():

        n_steps = 10
        initial_value = 0
        tau_val_list = np.random.randint(low=1, high=6, size=5)
        Vs_list = np.random.randint(10, 100, size=5)

        for Vs, tau_val in zip(Vs_list, tau_val_list):
            t_n = 5 * tau_val
            tau = [tau_val] * n_steps

            # Data sampling frequency
            h_f = 5 * tau_val / n_steps

            # Generate one point measurement for each timestep
            timesteps = np.arange(0, 5 * tau_val + h_f, h_f)
            true_values = rc_circuit(timesteps=timesteps[1:],
                                     tau=tau,
                                     init_v=initial_value,
                                     Vs=Vs)

            f = rc_derivative(tau=tau_val, Vs=Vs)
            sol, intermediate_sols = euler_method(f=f, f0=initial_value, t_n=t_n, n_steps=n_steps)

            graph = EulerMethod(full_unrolling=True,
                                h_f=t_n,
                                n_steps=n_steps,
                                Vs_input_shape=None,
                                tau_input_shape=None,
                                trainable_Vs=False,
                                trainable_tau=False,
                                Vs_init_val=Vs,
                                tau_init_val=tau_val)

            _, predictions = graph.forward(init_v=np.ones(shape=(1, 1)) * initial_value,
                                           inputs=None)
            predictions = np.squeeze(predictions.numpy())
            # Check that the TF implementation is correct
            np.testing.assert_almost_equal(predictions, intermediate_sols, decimal=4)

            mape = mean_absolute_percentage_error(true_values, predictions)
            print(f'Vs: {Vs} | Tau: {tau_val} | MAPE: {mape}')

    @staticmethod
    def test_02_rc_circuit_fitting():
        Vs = 70
        tau_val = 2
        n_samples = 10

        tau = [tau_val] * n_samples
        h_f = 5 * tau_val / n_samples

        timesteps = np.arange(0, 5 * tau_val + h_f, h_f)
        true_values = rc_circuit(timesteps=timesteps[1:],
                                 tau=tau,
                                 init_v=0,
                                 Vs=Vs)

        graph = EulerMethod(full_unrolling=False,
                            h_f=h_f,
                            n_steps=1,
                            Vs_input_shape=None,
                            tau_input_shape=None,
                            trainable_Vs=False,
                            trainable_tau=True,
                            Vs_init_val=Vs,
                            tau_init_val=1.)

        dataframe = pd.DataFrame(columns=['Timestep', 'Value'])
        dataframe['Timestep'] = timesteps
        dataframe['Value'] = true_values
        tr_set = create_training_set_with_timesteps(loadpath=dataframe,
                                                    target_column='Value',
                                                    input_columns=None,
                                                    param_column=None)
        init_val = np.expand_dims(tr_set['Value'].values, axis=1)
        target_vals = np.expand_dims(tr_set['Next_Value'].values, axis=1)

        # Fit the model
        train(model=graph,
              init_vals=init_val,
              target_vals=target_vals,
              inputs=None,
              epochs=100000,
              loss_fn='MeanSquaredError',
              optimizer='Adam',
              learning_rate=0.1,
              train_metric='Mean',
              patience=100,
              delta_loss=0.001,
              verbose=100)

        fitted_tau = graph.get_tau(inputs=None)

        print_string = f'True tau: {tau_val} | Tau: {fitted_tau}'
        print(print_string)
        print('-' * len(print_string))

    @staticmethod
    def test_03_rc_circuit_ude():
        true_a = 2
        true_b = 1
        n_samples = 10
        Vs = 70
        init_v = 0
        eoh = 1.2
        n_steps = 20

        train_data, h_f = \
            linear_rel_synthetic_data(linear_coeff=true_a,
                                      offset=true_b,
                                      n_samples=n_samples,
                                      Vs=Vs,
                                      init_v=init_v,
                                      n_steps=10000,
                                      eoh=eoh)
        # Create the training set
        tr_set = create_training_set_with_timesteps(loadpath=train_data,
                                                    target_column='Voltage',
                                                    input_columns=['x'],
                                                    param_column='Tau')

        init_val = np.expand_dims(tr_set['Voltage'].values, axis=1)
        x = np.expand_dims(tr_set['x'].values, axis=1)
        target_vals = np.expand_dims(tr_set['Next_Voltage'].values, axis=1)

        graph = EulerMethod(full_unrolling=False,
                            h_f=h_f,
                            n_steps=n_steps,
                            Vs_input_shape=None,
                            tau_input_shape=(1,),
                            trainable_Vs=False,
                            trainable_tau=True,
                            Vs_init_val=Vs,
                            tau_init_val=None,
                            kernel_init_val=.1,
                            bias_init_val=.1)

        history = train(model=graph,
                        init_vals=init_val,
                        target_vals=target_vals,
                        inputs=x,
                        epochs=10000,
                        loss_fn='MeanSquaredError',
                        optimizer='Adam',
                        learning_rate=0.1,
                        train_metric='Mean',
                        batch_size=16,
                        patience=100,
                        delta_loss=0.0001,
                        verbose=10)

        fitted_kernel = graph.tau.trainable_variables[0].numpy().item()
        fitted_bias = graph.tau.trainable_variables[1].numpy().item()

        print_str = f'\nFitted kernel: {fitted_kernel} | Fitted bias: {fitted_bias} | True kernel: {true_a}'
        print_str += f' | True bias: {true_b}'
        print(print_str)
        print('-' * len(print_str))


if __name__ == '__main__':
    unittest.main()
