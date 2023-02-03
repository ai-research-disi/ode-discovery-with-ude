"""
    Sanity check for the data generation process.
"""

import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers.utility import rc_circuit, linear_rel_synthetic_data
from helpers.utility import create_training_set_with_timesteps, euler_method

########################################################################################################################


class MyTestCase(unittest.TestCase):

    def test_01_rc_circuit(self):
        Vs = 10
        tau_val = 2
        init_v = 0
        timesteps = np.arange(0, 5*tau_val)
        tau = [tau_val] * len(timesteps)

        values = rc_circuit(timesteps=timesteps[1:],
                            tau=tau,
                            init_v=init_v,
                            Vs=Vs)

        self.assertTrue(isinstance(values, list))

        dataframe = pd.DataFrame(columns=['Timestep', 'Value'])
        dataframe['Timestep'] = timesteps
        dataframe['Value'] = values
        tr_set = create_training_set_with_timesteps(loadpath=dataframe,
                                                    target_column='Value',
                                                    input_columns=None,
                                                    param_column=None)

        self.assertTrue(isinstance(tr_set, pd.DataFrame))

    def test_04_linear_rel_synthetic_data(self):
        true_a = 2
        true_b = 1
        n_samples = 50
        Vs = 70
        init_v = 0
        eoh = 2

        train_data, h_f = \
            linear_rel_synthetic_data(linear_coeff=true_a,
                                      offset=true_b,
                                      n_samples=n_samples,
                                      Vs=Vs,
                                      init_v=init_v,
                                      n_steps=10000,
                                      eoh=eoh)

        self.assertTrue(isinstance(train_data, pd.DataFrame))
        self.assertTrue((train_data.columns == ['x', 'Tau', 'Voltage', 'Time']).all())

        # Create the training set
        tr_set = create_training_set_with_timesteps(loadpath=train_data,
                                                    target_column='Voltage',
                                                    input_columns=['x'],
                                                    param_column='Tau')
        self.assertTrue(isinstance(tr_set, pd.DataFrame))

########################################################################################################################


if __name__ == '__main__':
    unittest.main()
