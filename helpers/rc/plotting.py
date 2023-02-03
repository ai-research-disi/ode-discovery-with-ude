"""
    Plotting functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

from helpers.rc.utility import load_from_pickle, load_from_json, min_max_scaler

########################################################################################################################

sns.set_style('darkgrid')

########################################################################################################################


def show_res_exp_0(full_unroll_fitted_Vs_loadpath: str,
                   decomp_seq_fitted_Vs_loadpath: str,
                   true_Vs_loadpath: str,
                   full_unroll_fitted_tau_loadpath: str,
                   decomp_seq_fitted_tau_loadpath: str,
                   true_tau_loadpath: str,
                   full_unroll_preds_loadpath: str,
                   decomp_seq_preds_loadpath: str,
                   true_values_loadpath: str,
                   full_unroll_history_loadpath: str,
                   decom_seq_history_loadpath: str):
    """
    Show the results of the comparison between full-batch and mini-batch approaches.
    :param full_unroll_fitted_Vs_loadpath: str; loadpath for the full-batch estimated V_s; a .pkl file is expected.
    :param decomp_seq_fitted_Vs_loadpath: str; loadpath for the mini-batch estimated V_s; a .pkl file is expected.
    :param true_Vs_loadpath: str; loadpath for the true values of V_s; a .pkl file is expected.
    :param full_unroll_fitted_tau_loadpath: str; loadpath for the full-batch estimated \tau; a .pkl file is expected.
    :param decomp_seq_fitted_tau_loadpath: str; loadpath for the mini-batch estimated \tau; a .pkl file is expected.
    :param true_tau_loadpath: str; loadpath for the true values of \tau; a .pkl file is expected.
    :param full_unroll_preds_loadpath: str; loadpath for the full-batch V_c(t) predictions; a .pkl file is expected.
    :param decomp_seq_preds_loadpath: str; loadpath for the mini-batch V_c(t) predictions; a .pkl file is expected.
    :param true_values_loadpath: str; loadpath for the true values of V_c(t) predictions; a .pkl file is expected.
    :param full_unroll_history_loadpath: str; loadpath for the training history of full-batch; a .pkl file is expected.
    :param decom_seq_history_loadpath: str; loadpath for the training history of mini-batch; a .pkl file is expected.
    :return:
    """

    full_unroll_fitted_Vs = load_from_pickle(full_unroll_fitted_Vs_loadpath)
    full_unroll_fitted_Vs = np.asarray(full_unroll_fitted_Vs)

    decomp_seq_fitted_Vs = load_from_pickle(decomp_seq_fitted_Vs_loadpath)
    decomp_seq_fitted_Vs = np.asarray(decomp_seq_fitted_Vs)

    true_Vs = load_from_pickle(true_Vs_loadpath)
    true_Vs = np.asarray(true_Vs)

    full_unroll_fitted_tau = load_from_pickle(full_unroll_fitted_tau_loadpath)
    full_unroll_fitted_tau = np.asarray(full_unroll_fitted_tau)

    decomp_seq_fitted_tau = load_from_pickle(decomp_seq_fitted_tau_loadpath)
    decomp_seq_fitted_tau = np.asarray(decomp_seq_fitted_tau)

    true_tau = load_from_pickle(true_tau_loadpath)
    true_tau = np.asarray(true_tau)

    full_unroll_preds = load_from_pickle(full_unroll_preds_loadpath)
    full_unroll_preds = np.asarray(full_unroll_preds)

    decomp_seq_preds = load_from_pickle(decomp_seq_preds_loadpath)
    decomp_seq_preds = np.asarray(decomp_seq_preds)

    true_vals = load_from_pickle(true_values_loadpath)
    true_vals = np.asarray(true_vals)

    full_unroll_history = load_from_pickle(full_unroll_history_loadpath)
    full_unroll_end_epoch = [h['Epoch'][-1] for h in full_unroll_history]
    full_unroll_train_end = [h['Training duration'] for h in full_unroll_history]

    decomp_seq_history = load_from_pickle(decom_seq_history_loadpath)
    decomp_seq_end_epoch = [h['Epoch'][-1] for h in decomp_seq_history]
    decomp_seq_train_end = [h['Training duration'] for h in decomp_seq_history]

    full_unroll_fitted_Vs = min_max_scaler(full_unroll_fitted_Vs,
                                            lower=0,
                                            upper=true_Vs)
    decomp_seq_fitted_Vs = min_max_scaler(decomp_seq_fitted_Vs,
                                           lower=0,
                                           upper=true_Vs)

    full_unroll_fitted_tau = min_max_scaler(full_unroll_fitted_tau,
                                             lower=0,
                                             upper=true_tau)
    decomp_seq_fitted_tau = min_max_scaler(decomp_seq_fitted_tau,
                                            lower=0,
                                            upper=true_tau)
    true_tau = min_max_scaler(true_tau,
                               lower=0,
                               upper=true_tau)

    true_vals = min_max_scaler(true_vals,
                                lower=0,
                                upper=np.expand_dims(true_Vs, axis=1))

    decomp_seq_preds = min_max_scaler(decomp_seq_preds,
                                       lower=0,
                                       upper=np.expand_dims(true_Vs, axis=1))

    full_unroll_preds = min_max_scaler(full_unroll_preds,
                                        lower=0,
                                        upper=np.expand_dims(true_Vs, axis=1))

    true_Vs = min_max_scaler(true_Vs,
                              lower=0,
                              upper=true_Vs)

    assert true_Vs.shape == full_unroll_fitted_Vs.shape and true_Vs.shape == decomp_seq_fitted_Vs.shape
    assert true_tau.shape == full_unroll_fitted_tau.shape and true_tau.shape == decomp_seq_fitted_tau.shape
    assert true_vals.shape == full_unroll_preds.shape and true_vals.shape == decomp_seq_preds.shape

    full_unroll_Vs_mean = np.mean(np.abs(true_Vs - full_unroll_fitted_Vs), axis=0)
    decomp_seq_Vs_mean = np.mean(np.abs(true_Vs - decomp_seq_fitted_Vs), axis=0)
    full_unroll_Vs_std = np.std(np.abs(true_Vs - full_unroll_fitted_Vs), axis=0)
    decomp_seq_Vs_std = np.std(np.abs(true_Vs - decomp_seq_fitted_Vs), axis=0)

    full_unroll_tau_mean = np.mean(np.abs(true_Vs - full_unroll_fitted_tau), axis=0)
    decomp_seq_tau_mean = np.mean(np.abs(true_Vs - decomp_seq_fitted_tau), axis=0)
    full_unroll_tau_std = np.std(np.abs(true_Vs - full_unroll_fitted_tau), axis=0)
    decomp_seq_tau_std = np.std(np.abs(true_Vs - decomp_seq_fitted_tau), axis=0)

    full_unroll_preds_rmse = np.sqrt(np.mean(np.square(true_vals - full_unroll_preds), axis=1))
    decomp_seq_preds_rmse = np.sqrt(np.mean(np.square(true_vals - decomp_seq_preds), axis=1))

    print('-'*50 + ' EXPERIMENT 0 ' + '-'*50 + '\n')

    max_str_len = 0

    print_str = f"Full unrolling Vs error: {full_unroll_Vs_mean} +- {full_unroll_Vs_std} "
    print_str += f"| Decomposed sequence Vs error: {decomp_seq_Vs_mean} +- {decomp_seq_Vs_std}"
    max_str_len = max(max_str_len, len(print_str))
    print(print_str)

    print_str = f"Full unrolling tau error: {full_unroll_tau_mean} +- {full_unroll_tau_std} "
    print_str += f"| Decomposed sequence tau error: {decomp_seq_tau_mean} +- {decomp_seq_tau_std}"
    max_str_len = max(max_str_len, len(print_str))
    print(print_str)

    print_str = f"Full unrolling preds. RMSE: {np.mean(full_unroll_preds_rmse)} +- {np.std(full_unroll_preds_rmse)} | "
    print_str += f"Decomposed sequence preds. RMSE: {np.mean(decomp_seq_preds_rmse)} +- {np.std(decomp_seq_preds_rmse)}"
    max_str_len = max(max_str_len, len(print_str))
    print(print_str)

    print_str = f"Full unrolling avg. number of epochs: {np.mean(full_unroll_end_epoch)} +- "
    print_str += f"{np.std(full_unroll_end_epoch)} | Decomposed sequence avg. number of epochs: "
    print_str += f"{np.mean(decomp_seq_end_epoch)} +- {np.std(decomp_seq_end_epoch)}"
    max_str_len = max(max_str_len, len(print_str))
    print(print_str)

    print_str = f"Full unrolling avg. training duration: {np.mean(full_unroll_train_end)} +- "
    print_str += f"{np.std(full_unroll_train_end)} | Decomposed sequence avg. training duration: "
    print_str += f"{np.mean(decomp_seq_train_end)} +- {np.std(decomp_seq_train_end)}"
    max_str_len = max(max_str_len, len(print_str))
    print(print_str + '\n')

    print('-' * max_str_len + '\n')

########################################################################################################################


def show_res_exp_1(fitted_params_preds_loadpath: str,
                   true_params_preds_loadpath: str,
                   true_values_loadpath: str,
                   true_Vs_loadpath: str):

    """
    This results are not reported in the paper; this method shows that the V_c(t) predictions with the true parameters
    are worse compared to the ones obtained with the estimated parameters.
    :param fitted_params_preds_loadpath: str; loadpath for the V_c(t) predictions obtained with the estimated p
                                              arameters; a .pkl file is expected.
    :param true_params_preds_loadpath: str; loadpath for the V_c(t) predictions obtained with the true parameters;
                                            a .pkl file is expected.
    :param true_values_loadpath: str; loadpath for the V_c(t) true values; a .pkl file is expected.
    :param true_Vs_loadpath: str; loadpath for the V_s true values; a .pkl file is expected.
    :return:
    """

    fitted_params_preds = load_from_pickle(fitted_params_preds_loadpath)
    true_params_preds = load_from_pickle(true_params_preds_loadpath)
    true_vals = load_from_pickle(true_values_loadpath)
    true_Vs = load_from_pickle(true_Vs_loadpath)

    fitted_params_preds = np.asarray(fitted_params_preds)
    true_params_preds = np.asarray(true_params_preds)
    true_vals = np.asarray(true_vals)
    true_Vs = np.asarray(true_Vs)

    fitted_params_preds = min_max_scaler(val=fitted_params_preds,
                                          lower=0,
                                          upper=np.expand_dims(true_Vs, axis=1))
    true_params_preds = min_max_scaler(val=true_params_preds,
                                        lower=0,
                                        upper=np.expand_dims(true_Vs, axis=1))
    true_vals = min_max_scaler(val=true_vals,
                                lower=0,
                                upper=np.expand_dims(true_Vs, axis=1))

    fitted_preds_rmse = np.sqrt(np.mean(np.square(fitted_params_preds - true_vals), axis=1))
    true_preds_rmse = np.sqrt(np.mean(np.square(true_params_preds - true_vals), axis=1))

    print('-' * 50 + ' EXPERIMENT 1 ' + '-' * 50)
    print_str = f"Fitted parameters RMSE: {np.mean(fitted_preds_rmse)} +- {np.std(fitted_preds_rmse)} \n"
    print_str += f"True parameters RMSE: {np.mean(true_preds_rmse)} +- {np.std(true_preds_rmse)}"
    print(print_str)
    print('-' * len(print_str) + '\n')

########################################################################################################################


def plot_exp_2(fitted_Vs_loadpath: str,
               true_Vs_loadpath: str,
               fitted_tau_loadpath: str,
               true_tau_loadpath: str,
               n_steps_loadpath: str,
               plot_Vs: bool = True):
    """
    Show results of the 'Solver accuracy' section of the paper.
    :param fitted_Vs_loadpath: str; loadpath for the fitted V_s with the different integration iterations values of the
                                    Euler method.
    :param true_Vs_loadpath: str; loadpath for the V_s true values.
    :param fitted_tau_loadpath: str; loadpath for the fitted \tau with the different integration iterations values of the
                                    Euler method.
    :param true_tau_loadpath: str; loadpath for the \tau true values.
    :param n_steps_loadpath: str; loadpath for the integration iterations values of the
                                  Euler method.
    :param plot_Vs: bool; True if you want to visualize the results for V_s, False otherwise.
    :return:
    """

    fitted_Vs = load_from_pickle(fitted_Vs_loadpath)
    true_Vs = load_from_pickle(true_Vs_loadpath)
    fitted_tau = load_from_pickle(fitted_tau_loadpath)
    true_tau = load_from_pickle(true_tau_loadpath)
    n_steps = load_from_pickle(n_steps_loadpath)

    fitted_Vs = min_max_scaler(fitted_Vs,
                                lower=0,
                                upper=true_Vs)
    true_Vs = min_max_scaler(true_Vs,
                              lower=0,
                              upper=true_Vs)
    fitted_tau = min_max_scaler(fitted_tau,
                                 lower=0,
                                 upper=true_tau)
    true_tau = min_max_scaler(true_tau,
                               lower=0,
                               upper=true_tau)

    assert true_Vs.shape == fitted_Vs.shape
    assert true_tau.shape == fitted_tau.shape
    Vs_avg = np.mean(np.abs(true_Vs - fitted_Vs), axis=0)
    tau_avg = np.mean(np.abs(true_tau - fitted_tau), axis=0)
    Vs_std = np.std(np.abs(true_Vs - fitted_Vs), axis=0)
    tau_std = np.std(np.abs(true_tau - fitted_tau), axis=0)

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    ax1.plot(n_steps, tau_avg, label=r'$\tau$')
    ax1.fill_between(n_steps, tau_avg - tau_std, tau_avg + tau_std, alpha=0.1)
    if plot_Vs:
        ax1.plot(n_steps, Vs_avg, label=r'$V_s$')
        ax1.fill_between(n_steps, Vs_avg - Vs_std, Vs_avg + Vs_std, alpha=0.1)
    ax1.set_xlabel('# of iterations', fontsize=18)
    ax1.set_ylabel("AE", rotation=90, fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.legend(fontsize=18)
    plt.savefig('exp-2-det.png', bbox_inches='tight', dpi=1000, format='png')

########################################################################################################################


def plot_exp_3(fitted_kernel_loadpath: str,
               true_kernel_loadpath: str,
               predictions_loadpath: str,
               true_values_loadpath: str,
               pred_tau_loadpath: str,
               true_tau_loadpath: str,
               eoh_loadpath: str,
               data_gen_loadpath: str):

    """
    Show the results for the data sampling and linear functional relation.
    :param fitted_kernel_loadpath: str; loadpath for the estimated linear coefficients of the linear relation; a .pkl
                                        file is expected.
    :param true_kernel_loadpath: str; loadpath for the true linear coefficients of the linear relation; a .pkl
                                      file is expected.
    :param predictions_loadpath: str; loadpath for the predicted V_c(t); a .pkl file is expected.
    :param true_values_loadpath: str; loadpath for the true V_c(t); a .pkl file is expected.
    :param pred_tau_loadpath: str; loadpath for the predicted \tau(t); a .pkl file is expected.
    :param true_tau_loadpath: str; loadpath for the true \tau(t); a .pkl file is expected.
    :param eoh_loadpath: str; loadpath for the EOH values; a .pkl file is expected.
    :param data_gen_loadpath: str; loadpath for the configuration file of the data generation process; a .json file is
                                   expected.
    :return:
    """

    fitted_kernel = load_from_pickle(fitted_kernel_loadpath)
    true_kernel = load_from_pickle(true_kernel_loadpath)
    preds = load_from_pickle(predictions_loadpath)
    true_vals = load_from_pickle(true_values_loadpath)
    pred_tau = load_from_pickle(pred_tau_loadpath)
    true_tau = load_from_pickle(true_tau_loadpath)
    eoh = load_from_pickle(eoh_loadpath)

    data_gen_params = load_from_json(data_gen_loadpath)

    fitted_kernel = \
        min_max_scaler(fitted_kernel,
                       lower=0,
                       upper=true_kernel)

    true_kernel = \
        min_max_scaler(true_kernel,
                       lower=0,
                       upper=true_kernel)

    preds = \
        min_max_scaler(preds,
                       lower=0,
                       upper=data_gen_params["Vs_init_val"])

    true_vals = \
        min_max_scaler(true_vals,
                       lower=0,
                       upper=data_gen_params["Vs_init_val"])

    min_tau = np.min(true_tau, axis=2, keepdims=True)
    max_tau = np.max(true_tau, axis=2, keepdims=True)
    pred_tau = min_max_scaler(pred_tau, lower=0, upper=max_tau)
    true_tau = min_max_scaler(true_tau, lower=0, upper=max_tau)

    assert pred_tau.shape == true_tau.shape
    tau_rmse = np.sqrt(np.mean(np.square(pred_tau - true_tau), axis=2))
    avg_tau_rmse = np.mean(tau_rmse, axis=0)
    std_tau_rmse = np.std(tau_rmse, axis=0)

    assert true_kernel.shape == fitted_kernel.shape
    kernel_avg_error = np.mean(np.abs(true_kernel - fitted_kernel), axis=0)
    kernel_std_error = np.std(np.abs(true_kernel - fitted_kernel), axis=0)

    assert preds.shape == true_vals.shape
    rmse_preds = np.sqrt(np.mean(np.square(preds - true_vals), axis=2))
    avg_rmse_preds = np.mean(rmse_preds, axis=0)
    std_rmse_preds = np.std(rmse_preds, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.plot(eoh, kernel_avg_error, label='$a$')
    ax1.fill_between(eoh,
                     kernel_avg_error - kernel_std_error,
                     kernel_avg_error + kernel_std_error,
                     alpha=0.1)
    xticks_labels = [str(i) + r'$\tau$' for i in eoh]
    ax1.set_xticklabels(xticks_labels, fontsize=18)
    ax1.set_xticks(eoh)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.legend(fontsize=18)
    ax1.set_xlabel('EOH', fontsize=18)
    ax1.set_ylabel('AE', rotation=90, fontsize=18)

    ax2.plot(eoh, avg_rmse_preds, label=r'$V_c(t)$')
    ax2.fill_between(eoh, avg_rmse_preds - std_rmse_preds, avg_rmse_preds + std_rmse_preds, alpha=0.1)
    xticks_labels = [str(i) + r'$\tau$' for i in eoh]
    ax2.set_xticklabels(xticks_labels, fontsize=18)
    ax2.set_xticks(eoh)
    ax2.set_xlabel('EOH', fontsize=18)
    ax2.set_ylabel('RMSE', rotation=90, fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.legend(fontsize=18)

    plt.savefig('exp-3-det.png', bbox_inches='tight', dpi=1000, format='png')

########################################################################################################################


if __name__ == '__main__':
    prefix = 'results/deterministic/rc/experiment-0/'
    full_unrolling_str = 'full-unrolling'
    decomposed_seq_str = 'decomposed-seq'

    show_res_exp_0(full_unroll_fitted_Vs_loadpath=os.path.join(prefix, f'{full_unrolling_str}-Vs.pkl'),
                   full_unroll_fitted_tau_loadpath=os.path.join(prefix, f'{full_unrolling_str}-tau.pkl'),
                   full_unroll_preds_loadpath=os.path.join(prefix, f'{full_unrolling_str}-preds.pkl'),
                   decomp_seq_fitted_Vs_loadpath=os.path.join(prefix, f'{decomposed_seq_str}-Vs.pkl'),
                   decomp_seq_fitted_tau_loadpath=os.path.join(prefix, f'{decomposed_seq_str}-tau.pkl'),
                   decomp_seq_preds_loadpath=os.path.join(prefix, f'{decomposed_seq_str}-preds.pkl'),
                   true_values_loadpath=os.path.join(prefix, 'true-values.pkl'),
                   true_tau_loadpath=os.path.join(prefix, 'true-tau.pkl'),
                   true_Vs_loadpath=os.path.join(prefix, 'true-Vs.pkl'),
                   full_unroll_history_loadpath=os.path.join(prefix, 'full-unrolling-history.pkl'),
                   decom_seq_history_loadpath=os.path.join(prefix, 'decomposed-seq-history.pkl'))

    prefix = 'results/deterministic/rc/experiment-2/'

    plot_exp_2(fitted_Vs_loadpath=os.path.join(prefix, 'fitted-Vs.pkl'),
               true_Vs_loadpath=os.path.join(prefix, 'true-Vs.pkl'),
               fitted_tau_loadpath=os.path.join(prefix, 'fitted-tau.pkl'),
               true_tau_loadpath=os.path.join(prefix, 'true-tau.pkl'),
               predictions_loadpath=None,
               true_values_loadpath=None,
               n_steps_loadpath=os.path.join(prefix, 'n-steps.pkl'),
               data_gen_loadpath=os.path.join(prefix, 'data-gen-params.json'),
               plot_Vs=True)

    prefix = 'results/deterministic/rc/experiment-3/'

    plot_exp_3(fitted_kernel_loadpath=os.path.join(prefix, 'fitted-a.pkl'),
               true_kernel_loadpath=os.path.join(prefix, 'true-a.pkl'),
               predictions_loadpath=os.path.join(prefix, 'fitted-params-preds.pkl'),
               true_values_loadpath=os.path.join(prefix, 'true-values.pkl'),
               pred_tau_loadpath=os.path.join(prefix, 'pred-tau.pkl'),
               true_tau_loadpath=os.path.join(prefix, 'true-tau.pkl'),
               eoh_loadpath=os.path.join(prefix, 'eoh.pkl'),
               data_gen_loadpath=os.path.join(prefix, 'data-gen-params.json'))


