# An analysis of Universal Differential Equations for data-driven discovery of Ordinary Differential Equations

This repository contains the code to reproduce the experiments reported in the paper 
"An analysis of Universal Differential Equations for data-driven discovery of Ordinary Differential Equations" 
under review at ICCS-2023.

## RC circuit experiments

You can run the experiments by executing the module `main.rc`. 
With the `--exp-id` argument you can specify which experiment to run:

- `0`: compare `full-batch` and `mini-batch` training methods.
- `2`: asses how the accuracy of ODE parameters estimation increasing with increasing of integration iterations.
- `3`: see how the end-of-horizon of the dataset affects the estimation of the linear model that approximates the 
       relation between the observable and the \tau parameter.

Once you have the results, you can plot them using the `helpers.rc.plotting` script.

## SIR experiments

You can run the experiments by executing the script `main/sir/experiments.py`.
       


