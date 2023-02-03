import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow
import numpy as np
import time
import os

from models.sir.custom_models import UnrolledSIR, PairedSIR, EndtoEndIntervetionMap, SIR, build_model
import helpers.sir.data_generator as generator

########################################################################################################################


def test_unrolled(y0, n_days, beta_range, gamma, steps, epochs, callbacks=[], noise=None, n_iter=100): 
    
    results = {
        'beta_ground': [],
        'beta_approx':[],
        'std_ground':[],
        'std_approx':[],
        'SIR_ground':[],
        'SIR_approx':[],
        'steps':[],
        'epochs':[],
        'time':[],
        }
    
    stochastic = bool(noise)
    n = int(n_days/2)

    for i in range(n_iter):
        beta = np.random.uniform(beta_range[0], beta_range[1], size=(1,))
        betas = np.repeat(beta, n_days)
        gen = generator.CompartmentGenerator(y0, betas, generator.SIR, 10000, 1, {'gamma': gamma}, noise=noise)
        y = gen.sample()[:n]
        y = y[:n]
        print(len(y))
        model = UnrolledSIR(len(y), steps, gamma=gamma, stochastic=stochastic)
        model.compile(optimizer=Adam(1e-2), loss='mse')
        print('Executing Run n. {}'.format(i))
        t0 = time.time()
        history = model.fit(y, epochs=epochs, batch_size=len(y), shuffle=False, verbose=0, callbacks=callbacks)
        tf = time.time()
        t = tf - t0
        print('Duration: {}'.format(t))
        print('Epochs: {}'.format(callbacks[0].stopped_epoch))

        results['beta_ground'].append(beta[0])
        print(beta)
        results['beta_approx'].append(model.beta.numpy()[0])
        std = noise if stochastic else 0
        std_pred = model.std.numpy()[0] if stochastic else 0
        results['std_ground'].append(std)
        results['std_approx'].append(std_pred)
        results['SIR_ground'].append(y[1:])
        results['SIR_approx'].append(model(y0).numpy())
        results['steps'].append(steps)
        results['epochs'].append(callbacks[0].stopped_epoch)
        results['time'].append(t)
        
    df = pd.DataFrame.from_dict(results)
    # df.to_csv(filename)
    df.to_csv('results/unrolled_10000.csv', mode='a', header=False)

########################################################################################################################

    
def test_supervised(y0_base,
                    n_days,
                    beta_range,
                    gamma,
                    steps,
                    epochs,
                    batch_size,
                    callbacks=[],
                    noise=None,
                    n_iter=100):
    
    results = {
        'beta_ground': [],
        'beta_approx':[],
        'std_ground':[],
        'std_approx':[],
        'SIR_ground':[],
        'SIR_approx':[],
        'SIR_ground_run':[],
        'steps':[],
        'epochs':[],
        'time':[],
        }
    
    stochastic = bool(noise)
    n = int(n_days/2)

    for i in range(n_iter):
        beta = np.random.uniform(beta_range[0], beta_range[1], size=(1,))
        betas = np.repeat(beta, n_days)
        gen = generator.CompartmentGenerator(y0_base, betas, generator.SIR, 10000, 1, {'gamma': gamma}, noise=noise)
        y0, yt = gen.sample(training=True)
        y0 = y0[:n]
        yt = yt[:n]
        # y = gen.sample()
        # y = y[:n]
        # idx = np.arange(0, len(y[:,0]), 1)
        # idx = idx.reshape(-1, 2)
        # y0 = y[idx[:,0]]
        # yt = y[idx[:,1]]
        # print(len(yt))
        model = PairedSIR(steps, gamma=gamma, stochastic=stochastic)
        model.compile(optimizer=Adam(1e-2), loss='mse')
        print('Executing Run n. {}'.format(i))
        t0 = time.time()
        history = model.fit(y0, yt, epochs=epochs, batch_size=16, verbose=0, callbacks=callbacks)
        tf = time.time()
        t = tf - t0
        y = np.array([y0[0]])
        pred = []
        for i in range(n):
            y = model(y).numpy()
            pred.append(y)
        
        old_beta = model.beta
        model.beta = tensorflow.Variable(beta, dtype=tensorflow.float32)
        if stochastic:
            old_std = model.std
            model.std = tensorflow.Variable(noise, dtype=tensorflow.float32)

        y = np.array([y0[0]])
        pred_ground = []
        for i in range(n):
            y = model(y).numpy()
            pred_ground.append(y)
            
        model.beta = old_beta
        if stochastic:
            model.std = old_std
        
        print('Duration: {}'.format(t))
        print('Epochs: {}'.format(callbacks[0].stopped_epoch))
        results['beta_ground'].append(beta[0])
        results['beta_approx'].append(model.beta.numpy()[0])
        std = noise if stochastic else 0
        std_pred = model.std.numpy()[0] if stochastic else 0
        results['std_ground'].append(std)
        results['std_approx'].append(std_pred)
        results['SIR_ground'].append(yt)
        results['SIR_approx'].append(np.array(pred))
        results['SIR_ground_run'].append(np.array(pred_ground))
        results['steps'].append(steps)
        results['epochs'].append(callbacks[0].stopped_epoch)
        results['time'].append(t)
        
    df = pd.DataFrame.from_dict(results)
    # df.to_csv(filename)
    df.to_csv('results/supervised_10000.csv', mode='a', header=False)

########################################################################################################################
    

def test_observable(y0_base,
                    n_days,
                    beta_range,
                    gamma,
                    steps,
                    epochs,
                    batch_size,
                    callbacks=[],
                    npis=None,
                    noise=None,
                    n_iter=100):
    
    results = {
        'beta_ground': [],
        'beta_approx':[],
        'std_ground':[],
        'std_approx':[],
        'SIR_ground':[],
        'SIR_approx':[],
        'steps':[],
        'epochs':[],
        'time':[],
        }
    
    stochastic = bool(noise)

    for i in range(n_iter):
        beta = np.random.uniform(beta_range[0], beta_range[1], size=(1,))
        yt = []
        betas = np.repeat(beta, n_days)
        gen = generator.CompartmentGenerator(y0_base,
                                             betas,
                                             generator.SIR,
                                             10000,
                                             7,
                                             {'gamma': gamma},
                                             noise=noise,
                                             npis=npis)

        while len(yt) < 5:
            y0, yt = gen.sample(training=True)  
            print(len(yt))

        y0, yt = y0[:20], yt[:20]
        output_size = 2 if stochastic else 1
        beta_pred = build_model(input_size=len(npis), output_size=output_size, hidden=[16, 16])
        model = EndtoEndIntervetionMap(beta_pred, SIR, steps, 1, params, stochastic=stochastic)
        model.compile(optimizer=Adam(1e-2), loss='mse')
        
        print('Executing Run n. {}'.format(i))
        t0 = time.time()
        history = model.fit((y0, gen.npis_schedule[:len(y0)]), yt, batch_size=8, epochs=500, verbose=0, callbacks=callbacks)   
        tf = time.time()
        t = tf - t0
        beta_pred = []
        y_pred = []
        for i in range(len(y0)):
            y_t = np.array([y0[i]])
            npis_t = np.array([gen.npis_schedule[:len(y0)][i]])
            y_t = model((y_t, npis_t)).numpy()
            beta_pred.append(model.betas.numpy()[0, 0] if not stochastic else [model.x.numpy()[0, 0], model.x.numpy()[0, 1]])
            y_pred.append(list(y_t[0]))
        beta_pred = np.array(beta_pred)
        print('Duration: {}'.format(t))
        print('Epochs: {}'.format(callbacks[0].stopped_epoch))
        results['beta_ground'].append(gen.betas[:20])
        betas = beta_pred[:,0] if stochastic else beta_pred
        results['beta_approx'].append(beta_pred)
        std = noise if stochastic else 0
        std_pred = beta_pred[:,1] if stochastic else np.zeros(len(betas))
        results['std_ground'].append(std)
        results['std_approx'].append(np.mean(std_pred))
        results['SIR_ground'].append(yt)
        results['SIR_approx'].append(y_pred)
        results['steps'].append(steps)
        results['epochs'].append(callbacks[0].stopped_epoch)
        results['time'].append(t)

        
    df = pd.DataFrame.from_dict(results)
    # df.to_csv(filename)

    if not os.path.exists('results'):
        os.makedirs('results')

    df.to_csv('results/observable_2.csv', mode='a', header=False)

########################################################################################################################
    
    
def test_sampling(y0_base,
                  n_samples,
                  n_weeks,
                  beta_range,
                  gamma,
                  steps,
                  epochs,
                  batch_size,
                  callbacks=[],
                  npis=None,
                  noise=None,
                  n_iter=100):
    
    results = {
        'beta_ground': [],
        'beta_approx':[],
        'std_ground':[],
        'std_approx':[],
        'SIR_ground':[],
        'SIR_approx':[],
        'weeks': [],
        'steps':[],
        'epochs':[],
        'time':[],
        }
    
    stochastic = bool(noise)
    
    def generate_sample_weekly(gen):
        y0, x, yt = np.array([]), np.array([]), np.array([])
        while len(y0) < n_samples:
            # print(y0, yt)
            y0_tmp,  yt_tmp = gen.sample(training=True) 
            x_tmp = gen.npis_schedule[:len(y0_tmp)]
            y0 = np.concatenate([y0, y0_tmp], axis=0) if y0.size > 0 else y0_tmp
            yt = np.concatenate([yt, yt_tmp], axis=0) if yt.size > 0 else yt_tmp
            x = np.concatenate([x, x_tmp], axis=0) if x.size > 0 else x_tmp

        y0 = y0[:n_samples]
        yt = yt[:n_samples]
        x = x[:n_samples]
        return y0, x, yt
    
    for i in range(n_iter):
        beta = np.random.uniform(beta_range[0], beta_range[1], size=(1,))
        betas = np.repeat(beta, n_weeks * 7)
        gen = generator.CompartmentGenerator(y0_base, betas, generator.SIR, 10000, 7, {'gamma': gamma}, noise=noise, npis=npis)
        y0, x, yt = generate_sample_weekly(gen)
    
        output_size = 2 if stochastic else 1
        beta_pred = build_model(input_size=len(npis), output_size=output_size, hidden=[16, 16])
        model = EndtoEndIntervetionMap(beta_pred, SIR, steps, 1, params, stochastic=stochastic)
        model.compile(optimizer=Adam(1e-2), loss='mse')
        
        print('Executing Run n. {}'.format(i))
        t0 = time.time()
        history = model.fit((y0, x), yt, batch_size=8, epochs=500, verbose=0, callbacks=callbacks)   
        tf = time.time()
        t = tf - t0
        
        betas = np.repeat(beta, n_samples * 7)
        gen = generator.CompartmentGenerator(y0_base, betas, generator.SIR, 10000, 7, {'gamma': gamma}, noise=noise, npis=npis)
        y0, x, yt = generate_sample_weekly(gen)
        
        beta_pred = []
        y_pred = []
        for i in range(len(y0)):
            y_t = np.array([y0[i]])
            npis_t = np.array([x[i]])
            y_t = model((y_t, npis_t)).numpy()
            beta_pred.append(model.betas.numpy()[0, 0] if not stochastic else [model.x.numpy()[0, 0], model.x.numpy()[0, 1]])
            y_pred.append(list(y_t[0]))
            
        plt.plot(yt[:,1], label='true')
        plt.plot(y0[:,1], label='true0')
        plt.plot(np.array(y_pred)[:,1], label='predicted')
        plt.legend()
        plt.show()
        beta_pred = np.array(beta_pred)
        print('Duration: {}'.format(t))
        print('Epochs: {}'.format(callbacks[0].stopped_epoch))
        results['beta_ground'].append(gen.betas[:20])
        betas = beta_pred[:,0] if stochastic else beta_pred
        results['beta_approx'].append(beta_pred)
        std = noise if stochastic else 0
        std_pred = beta_pred[:,1] if stochastic else np.zeros(len(betas))
        results['std_ground'].append(std)
        results['std_approx'].append(np.mean(std_pred))
        results['SIR_ground'].append(yt)
        results['SIR_approx'].append(y_pred)
        results['weeks'].append(n_weeks)
        results['steps'].append(steps)
        results['epochs'].append(callbacks[0].stopped_epoch)
        results['time'].append(t)

    df = pd.DataFrame.from_dict(results)
    # df.to_csv(filename)
    df.to_csv('results/sampling_2.csv', mode='a', header=False)

########################################################################################################################


if __name__ == '__main__':
    
    y0 = np.array([.99, 0.01, .00])
    n_days = 200
    params = {'gamma': .1}
    window_size = 1
    steps = 1
    beta_range = [0.2, 0.4]
    gamma = 0.1
    noise = 0.01
    n_iter = 2
    batch_size = 16
    epochs = 1000
            
    npis = [
            generator.NPI(name='npi_1', effect=0.5),
            generator.NPI(name='npi_2', effect=0.8),
           ]
    
    callbacks = [EarlyStopping(monitor='loss',
                               min_delta=10e-8,
                               restore_best_weights=True,
                               patience=10)]
    test_observable(y0, n_days, beta_range, gamma, 10, epochs, 2, callbacks=callbacks, npis=npis, n_iter=n_iter)
    
    for weeks in [5, 10, 15, 20]:
            callbacks = [EarlyStopping(monitor='loss',
                                       min_delta=10e-8,
                                       restore_best_weights=True,
                                       patience=10)]
            test_sampling(y0, 20, weeks, beta_range, gamma, 10, epochs, 2, callbacks=callbacks, npis=npis, n_iter=n_iter)
    
    # deterministic
    callbacks = [EarlyStopping(monitor='loss',
                               min_delta=10e-4,
                               restore_best_weights=True,
                               patience=10)]
    test_unrolled(y0, n_days, beta_range, gamma, 1, epochs, callbacks=callbacks, n_iter=n_iter)
        
    steps = [1, 2, 5, 10, 20, 50]
    
    for step in steps:
        callbacks = [EarlyStopping(monitor='loss',
                                   min_delta=10e-9,
                                   restore_best_weights=True,
                                   patience=10)]
        test_supervised(y0, n_days, beta_range, gamma, step, epochs, 8, callbacks=callbacks, n_iter=n_iter)
        
        callbacks = [EarlyStopping(monitor='loss',
                                   min_delta=10e-5,
                                   restore_best_weights=True,
                                   patience=40)]
        test_supervised(y0, n_days, beta_range, gamma, step, epochs, 8, callbacks=callbacks, n_iter=n_iter, noise=noise)
        
    steps = [1, 2, 5, 10]
    
    for step in steps:
        callbacks = [EarlyStopping(monitor='loss',
                                   min_delta=10e-8,
                                   restore_best_weights=True,
                                   patience=40)]
        test_observable(y0, n_days, beta_range, gamma, step, epochs, 8, callbacks=callbacks, npis=npis, n_iter=n_iter)
        
        callbacks=[EarlyStopping(monitor='loss',
                                 min_delta=10e-8,
                                 restore_best_weights=True,
                                 patience=40)]
        test_observable(y0,
                        n_days,
                        beta_range,
                        gamma,
                        step,
                        epochs,
                        8,
                        callbacks=callbacks,
                        npis=npis,
                        noise=noise,
                        n_iter=n_iter)
