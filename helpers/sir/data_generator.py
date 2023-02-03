#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def SIR(y0, betas, steps, window_size, gamma):
    
    N = sum(y0)
    S0, I0, R0 = y0
    S, I, R = [S0], [I0], [R0]
    h = 1 / steps
    for beta in betas:
        St, It, Rt = S[-1], I[-1], R[-1]
        for _ in range(window_size):
            for _ in range(steps):
                S_to_I = h * St * It * beta # if in scale 0, 1 no need for N
                I_to_R = h * gamma * It
                St -= S_to_I
                It += S_to_I - I_to_R
                Rt += I_to_R
        S.append(St)
        I.append(It)
        R.append(Rt)
            
        if It < 1e-4:
            break
    
    S, I, R = np.array(S).reshape(-1, 1), np.array(I).reshape(-1, 1), np.array(R).reshape(-1, 1)
    
    return np.concatenate([S, I, R], axis=1)    


class NPI:
    def __init__(self, name, effect):
        self.name = name
        self.effect = effect
        
        
class CompartmentGenerator:
    """ 
    Method to generate compartmental model curves
    """
    
    def __init__(self, y0, betas, f_ode, steps, window_size, params, npis=None, f_npis=None, noise=None):
        
        self.y0 = y0
        self.betas = betas
        self.betas_bk = betas
        self.f_ode = f_ode
        self.steps = steps
        self.window_size = window_size
        self.params = params 
        self.npis = npis
        self.f_npis = f_npis
        self.noise = noise
    
    def sample(self, training=False):
        
        self.betas = self.betas_bk
        # print(self.betas)
        if self.noise:
            noisy_betas = []
            for beta in self.betas:
                while True:
                    noisy_beta = np.random.normal(loc=beta, scale=self.noise) 
                    if noisy_beta > 0:
                        break
                noisy_betas.append(noisy_beta)
            self.betas = np.array(noisy_betas)

        if self.npis:
            n = int(len(self.betas) / self.window_size)
            self.betas = self.betas[:n]
            self.npis_schedule = np.random.choice([0, 1], size=(n, len(self.npis)))
            for i, npi in enumerate(self.npis):
                effect = np.where(self.npis_schedule[:, i] == 0, 1, npi.effect)
                self.betas *= effect
            
        y = self.f_ode(self.y0, self.betas, self.steps, self.window_size, **self.params)
        
        if training:
            idx = np.arange(0, len(y[:,0]), 1)
            y0 = y[idx[:-1]]
            yt = y[idx[1:]]
            return y0, yt
        
        # if indipendent_pair:
        #     idx = np.arange(0, len(y[:,0]), 1)
        #     idx = idx.reshape(-1, 2)
        #     y0 = y[idx[:,0]]
        #     yt = y[idx[:,1]]
        
        return y
            
        
def linear_combination(betas, npis, npis_schedule):
    npis_effect = np.array([npi.effect for npi in npis]) 
    npis_effect_schedule = npis_schedule * npis_effect
    npis_effect_schedule[npis_effect_schedule == 0] = 1
    return np.sum((npis_effect_schedule.T * betas).T, axis=1)


def non_linear_combination(betas, npis, npis_schedule):
    npis_effect = np.array([npi.effect for npi in npis]) 
    npis_effect_schedule = npis_schedule * npis_effect
    return betas**np.sum(npis_effect_schedule, axis=1)
    