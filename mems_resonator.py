#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 16:01:47 2019

@author: sharon
"""

import numpy as np
import matplotlib.pyplot as plt
from cmath import polar
from cmath import exp
from cmath import sin
import pandas as pd

def G(w):
    # Frequency domain equation
    G_s = k_dc*w_n**2/(-w**2 + (2*zeta*w_n*w)*1j + w_n**2)
    return polar(G_s)

def C(t):
    # Time response equation
    C_t = 1 - exp(-zeta*w_n*t)*sin(w_n*(1-zeta**2)**(0.5)*t + phi)/(1-zeta**2)**(0.5)
    return C_t
    
    
    
# dc gain
k_dc = 1

# damping ratio
zeta = 0.2

# natural frequency
order = 1000
w_n = 12*order

# phase shift
phi = 2

# freqeuncy this is variable
w_range = np.arange(0.1,w_n + order*3,0.5)
t_range = np.arange(0,0.01,0.00001)

# get the data
Amp = [G(w)[0] for w in w_range]
Phase = [G(w)[1] for w in w_range]
Amp_t = [C(t) for t in t_range]

# dataset dictionary
Freq_response = {
        'Frequency': w_range,
        'Amplitude': Amp
        }

Phase_response = {
        'Frequency': w_range,
        'Phase': Phase
        }

Time_response = {
        'Time':t_range,
        'Amplitude':Amp_t
        }
# data frame for saving purpose
Dataset1 = pd.DataFrame(data = Freq_response)
Dataset1.to_csv('freq_response.csv')

Dataset2 = pd.DataFrame(data = Phase_response)
Dataset2.to_csv('phase_response.csv')

Dataset3 = pd.DataFrame(data = Time_response)
Dataset3.to_csv('Time_response.csv')

plt.figure(figsize = [8,12])

plt.subplot(511)
plt.plot(w_range,Amp)
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.title('Frequency Response')

plt.subplot(513)
plt.plot(w_range,Phase)
plt.xlabel('Frequency(Hz)')
plt.ylabel('Phase(degree)')
plt.title('Phase Response')

plt.subplot(515)
plt.plot(t_range,Amp_t)
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')
plt.title('Time Response')

plt.show()
