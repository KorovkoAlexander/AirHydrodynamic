#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 02:20:59 2017

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,5, 1000)

y = 20*(np.exp(-((x - 2.5)**2)/(2)) -0.5)


plt.plot(x,y)
plt.show()







