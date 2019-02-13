# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 08:40:04 2019

@author: shonnet
"""

import numpy as np
import matplotlib.pyplot as plt

X = np.load("data/trn_img.npy")
Y = np.load("data/trn_lbl.npy")


img =X[250].reshape(28,28)
plt.imshow(img,plt.cm.gray)
plt.show()
