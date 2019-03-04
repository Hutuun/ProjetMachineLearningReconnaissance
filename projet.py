# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 08:40:04 2019

@author: shonnet
@author: jvittone
"""

import numpy as np
import matplotlib.pyplot as plt
import fonction as fct
import deepPCA as dpca
from sklearn.decomposition import PCA

X = np.load("data/trn_img.npy")
Y = np.load("data/trn_lbl.npy")
A = np.load("data/dev_img.npy")
B = np.load("data/dev_lbl.npy")

classe = [0]*10
for i in range(10):
	classe[i]=X[Y==i]

Barycentre = fct.calculBaryClasse(classe)

classeTest = fct.PlusProche(A,Barycentre)

nbErreur = fct.calculErreur(B,classeTest)

print("Taux d'erreur du plus proche : ")
print((nbErreur*1.0)/(len(B)*1.0)*100)

pca=PCA(n_components=0.75)
tabPCA = pca.fit_transform(X)
testPCA = pca.transform(A)

dpca.PCAcalcul(A,B,X,Y,0.95)

dpca.PCAcalcul(A,B,X,Y,0.75)

dpca.PCAcalcul(A,B,X,Y,0.5)

dpca.PCAcalcul(A,B,X,Y,0.25)