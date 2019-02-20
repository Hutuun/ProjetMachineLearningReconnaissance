# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 08:40:04 2019

@author: shonnet
@author: jvittone
"""

import numpy as np
import matplotlib.pyplot as plt
import fonction as fct
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

pca=PCA(n_components=0.95)
tabPCA = pca.fit_transform(X)
testPCA = pca.transform(A)

classe2 = [0]*10
for i in range(10):
	classe2[i]=tabPCA[Y==i]

BarycentrePCA = fct.calculBaryClasse(classe2)

classeTest2 = fct.PlusProche(testPCA,BarycentrePCA)

nbErreurPCA = fct.calculErreur(B,classeTest2)

print("Taux d'erreur du PCA à 95% : ")
print((nbErreurPCA*1.0)/(len(B)*1.0)*100)

pca=PCA(n_components=0.75)
tabPCA = pca.fit_transform(X)
testPCA = pca.transform(A)

classe2 = [0]*10
for i in range(10):
	classe2[i]=tabPCA[Y==i]

BarycentrePCA = fct.calculBaryClasse(classe2)

classeTest2 = fct.PlusProche(testPCA,BarycentrePCA)

nbErreurPCA = fct.calculErreur(B,classeTest2)

print("Taux d'erreur du PCA à 75% : ")
print((nbErreurPCA*1.0)/(len(B)*1.0)*100)

pca=PCA(n_components=0.5)
tabPCA = pca.fit_transform(X)
testPCA = pca.transform(A)

classe2 = [0]*10
for i in range(10):
	classe2[i]=tabPCA[Y==i]

BarycentrePCA = fct.calculBaryClasse(classe2)

classeTest2 = fct.PlusProche(testPCA,BarycentrePCA)

nbErreurPCA = fct.calculErreur(B,classeTest2)

print("Taux d'erreur du PCA à 50% : ")
print((nbErreurPCA*1.0)/(len(B)*1.0)*100)

pca=PCA(n_components=0.25)
tabPCA = pca.fit_transform(X)
testPCA = pca.transform(A)

classe2 = [0]*10
for i in range(10):
	classe2[i]=tabPCA[Y==i]

BarycentrePCA = fct.calculBaryClasse(classe2)

classeTest2 = fct.PlusProche(testPCA,BarycentrePCA)

nbErreurPCA = fct.calculErreur(B,classeTest2)

print("Taux d'erreur du PCA à 25% : ")
print((nbErreurPCA*1.0)/(len(B)*1.0)*100)