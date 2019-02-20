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

Classe0 = X[Y==0]
Classe1 = X[Y==1]
Classe2 = X[Y==2]
Classe3 = X[Y==3]
Classe4 = X[Y==4]
Classe5 = X[Y==5]
Classe6 = X[Y==6]
Classe7 = X[Y==7]
Classe8 = X[Y==8]
Classe9 = X[Y==9]

classe = [Classe0,Classe1,Classe2, Classe3,Classe4,Classe5,Classe6,Classe7,Classe8,Classe9]

Barycentre = fct.calculBaryClasse(classe)

classeTest = fct.PlusProche(A,Barycentre)

nbErreur = fct.calculErreur(B,classeTest)

print("Taux d'erreur du plus proche : ")
print((nbErreur*1.0)/(len(B)*1.0)*100)

pca=PCA(n_components=0.95)
tabPCA = pca.fit_transform(X)
testPCA = pca.transform(A)

Classe0 = tabPCA[Y==0]
Classe1 = tabPCA[Y==1]
Classe2 = tabPCA[Y==2]
Classe3 = tabPCA[Y==3]
Classe4 = tabPCA[Y==4]
Classe5 = tabPCA[Y==5]
Classe6 = tabPCA[Y==6]
Classe7 = tabPCA[Y==7]
Classe8 = tabPCA[Y==8]
Classe9 = tabPCA[Y==9]

classe2 = [Classe0,Classe1,Classe2, Classe3,Classe4,Classe5,Classe6,Classe7,Classe8,Classe9]

BarycentrePCA = fct.calculBaryClasse(classe2)

classeTest2 = fct.PlusProche(testPCA,BarycentrePCA)

nbErreurPCA = fct.calculErreur(B,classeTest2)

print("Taux d'erreur du PCA à 95% : ")
print((nbErreurPCA*1.0)/(len(B)*1.0)*100)

pca=PCA(n_components=0.75)
tabPCA = pca.fit_transform(X)
testPCA = pca.transform(A)

Classe0 = tabPCA[Y==0]
Classe1 = tabPCA[Y==1]
Classe2 = tabPCA[Y==2]
Classe3 = tabPCA[Y==3]
Classe4 = tabPCA[Y==4]
Classe5 = tabPCA[Y==5]
Classe6 = tabPCA[Y==6]
Classe7 = tabPCA[Y==7]
Classe8 = tabPCA[Y==8]
Classe9 = tabPCA[Y==9]

classe2 = [Classe0,Classe1,Classe2, Classe3,Classe4,Classe5,Classe6,Classe7,Classe8,Classe9]

BarycentrePCA = fct.calculBaryClasse(classe2)

classeTest2 = fct.PlusProche(testPCA,BarycentrePCA)

nbErreurPCA = fct.calculErreur(B,classeTest2)

print("Taux d'erreur du PCA à 75% : ")
print((nbErreurPCA*1.0)/(len(B)*1.0)*100)

pca=PCA(n_components=0.5)
tabPCA = pca.fit_transform(X)
testPCA = pca.transform(A)

Classe0 = tabPCA[Y==0]
Classe1 = tabPCA[Y==1]
Classe2 = tabPCA[Y==2]
Classe3 = tabPCA[Y==3]
Classe4 = tabPCA[Y==4]
Classe5 = tabPCA[Y==5]
Classe6 = tabPCA[Y==6]
Classe7 = tabPCA[Y==7]
Classe8 = tabPCA[Y==8]
Classe9 = tabPCA[Y==9]

classe2 = [Classe0,Classe1,Classe2, Classe3,Classe4,Classe5,Classe6,Classe7,Classe8,Classe9]

BarycentrePCA = fct.calculBaryClasse(classe2)

classeTest2 = fct.PlusProche(testPCA,BarycentrePCA)

nbErreurPCA = fct.calculErreur(B,classeTest2)

print("Taux d'erreur du PCA à 50% : ")
print((nbErreurPCA*1.0)/(len(B)*1.0)*100)

pca=PCA(n_components=0.25)
tabPCA = pca.fit_transform(X)
testPCA = pca.transform(A)

Classe0 = tabPCA[Y==0]
Classe1 = tabPCA[Y==1]
Classe2 = tabPCA[Y==2]
Classe3 = tabPCA[Y==3]
Classe4 = tabPCA[Y==4]
Classe5 = tabPCA[Y==5]
Classe6 = tabPCA[Y==6]
Classe7 = tabPCA[Y==7]
Classe8 = tabPCA[Y==8]
Classe9 = tabPCA[Y==9]

classe2 = [Classe0,Classe1,Classe2, Classe3,Classe4,Classe5,Classe6,Classe7,Classe8,Classe9]

BarycentrePCA = fct.calculBaryClasse(classe2)

classeTest2 = fct.PlusProche(testPCA,BarycentrePCA)

nbErreurPCA = fct.calculErreur(B,classeTest2)

print("Taux d'erreur du PCA à 25% : ")
print((nbErreurPCA*1.0)/(len(B)*1.0)*100)