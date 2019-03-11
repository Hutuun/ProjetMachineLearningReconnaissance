# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 08:40:04 2019

@author: shonnet
@author: jvittone
"""

#Importation des librairies
import numpy as np
import matplotlib.pyplot as plt
import fonction as fct
import deepPCA as dpca

#Chargement des fichiers d'entrainement et de développement
X = np.load("data/trn_img.npy")
Y = np.load("data/trn_lbl.npy")
A = np.load("data/dev_img.npy")
B = np.load("data/dev_lbl.npy")

#Création du tableau stockant les différentes classes
classe = [0]*10
for i in range(10):
	classe[i]=X[Y==i]

#################Plus proche barycentre########################
#Calcul des barycentre des classes d'entrainement
Barycentre = fct.calculBaryClasse(classe)

#Calcul des classes des points pour l'ensemble de développement
classeTest = fct.PlusProche(A,Barycentre)

#Calcul du nombre d'erreur 
nbErreur = fct.calculErreur(B,classeTest)

#Affichage du taux d'erreur
print("Taux d'erreur du plus proche : ")
print((nbErreur*1.0)/(len(B)*1.0)*100)

#################Variation du PCA########################
#Calcul pour un PCA de 95%
dpca.PCAcalcul(A,B,X,Y,0.95)

#Calcul pour un PCA de 75%
dpca.PCAcalcul(A,B,X,Y,0.75)

#Calcul pour un PCA de 50%
dpca.PCAcalcul(A,B,X,Y,0.5)

#Calcul pour un PCA de 25%
dpca.PCAcalcul(A,B,X,Y,0.25)

#Calcul pour un PCA de 5%
dpca.PCAcalcul(A,B,X,Y,0.05)

#################SVM########################
#fct.calculSVM(X,Y,A,B)

#################Plus proche point########################
fct.calculPointProche(X,Y,A,B)	

#################X plus proches points########################
#Calcul en fonction du point le plus proche
fct.calculPointsProches(X,Y,A,B,10,1)

#Calcul en fonction des trois points les plus proches
fct.calculPointsProches(X,Y,A,B,10,3)

#Calcul en fonction des cinq points les plus proches
fct.calculPointsProches(X,Y,A,B,10,5)

#Calcul en fonction des dix points les plus proches
fct.calculPointsProches(X,Y,A,B,10,10)

#Calcul en fonction des vingt points les plus proches
fct.calculPointsProches(X,Y,A,B,10,20)

#Calcul en fonction des cent points les plus proches
fct.calculPointsProches(X,Y,A,B,10,100)

#Calcul en fonction des mille points les plus proches
fct.calculPointsProches(X,Y,A,B,10,1000)

#################Affichage d'une courbe de comparaison pour X plus proches points########################
inter=20
affi=[0]*inter
y=[0]*inter
for i in range(1,inter):
	affi[i]=fct.calculPointsProches(X,Y,A,B,10,i)
	y[i]=i
plt.plot(y,affi)
plt.show()