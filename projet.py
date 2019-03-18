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
import time
from sklearn.metrics import confusion_matrix

#Chargement des fichiers d'entrainement et de développement
X = np.load("data/trn_img.npy")
Y = np.load("data/trn_lbl.npy")
A = np.load("data/dev_img.npy")
B = np.load("data/dev_lbl.npy")

#Création du tableau stockant les différentes classes
classe = [0]*10
for i in range(10):
	classe[i]=X[Y==i]

	
confus = [0]*8
	
#################Plus proche barycentre########################
start=time.time()

#Calcul des barycentre des classes d'entrainement
Barycentre = fct.calculBaryClasse(classe)

#Calcul des classes des points pour l'ensemble de développement
classeTest = fct.PlusProche(A,Barycentre)

confus[0] = confusion_matrix(B,classeTest)

#Calcul du nombre d'erreur 
nbErreur = fct.calculErreur(B,classeTest)

#Affichage du taux d'erreur
print("Taux d'erreur du plus proche : ")
print((nbErreur*1.0)/(len(B)*1.0)*100)

print("Temps d'exécution")
end=time.time()
print(end - start)

#################Variation du PCA########################
start=time.time()

#Calcul pour un PCA de 95%
dpca.PCAcalcul(A,B,X,Y,0.95)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#Calcul pour un PCA de 75%
dpca.PCAcalcul(A,B,X,Y,0.75)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#Calcul pour un PCA de 50%
dpca.PCAcalcul(A,B,X,Y,0.5)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#Calcul pour un PCA de 25%
dpca.PCAcalcul(A,B,X,Y,0.25)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#Calcul pour un PCA de 5%
dpca.PCAcalcul(A,B,X,Y,0.05)

print("Temps d'exécution")
end=time.time()
print(end - start)

#################SVM########################
start=time.time()

#fct.calculSVM(X,Y,A,B)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#################Plus proche point########################
start=time.time()

fct.calculPointProche(X,Y,A,B)

print("Temps d'exécution")
end=time.time()
print(end - start)

#################X plus proches points########################
start=time.time()

#Calcul en fonction du point le plus proche
confus[1] = fct.calculPointsProches(X,Y,A,B,10,1)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#Calcul en fonction des trois points les plus proches
confus[2] = fct.calculPointsProches(X,Y,A,B,10,3)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#Calcul en fonction des cinq points les plus proches
confus[3] = fct.calculPointsProches(X,Y,A,B,10,5)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#Calcul en fonction des dix points les plus proches
confus[4] = fct.calculPointsProches(X,Y,A,B,10,10)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#Calcul en fonction des vingt points les plus proches
confus[5] = fct.calculPointsProches(X,Y,A,B,10,20)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#Calcul en fonction des cent points les plus proches
confus[6] = fct.calculPointsProches(X,Y,A,B,10,100)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#Calcul en fonction des mille points les plus proches
confus[7] = fct.calculPointsProches(X,Y,A,B,10,1000)

print("Temps d'exécution")
end=time.time()
print(end - start)

for i in confus:
	print(i)

#################Affichage d'une courbe de comparaison pour X plus proches points########################
inter=20
affi=[0]*inter
y=[0]*inter
for i in range(1,inter+1):
	affi[i-1]=fct.calculPointsProchesSansAffichage(X,Y,A,B,10,i)
	y[i-1]=i
plt.plot(y,affi,'ro')
plt.show()