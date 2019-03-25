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
ImagesTrain = np.load("../trn_img.npy")
LabelTrain = np.load("../trn_lbl.npy")
ImagesDev = np.load("../dev_img.npy")
LabelDev = np.load("../dev_lbl.npy")
test = np.load("../tst_img.npy")

#################Affichage d'une courbe de comparaison pour ImagesTrain plus proches points########################
inter=20
affi=[0]*inter
for i in range(1,inter+1):
	print(i)
	affi[i-1]=fct.calculPointsProchesSansAffichage(ImagesTrain,LabelTrain,ImagesDev,LabelDev,10,i)
plt.figure(2)
plt.plot(affi,'ro')
plt.show()

##################Génération des résultats#########################################
tempo=fct.calculPointsProchesRes(ImagesTrain,LabelTrain,test,10,5)
np.save("test.npy",tempo)
tempo=fct.calculPointsProchesRes(ImagesTrain,LabelTrain,ImagesDev,10,5)
np.save("test2.npy",tempo)
K=np.load("test2.npy")

#Calcul du nombre d'erreur
for i in range(len(res)):
	if K[i]!=LabelDev[i]:
		erreur=erreur+1
	
#Affichage du taux d'erreur
print("Taux d'erreur des ",voisins," plus proches voisins : ")
print(((erreur*1.0)/(len(K)*1.0))*100)