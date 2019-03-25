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

#Création du tableau stockant les différentes classes
classe = [0]*10
for i in range(10):
	classe[i]=ImagesTrain[LabelTrain==i]

	
MatriceDeConfusion = [0]*8
	
#################Plus proche barycentre########################
start=time.time()

#Calcul des barycentre des classes d'entrainement
Barycentre = fct.calculBaryClasse(classe)

#Calcul des classes des points pour l'ensemble de développement
classeTest = fct.PlusProche(ImagesDev,Barycentre)

MatriceDeConfusion[0] = confusion_matrix(LabelDev,classeTest)

#Calcul du nombre d'erreur 
nbErreur = fct.calculErreur(LabelDev,classeTest)

#Affichage du taux d'erreur
print("Taux d'erreur du plus proche : ")
print((nbErreur*1.0)/(len(LabelDev)*1.0)*100)

print("Temps d'exécution")
end=time.time()
print(end - start)

#################Variation du PCA########################
start=time.time()

#Calcul pour un PCA de 95%
dpca.PCAcalcul(ImagesDev,LabelDev,ImagesTrain,LabelTrain,0.95)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#Calcul pour un PCA de 75%
dpca.PCAcalcul(ImagesDev,LabelDev,ImagesTrain,LabelTrain,0.75)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#Calcul pour un PCA de 50%
dpca.PCAcalcul(ImagesDev,LabelDev,ImagesTrain,LabelTrain,0.5)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#Calcul pour un PCA de 25%
dpca.PCAcalcul(ImagesDev,LabelDev,ImagesTrain,LabelTrain,0.25)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#Calcul pour un PCA de 5%
dpca.PCAcalcul(ImagesDev,LabelDev,ImagesTrain,LabelTrain,0.05)

print("Temps d'exécution")
end=time.time()
print(end - start)

#################Affichage d'une courbe de comparaison pour PCA########################
inter=19
affi=[0]*inter
for i in range(1,inter+1):
	affi[i-1]=dpca.PCAcalculSansAffichage(ImagesTrain,LabelTrain,ImagesDev,LabelDev,i*0.05)
plt.figure(1)
plt.plot(affi,'ro')
plt.show()

#################SVM########################
start=time.time()

fct.calculSVM(ImagesTrain,LabelTrain,ImagesDev,LabelDev)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#################Plus proche point########################
start=time.time()

fct.calculPointProche(ImagesTrain,LabelTrain,ImagesDev,LabelDev)

print("Temps d'exécution")
end=time.time()
print(end - start)

#################ImagesTrain plus proches points########################
start=time.time()

#Calcul en fonction du point le plus proche
MatriceDeConfusion[1] = fct.calculPointsProches(ImagesTrain,LabelTrain,ImagesDev,LabelDev,10,1)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#Calcul en fonction des trois points les plus proches
MatriceDeConfusion[2] = fct.calculPointsProches(ImagesTrain,LabelTrain,ImagesDev,LabelDev,10,3)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#Calcul en fonction des cinq points les plus proches
MatriceDeConfusion[3] = fct.calculPointsProches(ImagesTrain,LabelTrain,ImagesDev,LabelDev,10,5)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#Calcul en fonction des dix points les plus proches
MatriceDeConfusion[4] = fct.calculPointsProches(ImagesTrain,LabelTrain,ImagesDev,LabelDev,10,10)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#Calcul en fonction des vingt points les plus proches
MatriceDeConfusion[5] = fct.calculPointsProches(ImagesTrain,LabelTrain,ImagesDev,LabelDev,10,20)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#Calcul en fonction des cent points les plus proches
MatriceDeConfusion[6] = fct.calculPointsProches(ImagesTrain,LabelTrain,ImagesDev,LabelDev,10,100)

print("Temps d'exécution")
end=time.time()
print(end - start)

start=time.time()

#Calcul en fonction des mille points les plus proches
MatriceDeConfusion[7] = fct.calculPointsProches(ImagesTrain,LabelTrain,ImagesDev,LabelDev,10,1000)

print("Temps d'exécution")
end=time.time()
print(end - start)

for i in MatriceDeConfusion:
	print(i)

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
#tempo=calculPointsProchesRes(ImagesTrain,LabelTrain,test,10,5)
tempo=calculPointsProchesRes(ImagesTrain,LabelTrain,ImagesDev,10,5)
np.save("test.npy",tempo)
LabelTest=np.load("test.npy")

#Calcul du nombre d'erreur
for i in range(len(res)):
	if LabelTest[i]!=LabelDev[i]:
		erreur=erreur+1
	
#Affichage du taux d'erreur
print("Taux d'erreur des ",voisins," plus proches voisins : ")
print(((erreur*1.0)/(len(LabelTest)*1.0))*100)