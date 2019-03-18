# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 08:40:04 2019

@author: shonnet
@author: jvittone
"""

#Importation des librairies
import numpy as np
import fonction as fct
from sklearn.decomposition import PCA

#################Création de la fonction de calcul de PCA########################
def PCAcalcul(A,B,X,Y,val):
	#Définition de la précision du PCA
	pca=PCA(n_components=val)
	
	#Calcul du PCA sur l'ensemble d'entrainement
	tabPCA = pca.fit_transform(X)
	
	#Calcul du PCA sur l'ensemble de développement
	testPCA = pca.transform(A)

	#Création du tableau stockant les différentes classes
	classe2 = [0]*10
	for i in range(10):
		classe2[i]=tabPCA[Y==i]

	#Calcul des barycentre des classes d'entrainement après PCA
	BarycentrePCA = fct.calculBaryClasse(classe2)

	#Calcul des classes des points pour l'ensemble de développement après PCA
	classeTest2 = fct.PlusProche(testPCA,BarycentrePCA)

	#Calcul du nombre d'erreur 
	nbErreurPCA = fct.calculErreur(B,classeTest2)

	#Affichage du taux d'erreur
	s=val*100
	print("Taux d'erreur du PCA de ",s,"% : ")
	print((nbErreurPCA*1.0)/(len(B)*1.0)*100)
	
def PCAcalculSansAffichage(A,B,X,Y,val):
	#Définition de la précision du PCA
	pca=PCA(n_components=val)
	
	#Calcul du PCA sur l'ensemble d'entrainement
	tabPCA = pca.fit_transform(X)
	
	#Calcul du PCA sur l'ensemble de développement
	testPCA = pca.transform(A)

	#Création du tableau stockant les différentes classes
	classe2 = [0]*10
	for i in range(10):
		classe2[i]=tabPCA[Y==i]

	#Calcul des barycentre des classes d'entrainement après PCA
	BarycentrePCA = fct.calculBaryClasse(classe2)

	#Calcul des classes des points pour l'ensemble de développement après PCA
	classeTest2 = fct.PlusProche(testPCA,BarycentrePCA)

	#Calcul du nombre d'erreur 
	nbErreurPCA = fct.calculErreur(B,classeTest2)

	return (nbErreurPCA*1.0)/(len(B)*1.0)*100