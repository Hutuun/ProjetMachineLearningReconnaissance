# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 08:40:04 2019

@author: shonnet
@author: jvittone
"""

#Importation des librairies
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

#Il s'agit de la boucle d'appel sur les classes
def calculBaryClasse(C):
	res=[0]*10
	for i in range(10):
		res[i]=np.mean(C[i],axis =0)
	return res
	
def PlusProche(C,Bary):
	res=[0]*len(C)
	for i in range(len(C)):
		res[i]=calculPlusProche(C[i],Bary)
	return res

def calculPlusProche(C,Bary):
	res = 0
	min = np.sum((C-Bary[0])**2)
	for i in range(1,10):
		tempo=np.sum((C-Bary[i])**2)
		if tempo<min:
			min=tempo
			res=i
	return res
	
#Calcul le nombre d'erreur entre LabelDev et Test
def calculErreur(LabelDev,Test):
	nbErreur=0
	for i in range(len(LabelDev)):
		if LabelDev[i]!=Test[i]:
			nbErreur+=1
	return nbErreur
	
#Utilisation de la méthode SVM
def calculSVM(ImagesTrain,LabelTrain,ImagesDev,LabelDev):
	#Paramétrage de la SVM
	clf = SVC(gamma='auto')
	
	#Entrainement de la SVM
	clf.fit(ImagesTrain,LabelTrain)
	
	#Calcul du nombre d'erreur
	erreur=0
	for i in range(len(ImagesDev)): 
		#Calcul de la classe de ImagesDev
		x=clf.predict([ImagesDev[i]])
		if x!=LabelDev[i]:
			erreur=erreur+1

	#Affichage du taux d'erreur
	print("Taux d'erreur du SVM : ")
	print(((erreur*1.0)/(len(ImagesDev)*1.0))*100)
	
#Calcul du point le plus proche à chaque fois et affichage du taux d'erreur 
def calculPointProche(ImagesTrain,LabelTrain,ImagesDev,LabelDev):
	erreur=0
	
	#Indication du nombre de voisins
	neigh = NearestNeighbors(n_neighbors=1)
	
	#Entrainement
	neigh.fit(ImagesTrain)
	
	#Calcul de la classe pour tous les éléments de l'ensemble de développement
	tempo=neigh.kneighbors(ImagesDev, return_distance=False)
	
	#Calcul du nombre d'erreur
	for i in range(len(tempo)):
		if LabelTrain[tempo[i]]!=LabelDev[i]:
			erreur=erreur+1
		
	#Affichage du taux d'erreur
	print("Taux d'erreur du plus proche voisin : ")
	print(((erreur*1.0)/(len(ImagesDev)*1.0))*100)

#Calcul de la position du maximum dans le tableau tempo
def posMaxi(init,tempo):
	res = init
	taille = 0
	for i in range(len(tempo)):
		if tempo[i]>taille:
			res=i
			taille=tempo[i]
		#elif tempo[i]==taille:
			#res=-1
	return res

#Calcul des point les plus proches à chaque fois, affichage du taux d'erreur et matrice de confusion
def calculPointsProches(ImagesTrain,LabelTrain,ImagesDev,LabelDev,nbclasse,voisins):
	classe=[0]*10
	erreur=0
	
	#Indication du nombre de voisins
	neigh = NearestNeighbors(n_neighbors=voisins)
	
	#Entrainement
	neigh.fit(ImagesTrain)
	
	#Calcul de la classe pour tous les éléments de l'ensemble de développement
	tempo=neigh.kneighbors(ImagesDev, return_distance=False)
	
	#Calcul de la classe des différents points de l'ensemble de développement
	res = [0]*len(tempo)
	for i in range(len(tempo)):
		classe=[0]*10
		for j in tempo[i]:
			k=LabelTrain[j]
			classe[k]=classe[k]+1
		res[i]=posMaxi(LabelTrain[tempo[i]],classe)
	
	#Création de la matrice de confusion_matrix
	
	MatriceDeConfusion = confusion_matrix(LabelDev,res)
	
	#Calcul du nombre d'erreur
	for i in range(len(res)):
		if res[i]!=LabelDev[i]:
			erreur=erreur+1
	
	#Affichage du taux d'erreur
	print("Taux d'erreur des ",voisins," plus proches voisins : ")
	print(((erreur*1.0)/(len(ImagesDev)*1.0))*100)
	
	return MatriceDeConfusion;
	
#Calcul des point les plus proches à chaque fois 
def calculPointsProchesSansAffichage(ImagesTrain,LabelTrain,ImagesDev,LabelDev,nbclasse,voisins):
	classe=[0]*10
	erreur=0
	
	#Indication du nombre de voisins
	neigh = NearestNeighbors(n_neighbors=voisins)
	
	#Entrainement
	neigh.fit(ImagesTrain)
	
	#Calcul de la classe pour tous les éléments de l'ensemble de développement
	tempo=neigh.kneighbors(ImagesDev, return_distance=False)
	
	#Calcul de la classe des différents points de l'ensemble de développement
	res = [0]*len(tempo)
	for i in range(len(tempo)):
		classe=[0]*10
		for j in tempo[i]:
			k=LabelTrain[j]
			classe[k]=classe[k]+1
		res[i]=posMaxi(LabelTrain[tempo[i]],classe)
	
	#Calcul du nombre d'erreur
	for i in range(len(res)):
		if res[i]!=LabelDev[i]:
			erreur=erreur+1
	
	return ((erreur*1.0)/(len(ImagesDev)*1.0))*100
	
#Calcul des point les plus proches à chaque fois et renvoie des valeurs prédites
def calculPointsProchesRes(ImagesTrain,LabelTrain,ImagesDev,nbclasse,voisins):
	classe=[0]*10
	erreur=0
	
	#Indication du nombre de voisins
	neigh = NearestNeighbors(n_neighbors=voisins)
	
	#Entrainement
	neigh.fit(ImagesTrain)
	
	#Calcul de la classe pour tous les éléments de l'ensemble de développement
	tempo=neigh.kneighbors(ImagesDev, return_distance=False)
	
	#Calcul de la classe des différents points de l'ensemble de développement
	res = [0]*len(tempo)
	for i in range(len(tempo)):
		classe=[0]*10
		for j in tempo[i]:
			k=LabelTrain[j]
			classe[k]=classe[k]+1
		res[i]=posMaxi(LabelTrain[tempo[i]],classe)
	
	return res;