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
	
#Calcul le nombre d'erreur entre B et Test
def calculErreur(B,Test):
	nbErreur=0
	for i in range(len(B)):
		if B[i]!=Test[i]:
			nbErreur+=1
	return nbErreur
	
#Utilisation de la méthode SVM
def calculSVM(X,Y,A,B):
	#Paramétrage de la SVM
	clf = SVC(gamma='auto')
	
	#Entrainement de la SVM
	clf.fit(X,Y)
	
	#Calcul du nombre d'erreur
	erreur=0
	for i in range(len(A)): 
		#Calcul de la classe de A
		x=clf.predict([A[i]])
		if x!=B[i]:
			erreur=erreur+1

	#Affichage du taux d'erreur
	print("Taux d'erreur du SVM : ")
	print(((erreur*1.0)/(len(A)*1.0))*100)
	
#Calcul du point le plus proche à chaque fois et affichage du taux d'erreur 
def calculPointProche(X,Y,A,B):
	erreur=0
	
	#Indication du nombre de voisins
	neigh = NearestNeighbors(n_neighbors=1)
	
	#Entrainement
	neigh.fit(X)
	
	#Calcul de la classe pour tous les éléments de l'ensemble de développement
	tempo=neigh.kneighbors(A, return_distance=False)
	
	#Calcul du nombre d'erreur
	for i in range(len(tempo)):
		if Y[tempo[i]]!=B[i]:
			erreur=erreur+1
		
	#Affichage du taux d'erreur
	print("Taux d'erreur du plus proche voisin : ")
	print(((erreur*1.0)/(len(A)*1.0))*100)

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

#Calcul des point les plus proches à chaque fois et affichage du taux d'erreur 
def calculPointsProches(X,Y,A,B,nbclasse,voisins):
	classe=[0]*10
	erreur=0
	
	#Indication du nombre de voisins
	neigh = NearestNeighbors(n_neighbors=voisins)
	
	#Entrainement
	neigh.fit(X)
	
	#Calcul de la classe pour tous les éléments de l'ensemble de développement
	tempo=neigh.kneighbors(A, return_distance=False)
	
	#Calcul de la classe des différents points de l'ensemble de développement
	res = [0]*len(tempo)
	for i in range(len(tempo)):
		classe=[0]*10
		for j in tempo[i]:
			k=Y[j]
			classe[k]=classe[k]+1
		res[i]=posMaxi(Y[tempo[i]],classe)
	
	#Calcul du nombre d'erreur
	for i in range(len(res)):
		if res[i]!=B[i]:
			erreur=erreur+1
	
	#Affichage du taux d'erreur
	print("Taux d'erreur des ",voisins," plus proches voisins : ")
	print(((erreur*1.0)/(len(A)*1.0))*100)