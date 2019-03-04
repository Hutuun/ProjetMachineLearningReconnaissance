# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 08:40:04 2019

@author: shonnet
@author: jvittone
"""
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
	
def calculErreur(B,Test):
	nbErreur=0
	for i in range(len(B)):
		if B[i]!=Test[i]:
			nbErreur+=1
	return nbErreur
	
def calculSVM(X,Y,A,B):
	clf = SVC(gamma='auto')
	clf.fit(X,Y)

	erreur=0
	for i in range(len(A)): 
		x=clf.predict([A[i]])
		if x!=B[i]:
			erreur=erreur+1

	print("Taux d'erreur du SVM : ")
	print(((erreur*1.0)/(len(A)*1.0))*100)
	
def calculPointProche(X,Y,A,B):
	erreur=0
	neigh = NearestNeighbors(n_neighbors=1)
	neigh.fit(X)
	tempo=neigh.kneighbors(A, return_distance=False)
	for i in range(len(tempo)):
		if Y[tempo[i]]!=B[i]:
			erreur=erreur+1
		
	print("Taux d'erreur du plus proche voisin : ")
	print(((erreur*1.0)/(len(A)*1.0))*100)
	
def maxi(tempo):
	res = -1
	taille = 0
	for i in range(len(tempo)):
		if tempo[i]>taille:
			res=i
			taille=tempo[i]
		#elif tempo[i]==taille:
			#res=-1
	return res

def choixMoyen(A,Y,nbclasse):
	tempo=[0]*nbclasse
	for i in nbclasse:
		tempo[Y[A[i]]-1]+=1
	res=maxi(tempo)
	return res

def calculPointsProches(X,Y,A,B,nbclasse,voisins):
	erreur=0
	neigh = NearestNeighbors(n_neighbors=voisins)
	neigh.fit(X)
	tempo=neigh.kneighbors(A, return_distance=False)
	res=[0]*len(B)
	for i in range(len(tempo)):
		res[i]=choixMoyen(tempo[i],Y,nbclasse)
	for i in range(len(res)):
		if res[i]!=B[i]:
			erreur=erreur+1
	
	print("Taux d'erreur des ",voisins," plus proches voisins : ")
	print(((erreur*1.0)/(len(A)*1.0))*100)