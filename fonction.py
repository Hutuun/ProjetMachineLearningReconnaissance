# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 08:40:04 2019

@author: shonnet
@author: jvittone
"""
import numpy as np
import sklearn.decomposition as dsk
import sklearn as svm


#Il s'agit de la boucle d'appel sur les classes
def calculBaryClasse(C):
	res=[0]*10
	for i in range(10):
		k=(calculBary(C[i]))
		res[i]=k
	return res

def calculBary(C):
	res=np.mean(C,axis =0)
	return res
	
def PlusProche(C,Bary):
	res=[0]*len(C)
	for i in range(len(C)):
		res[i]=calculPlusProche(C[i],Bary)
	return res

def calculPlusProche(C,Bary):
	res = [0]*10
	for i in range(10):
		tempo=0
		K=C-Bary[i]
		tempo+=K*K
		#for j in range(784):
		#	tempo+=(C[j]-Bary[i][j])*(C[j]-Bary[i][j])
		res[i]=tempo
	return minPos(res)
	
def minPos(C):
	mini=C[0]
	Pos=0
	for i in range(10):
		if mini>C[i]:
			mini=C[i]
			Pos=i
	return Pos
	
def calculErreur(B,Test):
	nbErreur=0
	for i in range(len(B)):
		if B[i]!=Test[i]:
			nbErreur+=1
	return nbErreur