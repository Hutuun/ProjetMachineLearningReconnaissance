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

def PCAcalcul(A,B,X,Y,val):
	pca=PCA(n_components=val)
	tabPCA = pca.fit_transform(X)
	testPCA = pca.transform(A)

	classe2 = [0]*10
	for i in range(10):
		classe2[i]=tabPCA[Y==i]

	BarycentrePCA = fct.calculBaryClasse(classe2)

	classeTest2 = fct.PlusProche(testPCA,BarycentrePCA)

	nbErreurPCA = fct.calculErreur(B,classeTest2)

	s=val*100
	print("Taux d'erreur du PCA de ",s,"% : ")
	print((nbErreurPCA*1.0)/(len(B)*1.0)*100)