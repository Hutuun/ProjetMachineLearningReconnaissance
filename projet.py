# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 08:40:04 2019

@author: shonnet
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn as dsk
import fonction as fct
X = np.load("data/trn_img.npy")
Y = np.load("data/trn_lbl.npy")
A = np.load("data/dev_img.npy")
B = np.load("data/dev_lbl.npy")


#img =X[250].reshape(28,28)
#plt.imshow(img,plt.cm.gray)
#plt.show()

Classe0 = X[Y==0]
Classe1 = X[Y==1]
Classe2 = X[Y==2]
Classe3 = X[Y==3]
Classe4 = X[Y==4]
Classe5 = X[Y==5]
Classe6 = X[Y==6]
Classe7 = X[Y==7]
Classe8 = X[Y==8]
Classe9 = X[Y==9]

classe = [Classe0,Classe1,Classe2, Classe3,Classe4,Classe5,Classe6,Classe7,Classe8,Classe9]

Barycentre = fct.calculBaryClasse(classe)

classeTest = fct.PlusProche(A,Barycentre)

nbErreur = fct.calculErreur(B,classeTest)

print("Taux d'erreur : ")
print((nbErreur*1.0)/(len(B)*1.0)*100)

