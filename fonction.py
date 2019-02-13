# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 08:40:04 2019

@author: shonnet
@author: jvittone
"""
import numpy as np

def calculBaryClasse(C):
	res=[]*10
	for i in range(10):
		k=(calculBary(C[i]))
		res[i]=k
	return res

def calculBary(C):
	res=[0]*784
	print(C[0])
	return res