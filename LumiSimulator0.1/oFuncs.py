import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
import csv
import ROOT as r
from array import array

def SG(x,A,mu,sigma,c):
	return A*np.exp(-(x-mu)**2/2/sigma**2)+c


def DG_strange(x,A,Frac,Mean,Sigma,SigmaRatio,c):
	#[2]*([3]*exp(-(x-[4])**2/(2*([0]*[1]/([3]*[1]+1-[3]))**2)) + (1-[3])*exp(-(x-[4])**2/(2*([0]/([3]*[1]+1-[3]))**2))
	return A*(	Frac*np.exp(-(x-Mean)**2. / (2.*(np.float64(Sigma)*np.float64(SigmaRatio)/(np.float64(Frac)*np.float64(SigmaRatio)+1.-np.float64(Frac)))**2.) )	
								+	(1. - np.float64(Frac))*np.exp(-(x-Mean)**2. / (2.*(np.float64(Sigma)/(np.float64(Frac)*np.float64(SigmaRatio)+1.-np.float64(Frac)))**2.) )	) + c

def DG_normal(x,A,Frac,Mean,Sigma1,Sigma2,c):
	return A*(	Frac*np.exp(-(x-Mean)**2/2/Sigma1**2) + (1-Frac)*np.exp(-(x-Mean)**2/2/Sigma2**2)	)+c



def eval_profile(offsets, profile, label):

	if profile.get_sgdg() == "SG":
	
		params	= profile.get_pv(label) 
		
		return SG(offsets, params[0], params[1], params[2], params[3])
	else:
		params	= profile.get_pv(label) 
		
		return DG_strange(offsets, params[0], params[1], params[2], params[3], params[4], params[5])
		
def calcTheorWeights(trueLambdas, param, profile):

	temp	= 1. / np.sqrt(	(np.exp(trueLambdas)-1)/param.get_nOrbits() + np.exp(trueLambdas)**2/param.get_nOrbits()**2 * (profile.get_noise()[1])) 
	
	
	for i in range(len(temp)):
		if np.isinf(temp[i]):
			temp[i]=-0.05
	temp	/= max(i for i in temp if i > 0)		
	
	
	#print trueLambdas , temp

	return temp

