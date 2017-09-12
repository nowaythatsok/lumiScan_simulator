
import numpy as np
from ROOT import TGraphErrors, TF1, gROOT
from array import array

from oFuncs import *
from Parameters import *
from Profile import *
from Cummulator import *


# [in Hz]
LHC_revolution_frequency = 11245




class Measurement:
	
	def __init__(self,profile,param_e,param_vdm,common):


		#copy raw data in here
		self._param_e		= param_e
		self._param_vdm		= param_vdm
		self._profile		= profile

		self._noise 		= profile.get_noise()


		#calculate the truelambda vaues 
		
		centerX	= profile.get_meanX()	
		widthX	= profile.get_sigmaX()
		
		centerY	= profile.get_meanY()	
		widthY	= profile.get_sigmaY()
		
		ne	= param_e.get_nOffOffsetPoints()
		nv	= param_vdm.get_nOffOffsetPoints()
		
		minrXe	= centerX - widthX * param_e.get_range()
		maxrXe	= centerX + widthX * param_e.get_range()
		minrXv	= centerX - widthX * param_vdm.get_range()
		maxrXv	= centerX + widthX * param_vdm.get_range()
		minrYe	= centerY - widthY * param_e.get_range()
		maxrYe	= centerY + widthY * param_e.get_range()
		minrYv	= centerY - widthY * param_vdm.get_range()
		maxrYv	= centerY + widthY * param_vdm.get_range()
	
		self._offsetsXe	= np.linspace(minrXe, maxrXe, ne)
		self._offsetsYe	= np.linspace(minrYe, maxrYe, ne)
		self._offsetsXv	= np.linspace(minrXv, maxrXv, nv)
		self._offsetsYv	= np.linspace(minrYv, maxrYv, nv)
		
		
		self._trueLambdaXe	= eval_profile(self._offsetsXe, profile, "X")
		self._trueLambdaYe	= eval_profile(self._offsetsYe, profile, "Y")
		self._trueLambdaXv	= eval_profile(self._offsetsXv, profile, "X")
		self._trueLambdaYv	= eval_profile(self._offsetsYv, profile, "Y")

		
		#containers to refill in each "BX"
		
		self._occupanciesXe	= []
		self._occupanciesYe	= []
		self._occupanciesXv	= []
		self._occupanciesYv	= []
		
		self._estLambdasMeanXe	= []
		self._estLambdasMeanYe	= []
		self._estLambdasMeanXv	= []
		self._estLambdasMeanYv	= []
		
		self._estLambdasEmpVarXe	= []
		self._estLambdasEmpVarYe	= []
		self._estLambdasEmpVarXv	= []
		self._estLambdasEmpVarYv	= []
		
		self._graphs		= [[],[],[],[]]
		
		self._fitedParams	= []

		#after this the fitted Parameters are collected in the "commulator"
		

	def purge_minuit_log(self):

		open("Minuit.log", 'w').close()
	
	
	def roll_the_dice(self):
		
		#Erease prev
		
		self._occupanciesXe	= []
		self._occupanciesYe	= []
		self._occupanciesXv	= []
		self._occupanciesYv	= []
					
		#XE
		ne	= (self._param_e).get_nOffOffsetPoints()
		me	= (self._param_e).get_nAtGivenOffset()
		N	= (self._param_e).get_nOrbits()
		
		noise_mu = self._noise[0]
		noise_sig = self._noise[1]
					
		for i1 in range(int(round(ne))):
		
			temp	= []
			
			for i2 in range(int(round(me))):
			
				mu	= (1-np.exp(-self._trueLambdaXe[i1])) + noise_mu/N
				"""
				mu	= (1-np.exp(-self._trueLambdaXe[i1])) + noise_mu/N
				
				sig	= np.sqrt( (np.exp(self._trueLambdaXe[i1])-1)/N + noise_sig / N**2 * np.exp(self._trueLambdaXe[i1])**2)	
				
				occupancy 	= np.random.normal(mu, sig)
				"""
				
				occupancy 	= np.random.binomial(N,mu)/N
				
				temp += [occupancy]
				
			self._occupanciesXe	+= [temp]
		
		#YE
		for i1 in range(int(round(ne))):
		
			temp	= []
			
			for i2 in range(int(round(me))):
			
				mu	= (1-np.exp(-self._trueLambdaYe[i1])) + noise_mu/N
				"""
				mu	= (1-np.exp(-self._trueLambdaXe[i1])) + noise_mu/N
				
				sig	= np.sqrt( (np.exp(self._trueLambdaXe[i1])-1)/N + noise_sig / N**2 * np.exp(self._trueLambdaXe[i1])**2)	
				
				occupancy 	= np.random.normal(mu, sig)
				"""
				
				occupancy 	= np.random.binomial(N,mu)/N
				
				temp += [occupancy]
				
			self._occupanciesYe	+= [temp]
			
		#XV
		nv	= (self._param_vdm).get_nOffOffsetPoints()
		mv	= (self._param_vdm).get_nAtGivenOffset()
		N	= (self._param_vdm).get_nOrbits()
		
		for i1 in range(int(round(nv))):
		
			temp	= []
			
			for i2 in range(int(round(mv))):
			
				mu	= (1-np.exp(-self._trueLambdaXv[i1])) + noise_mu/N
				"""
				mu	= (1-np.exp(-self._trueLambdaXe[i1])) + noise_mu/N
				
				sig	= np.sqrt( (np.exp(self._trueLambdaXe[i1])-1)/N + noise_sig / N**2 * np.exp(self._trueLambdaXe[i1])**2)	
				
				occupancy 	= np.random.normal(mu, sig)
				"""
				
				occupancy 	= np.random.binomial(N,mu)/N
				
				temp += [occupancy]
				
			self._occupanciesXv	+= [temp]
		
		#YV
		for i1 in range(int(round(nv))):
		
			temp	= []
			
			for i2 in range(int(round(mv))):
			
				mu	= (1-np.exp(-self._trueLambdaYv[i1])) + noise_mu/N
				"""
				mu	= (1-np.exp(-self._trueLambdaXe[i1])) + noise_mu/N
				
				sig	= np.sqrt( (np.exp(self._trueLambdaXe[i1])-1)/N + noise_sig / N**2 * np.exp(self._trueLambdaXe[i1])**2)	
				
				occupancy 	= np.random.normal(mu, sig)
				"""
				
				occupancy 	= np.random.binomial(N,mu)/N
				
				temp += [occupancy]
				
			self._occupanciesYv	+= [temp]
			
	
	def build_statistics(self):	
	
		noise_mu = self._noise[0]
		noise_sig = self._noise[1]
					
		
		
		N	= [(self._param_e).get_nOrbits(),(self._param_e).get_nOrbits(),(self._param_vdm).get_nOrbits(),(self._param_vdm).get_nOrbits()]
		
		trueLambdas	= [	self._trueLambdaXe,
					self._trueLambdaYe,
					self._trueLambdaXv,
					self._trueLambdaYv]
					
		
		
		occv	= [	self._occupanciesXe,
				self._occupanciesYe,
				self._occupanciesXv,
				self._occupanciesYv]
		
				
		estLv				= []
		avgerage_of_empirical_Lambdas	= []		
		estLambda_empyrical_variances	= []
		semiTheor_variances		= []
		totTheor_variances		= []
				
		for i1 in range(4):
			
			estLv				+= [[]]
			avgerage_of_empirical_Lambdas	+= [[]]		
			estLambda_empyrical_variances	+= [[]]
			semiTheor_variances		+= [[]]
			totTheor_variances		+= [[]]
			for i2 in range(len(occv[i1])):
				
				
				estLv[i1]	+= [[]]
				
				for i3 in range(len(occv[i1][i2])):
					estLv[i1][i2]	+= [-np.log(1 - occv[i1][i2][i3] )]
				
				#here noise mu is totally neglected
				
				avgerage_of_empirical_Lambdas[i1]	+= [np.mean(estLv[i1][i2])]		
				estLambda_empyrical_variances[i1]	+= [np.var(estLv[i1][i2],  ddof=1)]
				semiTheor_variances[i1]		+= [  (np.exp(avgerage_of_empirical_Lambdas[i1][i2])-1)/N[i1] + noise_sig / N[i1]**2 * np.exp(avgerage_of_empirical_Lambdas[i1][i2])**2	]
				totTheor_variances[i1]		+= [  (np.exp(trueLambdas[i1][i2])-1)/N[i1] + noise_sig / N[i1]**2 * np.exp(trueLambdas[i1][i2])**2	 ]
		
		#fill RGraphErrors structures
		
		offsetss	= [	self._offsetsXe,
					self._offsetsYe,
					self._offsetsXv,
					self._offsetsYv]
					
		self._estLambdasMeanXe	= avgerage_of_empirical_Lambdas[0]
		self._estLambdasMeanYe	= avgerage_of_empirical_Lambdas[1]
		self._estLambdasMeanXv	= avgerage_of_empirical_Lambdas[2]
		self._estLambdasMeanYv	= avgerage_of_empirical_Lambdas[3]
		
		self._estLambdasEmpVarXe	= estLambda_empyrical_variances[0]
		self._estLambdasEmpVarYe	= estLambda_empyrical_variances[1]
		self._estLambdasEmpVarXv	= estLambda_empyrical_variances[2]
		self._estLambdasEmpVarYv	= estLambda_empyrical_variances[3]
		
		for i1 in range(4):
			
			##
			graph_e	= TGraphErrors( len(offsetss[i1]) , 	array( 'f', offsetss[i1]), 
								array( 'f', avgerage_of_empirical_Lambdas[i1]), 
								array( 'f', [0.]*len(offsetss[i1])), 
								array( 'f', np.sqrt(	np.abs(estLambda_empyrical_variances[i1]	)))	)
			
			graph_s	= TGraphErrors( len(offsetss[i1]) , 	array( 'f', offsetss[i1]), 
								array( 'f', avgerage_of_empirical_Lambdas[i1]), 
								array( 'f', [0.]*len(offsetss[i1])), 
								array( 'f', np.sqrt(	np.abs(semiTheor_variances[i1]	)))	)
									
			graph_t	= TGraphErrors( len(offsetss[i1]) , 	array( 'f', offsetss[i1]), 
								array( 'f', avgerage_of_empirical_Lambdas[i1]), 
								array( 'f', [0.]*len(offsetss[i1])), 
								array( 'f', np.sqrt(	np.abs(totTheor_variances[i1]	)))	)
		
			self._graphs[i1] = [graph_e, graph_s, graph_t]
					
		
	
	
	def fit(self):
	
		gROOT.ProcessLine("gSystem->RedirectOutput(\"./Minuit.log\", \"a\");")
	
		fit_results	= [ {}, {}, {}, {}]
		
		method_labels	= ['e','s','t']
	
		for i1 in range(4):
		
		## single fit
			
			#fit func def

			ffs = TF1("ffs","[0]*exp(-(x-[2])**2/(2*([3])**2))")
			ffs.SetParNames("Amp","Mean","Sigma")
			
			for i2 in range(3):
			
				graph	= self._graphs[i1][i2]
			
				#set initial parameters and constraints
				
				StartMean	= graph.GetMean()

				ExpSigma	= graph.GetRMS()
				ExpPeak 	= graph.GetHistogram().GetMaximum()

				LimitMean_lower= StartMean-3*ExpSigma
				LimitMean_upper= StartMean+3*ExpSigma

				StartSigma 	= ExpSigma
				LimitSigma_lower= ExpSigma*1e-1
				LimitSigma_upper= ExpSigma*1e+1

				StartPeak 	= ExpPeak
				LimitPeak_lower = ExpPeak*0.5
				LimitPeak_upper = ExpPeak*2.
				
				
				#set up fitting env

				ffs.SetParameters(StartPeak,StartMean,StartSigma)
				ffs.SetParLimits(0, LimitPeak_lower,LimitPeak_upper)
				ffs.SetParLimits(1, LimitMean_lower,LimitMean_upper)
				ffs.SetParLimits(2, LimitSigma_lower,LimitSigma_upper)
	
	
				#do the fit	
	
				for j in range(5):
				    fit = graph.Fit("ffs","S")
				    if fit.CovMatrixStatus()==3 and fit.Chi2()/fit.Ndf() < 2: break

				#save results

				fitStatus	= fit.Status()
				
				
				fitAmp = ffs.GetParameter("Amp")
				m = ffs.GetParNumber("Amp")
				ampErr = ffs.GetParError(m)
	
				fitMean = ffs.GetParameter("Mean")
				m = ffs.GetParNumber("Mean")
				meanErr = ffs.GetParError(m)

				fitSigma	= ffs.GetParameter("Sigma")
				m = ffs.GetParNumber("Sigma")
				sigmaErr = ffs.GetParError(m)
				
				chi2	= ffs.GetChisquare()
				
				fit_results[i1]['s'+method_labels[i2]]	= [fitAmp, fitMean, fitSigma, ampErr, meanErr, sigmaErr, chi2]
				
			
		## double fit
		
			#fit func def

			ffd = r.TF1("ffd","[0]*([1]*exp(-(x-[2])**2/(2*([3]*[4]/([1]*[4]+1-[1]))**2)) + (1-[1])*exp(-(x-[2])**2/(2*([3]/([1]*[4]+1-[1]))**2)) )")
			ffd.SetParNames("Amp","Frac","Mean","Sigma","SigmaRatio")

			for i2 in range(3):
			
				graph	= self._graphs[i1][i2]
				
				#set initial parameters and constraints

				ExpSigma	= graph.GetRMS()
				ExpPeak 	= graph.GetHistogram().GetMaximum()

				StartMean	= graph.GetMean()
				LimitMean_lower= StartMean-3*ExpSigma
				LimitMean_upper= StartMean+3*ExpSigma

				StartSigma 	= ExpSigma
				LimitSigma_lower= ExpSigma*1e-3
				LimitSigma_upper= ExpSigma*1e+3

				StartRatio 	=  1.5
				LimitRatio_lower=  0.3
				LimitRatio_upper=  3.		# if i let them far, there will be a dirac delta reacching out for the central point. Even worse if it is also a bit off centered!

				StartFrac 	= 0.25
				LimitFrac_lower = 0.001
				LimitFrac_upper = 0.999

				StartPeak 	= ExpPeak
				LimitPeak_lower = ExpPeak*0.5
				LimitPeak_upper = ExpPeak*2.

				#StartC		= percentile 10%
				#LimitC_lower	= -ExpPeak*0.1
				#LimitC_upper	= ExpPeak


				#set up fitting env

				ffd.SetParameters(StartPeak,StartFrac,StartMean,StartSigma,StartRatio)
	
				ffd.SetParLimits(0, LimitPeak_lower,LimitPeak_upper)
				ffd.SetParLimits(1, LimitFrac_lower,LimitFrac_upper)
				ffd.SetParLimits(3, LimitSigma_lower,LimitSigma_upper)
				ffd.SetParLimits(4, LimitRatio_lower,LimitRatio_upper)

				#do the fit	
	
				for j in range(5):
				    fit = graph.Fit("ffd","S")
				    if fit.CovMatrixStatus()==3 and fit.Chi2()/fit.Ndf() < 2: break

				#save results

				fitStatus	= fit.Status()
				
				fitAmp = ffd.GetParameter("Amp")
				m = ffd.GetParNumber("Amp")
				ampErr = ffd.GetParError(m)
	
				fitFrac = ffd.GetParameter("Frac")
				m = ffd.GetParNumber("Frac")
				fracErr = ffd.GetParError(m)
	
				fitMean = ffd.GetParameter("Mean")
				m = ffd.GetParNumber("Mean")
				meanErr = ffd.GetParError(m)

				fitSigma	= ffd.GetParameter("Sigma")
				m = ffd.GetParNumber("Sigma")
				sigmaErr = ffd.GetParError(m)
	
				fitRatio = ffd.GetParameter("SigmaRatio")
				m = ffd.GetParNumber("SigmaRatio")
				ratioErr = ffd.GetParError(m)
				
				chi2	= ffd.GetChisquare()
				
				fit_results[i1]['d'+method_labels[i2]]	= [fitAmp, fitFrac, fitMean, fitSigma, fitRatio, ampErr, fracErr, meanErr, sigmaErr, fitRatio, chi2]
				
		self._fitedParams	= fit_results
		
		gROOT.ProcessLine("gSystem->RedirectOutput(0)")
		
	
	def get_params(self):
		
		sigmas	= {}		
		
		for key in self._fitedParams[0].keys():

			# units, want visible cross section in microbarn
		
			if key[0] == 's':
			
				sigmaX	= self._fitedParams[0][key][2]*1000
				sigmaXe	= self._fitedParams[0][key][5]*1000
				sigmaY	= self._fitedParams[1][key][2]*1000
				sigmaYe	= self._fitedParams[1][key][5]*1000
				
				aX	= self._fitedParams[0][key][0]
				aXe	= self._fitedParams[0][key][3]
				aY	= self._fitedParams[1][key][0]
				aYe	= self._fitedParams[1][key][3]
				
				sigVis	= np.pi * sigmaX * sigmaY * (aX + aY )  / (LHC_revolution_frequency)#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
				
				sigVErr	= sigVis * np.sqrt(   sigmaXe**2 / sigmaX**2 + sigmaYe**2 / sigmaY**2 + (aXe + aYe)**2/(aX+aY)**2  )
				
			else:
			
				sigmaX	= self._fitedParams[0][key][3]*1000
				sigmaXe	= self._fitedParams[0][key][8]*1000
				sigmaY	= self._fitedParams[1][key][3]*1000
				sigmaYe	= self._fitedParams[1][key][8]*1000
				
				aX	= self._fitedParams[0][key][0]
				aXe	= self._fitedParams[0][key][5]
				aY	= self._fitedParams[1][key][0]
				aYe	= self._fitedParams[1][key][5]
				
				sigVis	= np.pi * sigmaX * sigmaY * (aX + aY )  / (LHC_revolution_frequency)#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			
				#sigVErr	= sigVis * (   sigmaXe / sigmaX + sigmaYe / sigmaY + (aXe + aYe)/(aX+aY)  )
				sigVErr	= sigVis * np.sqrt(   sigmaXe**2 / sigmaX**2 + sigmaYe**2 / sigmaY**2 + (aXe + aYe)**2/(aX+aY)**2  )
				
				
			sigmas['E'+key]	= [sigVis, sigVErr, self._fitedParams[0][key], self._fitedParams[1][key] ]
			
			if key[0] == 's':
			
				sigmaX	= self._fitedParams[2][key][2]*1000
				sigmaXe	= self._fitedParams[2][key][5]*1000
				sigmaY	= self._fitedParams[3][key][2]*1000
				sigmaYe	= self._fitedParams[3][key][5]*1000
				
				aX	= self._fitedParams[2][key][0]
				aXe	= self._fitedParams[2][key][3]
				aY	= self._fitedParams[3][key][0]
				aYe	= self._fitedParams[3][key][3]
				
				sigVis	= np.pi * sigmaX * sigmaY * (aX + aY )  / (LHC_revolution_frequency)#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
				
				#sigVErr	= sigVis * (   sigmaXe / sigmaX + sigmaYe / sigmaY + (aXe + aYe)/(aX+aY)  )
				sigVErr	= sigVis * np.sqrt(   sigmaXe**2 / sigmaX**2 + sigmaYe**2 / sigmaY**2 + (aXe + aYe)**2/(aX+aY)**2  )
				
			else:
			
				sigmaX	= self._fitedParams[2][key][3]*1000
				sigmaXe	= self._fitedParams[2][key][8]*1000
				sigmaY	= self._fitedParams[3][key][3]*1000
				sigmaYe	= self._fitedParams[3][key][8]*1000
				
				aX	= self._fitedParams[2][key][0]
				aXe	= self._fitedParams[2][key][5]
				aY	= self._fitedParams[3][key][0]
				aYe	= self._fitedParams[3][key][5]
				
				sigVis	= np.pi * sigmaX * sigmaY * (aX + aY )  / (LHC_revolution_frequency)#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
				
				sigVErr	= sigVis * (   sigmaXe / sigmaX + sigmaYe / sigmaY + (aXe + aYe)/(aX+aY)  )
				
				#sigVErr	= sigVis * (   sigmaXe / sigmaX + sigmaYe / sigmaY + (aXe + aYe)/(aX+aY)  )
				sigVErr	= sigVis * np.sqrt(   sigmaXe**2 / sigmaX**2 + sigmaYe**2 / sigmaY**2 + (aXe + aYe)**2/(aX+aY)**2  )
				
			sigmas['V'+key]	= [sigVis, sigVErr, self._fitedParams[2][key], self._fitedParams[3][key] ]
		
		#print sigmas
		return sigmas
		
		
		
		
		
		
	def plot_this_fit(self):
	
		figure1 = plt.figure()
		gs = plt.GridSpec(3, 3)
		ax1 = figure1.add_subplot(gs[0:2, :])
		ax2 = figure1.add_subplot(gs[2, :])
		
		minX	= self._offsetsXv[0]
		maxX	= self._offsetsXv[-1]
				
		#ax1
		ax1.set_xlim(minX*1.1, maxX*1.1)
		
		X	= np.linspace(minX, maxX, 200)
		Xe	= np.linspace(self._offsetsXe[0], self._offsetsXe[-1], 100)
		
		trueLambdas_vdm	= eval_profile(X, self._profile, "X")
		trueLambdas_e	= eval_profile(Xe, self._profile, "X")
		
		ax1.plot(X, trueLambdas_vdm, "-", color="blue", label="$\widehat \lambda$")
		
		ax1.errorbar(self._offsetsXv, self._estLambdasMeanXv, fmt="o", yerr=np.sqrt(self._estLambdasEmpVarXv), label="VdM exp. averages", color="red")
		
		ax1.errorbar(self._offsetsXe, self._estLambdasMeanXe, yerr=np.sqrt(self._estLambdasEmpVarXe), fmt="o", label="Emit. exp. averages", color="green")
		
		
		
		ax1.plot( 	[ self._offsetsXe[0], self._offsetsXe[0] ], 
				[0, eval_profile(0, self._profile, "X")]
				, "--", color="green" )
				
		ax1.plot( 	[ self._offsetsXe[-1], self._offsetsXe[-1] ], 
				[0, eval_profile(0, self._profile, "X")]
				, "--", color="green" )
				
		ax1.plot( 	[ self._offsetsXv[-1], self._offsetsXv[-1] ], 
				[0, eval_profile(0, self._profile, "X")]
				, "--", color="red" )
				
		ax1.plot( 	[ self._offsetsXv[0], self._offsetsXv[0] ], 
				[0, eval_profile(0, self._profile, "X")]
				, "--", color="red" )
		
		ax1.legend()

		ax1.set_title("The beam scan profile and the MC scan points")
		
		#ax2


		ax2.set_title("Fit weights. Zero values indicate zero variance and hence enormous weights.")
		
		ax2.set_xlim(minX*1.1, maxX*1.1)
		ax2.set_ylim(-.05, 1.)
		
		tWe	= calcTheorWeights(trueLambdas_e, self._param_e, self._profile)
		tWvdm	= calcTheorWeights(trueLambdas_vdm, self._param_vdm, self._profile)
		
		
		ax2.plot(X, tWvdm, "-", label="VdM theor weights", color="blue")
		ax2.plot(Xe, tWe, "--", label="Emittance theor weights", color="blue")
		
		ax2.plot([minX*1.1,maxX*1.1], [0, 0], "-", label="zero", color="cyan")
		
		#Emp
		temp = np.sqrt(self._estLambdasEmpVarXv)
		temp = 1. / temp

		for i in range(len(temp)):
			if np.isinf(temp[i]):
				temp[i]=-0.1

		temp /= min(i for i in temp if i>0) 
		temp *= min(i for i in tWvdm if i>0)
		ax2.plot(self._offsetsXv, temp, "o-", label="Emp weight VdM", color="red")
		
		temp = 1. / np.sqrt(self._estLambdasEmpVarXe)

		for i in range(len(temp)):
			if np.isinf(temp[i]):
				temp[i]=-0.1

		temp /= min(i for i in temp if i>0) 
		temp	*= min(i for i in tWe if i>0)
		ax2.plot(self._offsetsXe, temp, "o--", label="Emp weight E", color="red")
		
		#ST
		print self._estLambdasMeanXv
		print self._trueLambdaXv
		stWe	= calcTheorWeights(self._estLambdasMeanXe, self._param_e, self._profile)
		stWvdm	= calcTheorWeights(self._estLambdasMeanXv, self._param_vdm, self._profile)
		
		stWe /= min(i for i in stWe if i>0) 
		stWe	*= min(i for i in tWe if i>0)
		
		stWvdm /= min(i for i in stWvdm if i>0) 
		stWvdm	*= min(i for i in tWvdm if i>0)

		
		
		ax2.plot(self._offsetsXv, stWvdm, "o-", label="VdM semi-theor weights", color="green")
		ax2.plot(self._offsetsXe, stWe, "o--", label="Emit semi-theor weights", color="green")
		
		ax2.legend()
		
		##########
		figure2 = plt.figure()
		gs = plt.GridSpec(4, 1)
		ax3e = figure2.add_subplot(gs[0, 0])
		ax4e = figure2.add_subplot(gs[1, 0])
		
		ax3v = figure2.add_subplot(gs[2, 0])
		ax4v = figure2.add_subplot(gs[3, 0])
		
		
		fit_results	= self._fitedParams #[4][dict] index: Xe,Ye,Xv,Yv; dict: [s,d] + ['e','s','t']
		
		#e
		ax3e.set_title('Emittance scan data')
		ax3e.errorbar(self._offsetsXe, self._estLambdasMeanXe, yerr=np.sqrt(self._estLambdasEmpVarXe), fmt="o", label="Emit. exp. averages", color="green")		
		
		for i in ['e','s','t']:
			p	= fit_results[0]['s'+i]
			ax3e.plot(X, SG(X,p[0],p[1],p[2],0), '-', label='s'+i)
		
		ax3e.legend()
		
		ax4e.errorbar(self._offsetsXe, self._estLambdasMeanXe, yerr=np.sqrt(self._estLambdasEmpVarXe), fmt="o", label="Emit. exp. averages", color="green")
		for i in ['e','s','t']:
			p	= fit_results[0]['d'+i]
			ax4e.plot(X, DG_strange(X,p[0],p[1],p[2],p[3],p[4],0), '-', label='d'+i)
		
		ax4e.legend()
		#v
		
		ax3v.set_title('VdM scan data')
		ax3v.errorbar(self._offsetsXv, self._estLambdasMeanXv, fmt="o", yerr=np.sqrt(self._estLambdasEmpVarXv), label="VdM exp. averages", color="red")
		for i in ['e','s','t']:
			p	= fit_results[2]['s'+i]
			ax3v.plot(X, SG(X,p[0],p[1],p[2],0), '-', label='s'+i)
		
		ax3v.legend()
		ax4v.errorbar(self._offsetsXv, self._estLambdasMeanXv, fmt="o", yerr=np.sqrt(self._estLambdasEmpVarXv), label="VdM exp. averages", color="red")
		for i in ['e','s','t']:
			p	= fit_results[2]['d'+i]
			ax4v.plot(X, DG_strange(X,p[0],p[1],p[2],p[3],p[4],0), '-', label='d'+i)
		ax4v.legend()
		
		##########
		figure3 = plt.figure()
		gs = plt.GridSpec(3, 2)
		ax5 = figure3.add_subplot(gs[0, 0])
		ax6 = figure3.add_subplot(gs[0, 1])
		ax7 = figure3.add_subplot(gs[1, 0])
		ax8 = figure3.add_subplot(gs[1, 1])
		ax9 = figure3.add_subplot(gs[2, 0])
		ax10= figure3.add_subplot(gs[2, 1])
		
		axs	= [ax5,ax6,ax7,ax8,ax9,ax10]
		titles	= ['A','Frac','Mean','Sigma','SigmaRatio','c']
		
		true_params	= self._profile.get_pv("X")
		
		label1	= ['s','d']
		label2	= ['e','s','t']
		
		for i in range(5):
			axs[i].set_title(titles[i])
			axs[i].set_xlim(-6.5,8.5)
			
			cntrP	= 1
			cntrN	= -1
			
			for i2 in range(2):
				for i3 in range(3):
				
				
					if label1[i2]=='s' and i in [0,2,3]:
				
						ti	= [0,99,1,2][i]
				
						p1	= fit_results[0][label1[i2]+label2[i3]][ti]#Xe
						p2	= fit_results[2][label1[i2]+label2[i3]][ti]#Xv

					
						axs[i].plot([cntrP], [p1], "o",label= 'e'+label1[i2]+label2[i3])
						axs[i].plot([cntrN], [p2], "o",label= 'v'+label1[i2]+label2[i3])
					
						cntrP+=1
						cntrN-=1
					
					elif label1[i2]!='s':
					
						
					
						p1	= fit_results[0][label1[i2]+label2[i3]][i]#Xe
						p2	= fit_results[2][label1[i2]+label2[i3]][i]#Xv
						
						
						if self._profile.get_sgdg() == "DG":
							true_paramsi	= true_params[i]
						else:
							true_paramsi	= 10
						
						
						if titles[i] == 'Frac' and true_paramsi < 0.5:
							if p1>.5:
								p1	= 1-p1
							if p2>.5:
								p2	= 1-p2
						elif titles[i] == 'Frac' and true_paramsi >= 0.5:
							if p1<.5:
								p1	= 1-p1
							if p2<.5:
								p2	= 1-p2
							
						if titles[i] == 'SigmaRatio' and true_paramsi > 1.:	
							if p1<1.:
								p1	= 1./p1
							if p2<1.:
								p2	= 1./p2
						elif titles[i] == 'SigmaRatio' and true_paramsi <= 1.:	
							if p1>1.:
								p1	= 1./p1
							if p2>1.:
								p2	= 1./p2
					
						axs[i].plot([cntrP], [p1], "o",label= 'e'+label1[i2]+label2[i3])
						axs[i].plot([cntrN], [p2], "o",label= 'v'+label1[i2]+label2[i3])
						
					
						cntrP+=1
						cntrN-=1
						
			
			if self._profile.get_sgdg() == "DG":
				axs[i].plot([0],[true_params[i]],'x')
			elif i in [0,2,3]:
				ti	= [0,99,1,2][i]
				axs[i].plot([0],[true_params[ti]],'x')
		
			axs[i].legend()
			
			
			
			
		
	
		return [figure1, figure2, figure3]
	
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
