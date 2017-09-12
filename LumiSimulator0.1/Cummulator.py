import matplotlib.pyplot as plt
import numpy as np


# [in Hz]
LHC_revolution_frequency = 11245.


class Cummulator:

	def __init__(self, profile):
	
		self._sigmaX		= profile.get_csigmaX()*1000
		self._sigmaY		= profile.get_csigmaY()*1000
		
		self._amplitudeX	= profile.get_amplitudeX()
		self._amplitudeY	= profile.get_amplitudeY()

		#the dictonaries are accordint to the fit method
		self._sigmaVis		= {}
		self._sigmaVisErr	= {}
		
		

		self._singleGParams_Ax	= {}
		self._singleGParams_Mx	= {}
		self._singleGParams_Sx	= {}
		
		self._singleGParams_Ay	= {}
		self._singleGParams_My	= {}
		self._singleGParams_Sy	= {}
		
		

		self._doubleGParams_Ax	= {}
		self._doubleGParams_Fx	= {}
		self._doubleGParams_Mx	= {}
		self._doubleGParams_Sx	= {}
		self._doubleGParams_Rx	= {}
		
		self._doubleGParams_Ay	= {}
		self._doubleGParams_Fy	= {}
		self._doubleGParams_My	= {}
		self._doubleGParams_Sy	= {}
		self._doubleGParams_Ry	= {}

	def store(self, s):
	
		for key,val in s.items():
		
			if  not( key in  self._sigmaVis ):
			
				self._sigmaVis[key]	=  [val[0]]
				self._sigmaVisErr[key]	=  [val[1]]
				
			else:
				self._sigmaVis[key]	+= [val[0]]
				self._sigmaVisErr[key]	+= [val[1]]
				
			
		
			if key[1] == 's':
			
				if  not( key in  self._singleGParams_Ax ):
				
					self._singleGParams_Ax[key]	= [val[2][0]]
					self._singleGParams_Mx[key]	= [val[2][1]]
					self._singleGParams_Sx[key]	= [val[2][2]]
				
					self._singleGParams_Ay[key]	= [val[3][0]]
					self._singleGParams_My[key]	= [val[3][1]]
					self._singleGParams_Sy[key]	= [val[3][2]]
					
				else:
					self._singleGParams_Ax[key]	+= [val[2][0]]
					self._singleGParams_Mx[key]	+= [val[2][1]]
					self._singleGParams_Sx[key]	+= [val[2][2]]
				
					self._singleGParams_Ay[key]	+= [val[3][0]]
					self._singleGParams_My[key]	+= [val[3][1]]
					self._singleGParams_Sy[key]	+= [val[3][2]]
					
			else:
				if  not( key in  self._doubleGParams_Ax ):
				
					self._doubleGParams_Ax[key]	= [val[2][0]]
					self._doubleGParams_Fx[key]	= [val[2][1]]
					self._doubleGParams_Mx[key]	= [val[2][2]]
					self._doubleGParams_Sx[key]	= [val[2][3]]
					self._doubleGParams_Rx[key]	= [val[2][4]]
					
					self._doubleGParams_Ay[key]	= [val[3][0]]
					self._doubleGParams_Fy[key]	= [val[3][1]]
					self._doubleGParams_My[key]	= [val[3][2]]
					self._doubleGParams_Sy[key]	= [val[3][3]]
					self._doubleGParams_Ry[key]	= [val[3][4]]
					
				else:
					self._doubleGParams_Ax[key]	+= [val[2][0]]
					self._doubleGParams_Fx[key]	+= [val[2][1]]
					self._doubleGParams_Mx[key]	+= [val[2][2]]
					self._doubleGParams_Sx[key]	+= [val[2][3]]
					self._doubleGParams_Rx[key]	+= [val[2][4]]
					
					self._doubleGParams_Ay[key]	+= [val[3][0]]
					self._doubleGParams_Fy[key]	+= [val[3][1]]
					self._doubleGParams_My[key]	+= [val[3][2]]
					self._doubleGParams_Sy[key]	+= [val[3][3]]
					self._doubleGParams_Ry[key]	+= [val[3][4]]
					
				
		
		
		

	def plt_sigmaVis(self):

		plots	= []
		

		plots	+= [plt.figure()]
		
		ax1	= plots[0].add_subplot(1,1,1)
		ax1.set_title('$\Sigma_1\Sigma_2\overline{Amp}$')
		
		limlow	= -len(self._sigmaVis.items()) / 2.
		
		ax1.plot([limlow-0.5, -limlow+.5], [ np.pi * self._sigmaX * self._sigmaY * (self._amplitudeX + self._amplitudeY) / LHC_revolution_frequency ]*2, "-", color="grey")
		
		
		for key, value in sorted(self._sigmaVis.items()):
		
			ax1.errorbar(np.linspace(limlow-0.45, limlow+0.45, len(value)), value, 
					yerr=self._sigmaVisErr[key], 
					fmt='o', label=key)
					
			ax1.legend()
			
			limlow +=1
			
			if limlow == 0: limlow +=1
		
		return plots
		
		
	def plt_Amp(self):

		
		figure3 = plt.figure()
		gs = plt.GridSpec(2, 3)
		ax1 = figure3.add_subplot(gs[0, :])
		ax2 = figure3.add_subplot(gs[1, :])
		
		##########X prams
		ax1.set_title("Fitted X amplitudes")
		limlow	= -len(self._sigmaVis.items()) /2.
		
		ax1.set_xlim(limlow-0.3, -limlow+1.5)
		
		for key, value in sorted(self._singleGParams_Ax.items()):
		
			ax1.plot(np.linspace(limlow-0.45, limlow+0.45, len(value)), value, 
						'o', label=key)
			limlow +=1
		limlow	+= 1

		
		for key, value in sorted(self._doubleGParams_Ax.items()):
		
			ax1.plot(np.linspace(limlow-0.45, limlow+0.45, len(value)), value, 
						'o', label=key)
			limlow +=1
		ax1.plot([limlow, -limlow], [self._amplitudeX,self._amplitudeX], color="grey")
		ax1.legend()
		
		##########Y prams
		ax2.set_title("Fitted Y amplitudes")
		limlow	= -len(self._sigmaVis.items()) /2.
		
		ax2.set_xlim(limlow-0.3, -limlow+1.5)
		
		for key, value in sorted(self._singleGParams_Ay.items()):
		
			ax2.plot(np.linspace(limlow-0.45, limlow+0.45, len(value)), value, 
						'o', label=key)
			limlow +=1
		limlow	+= 1
		
		for key, value in sorted(self._doubleGParams_Ay.items()):
		
			ax2.plot(np.linspace(limlow-0.45, limlow+0.45, len(value)), value, 
						'o', label=key)
			limlow +=1

		ax2.plot([limlow, -limlow], [self._amplitudeY,self._amplitudeY], color="grey")
		
		ax2.legend()
	
	
		return figure3
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
