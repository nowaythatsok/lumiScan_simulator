import numpy as np
import matplotlib.pyplot as plt


class container1:

	def __init__(self,N,No,BigN):
	
		self._nOrbits	= No
	
		self._minLambda	= 10**-3
		self._maxLambda	= 0.15
		
		self._BigN	= BigN

		self._LambdaRange	= np.linspace(	self._minLambda,
							self._maxLambda,
							N)
							
		self._pRange		= 1 - np.exp(-self._LambdaRange)
							
		self._ExpAvgLambda	= np.linspace(	self._minLambda,
							self._maxLambda,
							N)

	
	def roll_the_dice(self):
		
		for i in range(len(self._pRange)):
			
			n	= []
			
			for j in range(self._BigN):
				n += [np.random.binomial(self._nOrbits, self._pRange[i])]
			
			n = np.array(n,dtype=float)/self._nOrbits
			
			self._ExpAvgLambda[i]	= np.average(- np.log(1-n))
			
		
	def plot_percentage(self):
	
		figure1 = plt.figure()
		
		ax1	= figure1.add_subplot(1,1,1)	
		
		ax1.plot([self._minLambda, self._maxLambda], [1,1],"-",color="blue")
		
		ax1.plot(self._LambdaRange, self._ExpAvgLambda / self._LambdaRange ,"-",color="green")
		
		l	= len(self._ExpAvgLambda)
		
		avg	= np.average( (self._ExpAvgLambda / self._LambdaRange)[int(l/10):] )
		
		ax1.set_title("simulation result with {0} LHC orbits. The bias is: {1}".format( self._nOrbits, avg ))
		
		ax1.plot([self._minLambda, self._maxLambda], [avg,avg],"-",color="red")
		
		
