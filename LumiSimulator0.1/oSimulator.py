import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from oFuncs import *
from Parameters import *
from Profile import *
from Measurement import *
from Cummulator import *



#Setting up statistics parameters
#See 	Prameters.py	for more
emittance_parameters 	= Fill_parameters("emittance")
vanderMeer_parameters	= Fill_parameters("vdM")


#Setting up bream profile based on experimental fit results
#See	Profile.py	for more
profile	= Profile()

profile.load_DG(41)	#fill a VdM profile with the data of the 41st BX
#profile.load_SG(60)	#fill an emittance profile with the data of the 60st BX


#Measurement wraps the parameters and the occupancy data generated in each BX
measurement 	= Measurement(profile, emittance_parameters, vanderMeer_parameters, False)
measurement.purge_minuit_log()

#Commulator wraps the results of the fits. 
cummulator	= Cummulator(profile)

for i in range(10):	#(how many BXs?)
	
	measurement.roll_the_dice()
	
	measurement.build_statistics()
	
	measurement.fit()
	
	s	= measurement.get_params()

	cummulator.store(s)


measurement.plot_this_fit()

cummulator.plt_sigmaVis()
cummulator.plt_Amp()



plt.show()
