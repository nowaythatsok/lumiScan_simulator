#!/usr/bin/python
import csv


def load_params(file_name):

	params  = {}
	temp	= []
	
	with open(file_name, 'rb') as csvfile:
		
		read 	= csv.reader(csvfile)
		
		for row in read:
			if row != []:
				temp	+= [row]	
	
			
	for i in range(len(temp[0])):
		params[temp[0][i]]	= []
		
		if i!=1:
		
			for j in range(1, (len(temp)-1)/2):
				params[temp[0][i]] += [ float(temp[j][i]) ]
			for j in range((len(temp)-1)/2+1, len(temp)-1):
				params[temp[0][i]] += [ float(temp[j][i]) ]
				
		else:
			for j in range(1, (len(temp)-1)/2):
				params[temp[0][i]] += [ temp[j][i] ]
			for j in range((len(temp)-1)/2+1, len(temp)-1):
				params[temp[0][i]] += [ temp[j][i] ]
			
	params['len']		= len(params['BCID'])
			
		
	return params




class Profile:
	
	def __init__(self):
	
		self.sgdg	= ""
		
		self.paramsX	= []
		self.paramsY	= []
		
		self.noise_mu	= 0
		self.noise_sigma= 0
		
	def set_noise(self, mupc, sigpc):
		
		self.noise_mu	= mupc * (paramsX[0] + paramsY[0])/2.
		self.noise_sigma= sigpc * (paramsX[0] + paramsY[0])/2.
		
	def get_noise(self):
		return [self.noise_mu, self.noise_sigma]

	def get_amplitudeX(self):
		
		return self.paramsX[0]

	def get_amplitudeY(self):
		
		return self.paramsY[0]
		

	def get_meanX(self):
		if self.sgdg == "SG":
			return self.paramsX[1]
		else: 
			return self.paramsX[2]
			
	def get_meanY(self):
		if self.sgdg == "SG":
			return self.paramsY[1]
		else: 
			return self.paramsY[2]
	
	def get_csigmaX(self):
		if self.sgdg == "SG":
			return self.paramsX[2]
		else: 
			return self.paramsX[3]
			
	def get_csigmaY(self):
		if self.sgdg == "SG":
			return self.paramsY[2]
		else: 
			return self.paramsY[3]



	def get_sigmaX(self):
		if self.sgdg == "SG":
			return self.paramsX[2]
		else: 
			return max(1,self.paramsX[4])*self.paramsX[3]/(self.paramsX[1]*self.paramsX[4]+1-self.paramsX[1])
			
	def get_sigmaY(self):
		if self.sgdg == "SG":
			return self.paramsY[2]
		else: 
			return max(1,self.paramsY[4])*self.paramsY[3]/(self.paramsY[1]*self.paramsY[4]+1-self.paramsY[1])
			
	def get_sgdg(self):
		return self.sgdg
		
	def get_pv(self, label):
		if label == "X":
			return self.paramsX
		else:
			return self.paramsY

		
	def load_SG(self,BCID):
	
		self.sgdg	= "SG"
	
		param_dict = load_params("./xSG_FitResults6020.csv")
		#dict structure: {'key (column name)', [col elements]}
		
		first_half	= param_dict['len']/2
		
		#choose the correct row
		index		= param_dict['BCID'][:first_half].index(BCID)
		
		self.paramsX	= [	param_dict['peak'][index],
					param_dict['Mean'][index],
					param_dict['sigma'][index],
					0
				]
		
		index		= param_dict['BCID'][first_half:].index(BCID)
		
		self.paramsY	= [	param_dict['peak'][index],
					param_dict['Mean'][index],
					param_dict['sigma'][index],
					0
				]
	
	
	def load_DG(self,BCID):
	
		self.sgdg	= "DG"
		
		param_dict = load_params("./xDG_FitResults6016.csv")
		#dict structure: {'key (column name)', [col elements]}
		
		first_half	= param_dict['len']/2
		
		#choose the correct row
		index		= param_dict['BCID'][:first_half].index(BCID)
		
		self.paramsX	= [	param_dict['peak'][index],
					param_dict['Frac'][index],
					param_dict['Mean'][index],
					param_dict['sigma'][index],
					param_dict['sigmaRatio'][index],
					0
				]
		
		index		= param_dict['BCID'][first_half:].index(BCID)
		
		self.paramsY	= [	param_dict['peak'][index],
					param_dict['Frac'][index],
					param_dict['Mean'][index],
					param_dict['sigma'][index],
					param_dict['sigmaRatio'][index],
					0
				]
		
		
		
