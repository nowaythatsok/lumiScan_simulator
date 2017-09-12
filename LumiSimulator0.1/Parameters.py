#!/usr/bin/python



class Parameters:

	def __init__(self, typ):
		self._typ		= typ # emittance / VdM
		self._nOrbits		= 0
		self._range		= 0
		self._nAtGivenOffset	= 0
		self._nOffOffsetPoints	= 0

		
		
	#setters
	def set_nOrbits(self,n):
		self._nOrbits	= float(n)
		
	def set_range(self,r):
		self._range	= float(r)
		
	def set_nAtGivenOffset(self,n):
		self._nAtGivenOffset	= float(n)
		
	def set_nOffOffsetPoints(self,n):
		self._nOffOffsetPoints	= float(n)
		
		
	
	#getters
	def get_nOrbits(self):
		return self._nOrbits
		
	def get_range(self):
		return self._range
		
	def get_nAtGivenOffset(self):
		return self._nAtGivenOffset
		
	def get_nOffOffsetPoints(self):
		return self._nOffOffsetPoints
		
	
	
		
	
def Fill_parameters(typ):
	
	if typ == "emittance":
		ret	= Parameters("vdM")
		
		ret.set_nOrbits(2**14)
		ret.set_range(1.5)
		ret.set_nAtGivenOffset(7)
		ret.set_nOffOffsetPoints(7)
		
		return ret		
		
	elif typ == "vdM" or typ == "VdM":
		ret	= Parameters("vdM")
		
		ret.set_nOrbits(2**14)
		ret.set_range(3)
		ret.set_nAtGivenOffset(21)
		ret.set_nOffOffsetPoints(17)
		
		return ret
		
	else:
		return Parameters("err")
	
	
