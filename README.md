# lumiScan_simulator
A Monte Carlo simulation for the CMS BRIL group concerning the error handling of the occupancy values in the course of the fitting procedure. Originally the error of each measurement point was establised by cutting them up into several temporal subunits and taking the variance of these points.
Currently 3 error handling methods are available and used in the simulation
* Orgignal (empirical variance)
* Semi-theoretical variance. (The theoretical variance is determined by the theoretical exected occupancy. Changing the latter for the empyrical average gives the semi-theoretical value.)
* Totally theoretical variance. (The simulaton knows the theoretical expected occupancy - which of course in practice is not accessible for the researchers, this method is for reference, as it probably never gets muc better than this - and hence can calculate the theoretical variance.)

The MC also uses two different fit models

* Single Gaussian
* Double Gaussian

and does so for both on a [-3\sigma, 3\sigma] and a restricted [-1.5\sigma, 1.5\sigma] range (see & set in *Parameters.py*)

#### run simulation using 
python oSimulator.py

#### changing statistical parameters (\#LHC orbits, data points, segmentation, range )
see the *Fill_parameters* function of *Parameters.py*

#### setting the "profile"
The profile contains all assumptions on the exact shape of the 2 beam convolution. 
Its data is loaded from previous fit data from either 
* fill 6016 (contains fits for the VdM scans, hence gives a double Gaussian profile)
* fill 6020 (emittance profile)

see *oSimulator.py* for chosing between these two.

#### further actions
After the initial setup is done all actions are performed on the "measurement" object.
This contains all parameters and has methods to perform
* a simulated scan (Full scale and restricted simultaneously, these are genrtated according to the vdM and emittance parameters respectively)
* a post processing of the occupancies to get fittable data
* a fit
* a plot of the state of the object 
* an export of the fit parameters

The results are then stored in the Cummulator object.

# Zero Counting Bias simulator
This tiny script shows how zero counting overestimates the parameter of the Poisson distribution especially for small sample sises. 
