import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as opt

#Step 1

'''Loading the data given for the Cepheids'''
p, pe, P, m, A , Ae = np.loadtxt('MW_Cepheids.dat',\
                            unpack=True, \
                            usecols=(1,2,3,4,5,6), \
                            dtype=float, \
                            comments="#")  

#Absolute Magnitude:                     
dp = 1000/p
logdp = np.log10(dp)
logP = np.log10(P)     
M = m - 5.0*logdp - A + 5.0  
wmean = np.ma.average(logP)    
              
#The uncertainty of absolute magnitude:
sig_M = np.sqrt((Ae)**2 + ((5*pe)/(p*np.log(10)))**2)

def func(logP, alpha, beta):
    """calculates the model"""
    line = alpha*logP + beta
    return line

dof = len(logP)-1.0 
""" this calculates the degrees of freedom"""
start_alpha = -5.0 #this is our starting slope for the algorithm
start_beta = -5.0 #this is our starting intercept for the algorithm
fit, covar = opt.curve_fit(f=func, xdata=logP-wmean, ydata=M,
                    sigma=sig_M, p0=(start_alpha, start_beta),
                    absolute_sigma=True)

print('The covariance matrix is:',covar[(0,1),(1,0)])


'''the best slope and intercept can now be found:'''                   
best_alpha = fit[0]
best_beta = fit[1]
'''As well as the best fitted values of absolute magnitude and the chi square '''
M_best = func(logP-wmean, best_alpha, best_beta)
best_chi2 = np.sum(((M-M_best)**2.0)/(sig_M**2.0)) #calculates the best value ofχ^2
best_redc2 = best_chi2/dof                         #calculates the reduced χ^2
"""The covariance matrix includes for the errors of intercept and slope"""
alpha_err = np.sqrt(covar[0,0])
beta_err = np.sqrt(covar[1,1])
print('The error on the slope is:',alpha_err)
print('The error on the intercept is:',beta_err)

def chisq(M, sig_M, M_m):
    """takes model, data and error vectors and calculates the chi square"""
    chi2 = np.sum(((M-M_m)**2.0)/(sig_M**2.0))
    return chi2

plt.figure(1)                      
plt.xlabel('log (Period) [log(days)]')
plt.ylabel('Absolute Magnitude [mag]')
plt.title('The Cepheid Period-Luminosity Relation')
plt.plot(logP, M_best, color='black', marker="None", linewidth=2, linestyle="-")
plt.errorbar(logP, M, yerr=sig_M, linestyle="None", marker='o', markersize=7, color='red')
chi2 = chisq(M, sig_M, M_best)
cstring1 = 'Chi-squared =' + str(np.round(chi2))
print()
print('CURVE_FIT RESULTS:')
print('-----------------')
print('Best-fitting alpha = ', best_alpha)
print('Best-fitting beta = ', best_beta)
print('Corresponding Chi^2 = ', best_chi2)
print('Corresponding Reduced Chi^2 =', best_redc2)
print()
print()
plt.show()


#Step 2
''' The best fitting values found above are now used as α and β for the calculation of absolute magnitude'''
α = best_alpha
β = best_beta  

#The data of period and apparent magnitude for all 8 galaxies:
logP1, m1 = np.loadtxt('hst_gal1_cepheids.dat', unpack=True, usecols=(1,2), dtype=float, comments="#") 
                    
logP2, m2 = np.loadtxt('hst_gal2_cepheids.dat', unpack=True, usecols=(1,2), dtype=float, comments="#")   

logP3, m3 = np.loadtxt('hst_gal3_cepheids.dat', unpack=True, usecols=(1,2), dtype=float, comments="#")
                 
logP4, m4 = np.loadtxt('hst_gal4_cepheids.dat', unpack=True, usecols=(1,2), dtype=float, comments="#") 
                 
logP5, m5 = np.loadtxt('hst_gal5_cepheids.dat', unpack=True, usecols=(1,2), dtype=float, comments="#") 
                 
logP6, m6 = np.loadtxt('hst_gal6_cepheids.dat', unpack=True, usecols=(1,2), dtype=float, comments="#") 
                
logP7, m7 = np.loadtxt('hst_gal7_cepheids.dat', unpack=True, usecols=(1,2), dtype=float, comments="#") 
               
logP8, m8 = np.loadtxt('hst_gal8_cepheids.dat', unpack=True, usecols=(1,2), dtype=float, comments="#") 
                       
#The recession velocities and names data                    
vrec = np.loadtxt('galaxy_data.dat', unpack=True, usecols=(1), dtype=float, comments="#")     
names =  np.loadtxt('galaxy_data.dat', unpack=True, usecols=(0), dtype=str, comments="#")  
                    
'''Putting the data for all stars in arrays'''                        
logPer = [logP1-wmean, logP2-wmean, logP3-wmean, logP4-wmean, logP5-wmean, logP6-wmean, logP7-wmean, logP8-wmean] 
m = [m1, m2, m3, m4, m5, m6, m7, m8] 

'''These are arrays that the distance and distance modulus for every galaxy will be saved'''
distances = np.zeros(8) 
distance_moduli = np.zeros(8) 

#Function for the calculation of both d and μ
def distance(logPer, m, A_mw):
    M_ceph = α*(logPer[i]) + β 
    μ = m[i] - M_ceph
    distance = 10**((μ + 5 - A_mw[i])/5)*(10**-6)
    return distance

def distmodulus(logPer, m):
    M = α*(logPer[i]) + β
    μ = m[i] - M
    return μ

#The data of extinction for each galaxy 
A_mw = np.loadtxt('galaxy_data.dat', unpack=True, usecols=(2))


for i in range (0, 8):
    d = distance(logPer, m, A_mw)
    d_gal = np.mean(d)
    dmod = distmodulus(logPer, m)
    μ = np.mean(dmod)
    distance_moduli[i] = μ
    np.append(distance_moduli, μ) #saves the distance moduli in an array
    distances[i] = d_gal
    np.append(distances, d_gal)  #saves the calculated distances in another array
    i = i+1

'''Calculation of the distance error'''  


sdist = np.zeros(8) 

def derror(distances, logPer):
    derror = (distances[i])*((np.log(10))/5)*(np.sqrt(((logPer[i])*alpha_err)**2) + beta_err**2)
    return derror

for i in range (0, 8):
    sig_d = derror(distances, logPer)
    sigma = np.mean(sig_d)
    print('The distance for galaxy', names[i], ' is:', "%.10f" % distances[i],'±', "%.10f" % sigma, 'Mpc' )
    print('and the distance modulus μ for that galaxy is:', format(distance_moduli[i]*(10**6),'E'), '±', format(sigma, 'E'), 'pc')
    print()
    sdist[i] = sigma
    np.append(sdist, sigma) #saves the distance error 
    i=i+1
    
 #calculates the error for the Hubble constant

'''Here the intrinsic dispresion is used for this fit'''
σ_in = 1.8 #this os the best intrinsic uncertainty trial
line_sigmas = np.sqrt(sdist**2 + σ_in**2) #the final errors of the line fit

#Step 3
     
def func2(vrec, alpha):
    """calculates the model"""
    line = alpha*vrec 
    return line

'''Same method as step 1 with different variables and no intercept'''
start_α = 1.0
dof2 = len(distances)-1.0     
fit2, covar2 = opt.curve_fit(f=func2, xdata=vrec, ydata=distances,
                    sigma=line_sigmas, p0=start_α,
                    absolute_sigma=True)    

best_α = fit2[0]
d_best = func2(vrec, best_α)
best_χ2 = np.sum(((distances-d_best)**2.0)/(line_sigmas**2.0))
best_redχ2 = best_χ2/dof2

#Step 4

sigma_α = np.sqrt(covar2[0,0])
sigma_H = np.sqrt(((1/(best_α**2))**2) * (sigma_α**2)) #calculates the error for the Hubble constant
sigma_τ = np.sqrt(covar2[0,0])*(3.086*3.16887646)*(10**2) #calculates the error for the age of the Universe in units of billion years


plt.figure(2)
plt.xlabel('Recession Velocity [km/s]')
plt.ylabel('Dgal [Mpc]')
plt.title("Hubble's constant")    
plt.plot(vrec, d_best, color='black', marker="None", linewidth=2, linestyle="-")
plt.errorbar(vrec, distances, yerr=line_sigmas, linestyle="None", marker='o', markersize=7, color='red')
plt.show()
print()
#The inverse of the best slope is Hubble constant:
print('Hubble constant is:', 1/best_α, '±', sigma_H, 'km/s/Mpc' )
print()
print('CURVE_FIT 2 RESULTS:')
print('-----------------')
print('Best-fitting alpha = ', best_α)
print('Corresponding Chi^2 = ', best_χ2)
print('Corresponding Reduced Chi^2 =', best_redχ2)
print()
print('The age of the Universe is:', best_α*(3.086*3.16887646)*(10**2),'±', sigma_τ, 'billion years')




