"""
Version 2.0
Author: Alpish Srivastava
Date: February 2024

"""

import numpy as np
import matplotlib.pyplot as plt
from uncertainties import unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from scipy.stats import median_abs_deviation as mad
import plotly.graph_objects as go
import pandas as pd

plt.style.use("default")
plt.rc('xtick', direction='in',top=True)
plt.rc('ytick', direction='in',right=True)
plt.rc('xtick.major', size=6, width=1.5)  # Change the size and thickness of the x-axis ticks
plt.rc('ytick.major', size=6, width=1.5)  # Change the size and thickness of the y-axis ticks
plt.rc('xtick.minor', size=3, width=1)  # Change the size and thickness of the x-axis minor ticks
plt.rc('ytick.minor', size=3, width=1)  # Change the size and thickness of the y-axis minor ticks
plt.rc('axes', linewidth=1.5)

size=13
plt.rc("mathtext", fontset="dejavuserif")
plt.rc("font", family="DejaVu Serif", size=size)


@staticmethod
def betaprof(r,S0,rc,beta,cxb=None):
    """
    Compute the beta model surface brightness profile.
    
    Parameters:
    r   : array-like, radii at which to compute the profile
    S0  : float, normalization (central surface brightness)
    rc  : float, core radius
    beta: float, slope parameter
    cxb : float, background level (optional)
    
    Returns:
    S_X : array-like, surface brightness profile
    """
    if cxb==None:
        return S0*(1 + (r / rc) ** 2) ** (-3 * beta + 0.5)
    else:
        return S0*(1 + (r / rc) ** 2) ** (-3 * beta + 0.5) + cxb


@staticmethod
def doublebeta(r,S0,rc,beta,S0_2,rc2,beta2,cxb=None):
    """
    Compute the double beta model surface brightness profile.

    Parameters:
    r   : array-like, radii at which to compute the profile
    S0  : float, normalization (central surface brightness)
    rc  : float, core radius
    beta: float, slope parameter
    S0_2 : float, normalization 2
    rc2 : float, core radius 2
    beta2: float, slope parameter 2
    cxb : float, background level (optional)

    Returns:
    S_X : array-like, surface brightness profile
    """

    if cxb==None:
        return S0*(1 + (r / rc) ** 2) ** (-3 * beta + 0.5) + S0_2*(1 + (r / rc2) ** 2) ** (-3 * beta2 + 0.5)
    else:
        return S0*(1 + (r / rc) ** 2) ** (-3 * beta + 0.5) + S0_2*(1 + (r / rc2) ** 2) ** (-3 * beta2 + 0.5) + cxb
        

@staticmethod        
def vikhlinin_profile(r, S0, r_c, beta, alpha, r_s, gamma, epsilon, cxb=None):
    """
    Compute the Vikhlinin (2006) X-ray surface brightness profile.

    Parameters:
    r      : array-like, radii at which to compute the profile
    S0     : float, normalization (central surface brightness)
    r_c    : float, core radius
    beta   : float, outer slope parameter
    alpha  : float, inner steepening exponent
    r_s    : float, radius where steepening occurs
    gamma  : float, smoothness parameter for the transition
    epsilon: float, steepening exponent
    cxb    : float, background level (optional)

    Returns:
    S_X    : array-like, surface brightness profile
    """
    term1 = (r / r_c) ** (-alpha)
    term2 = (1 + (r / r_c) ** 2) ** (-3 * beta + alpha / 2)
    term3 = (1 + (r / r_s) ** gamma) ** (-epsilon / gamma)
    if cxb is None:
        return S0 * term1 * term2 * term3
    else:
        return S0 * term1 * term2 * term3 + cxb

def mock_profiles(N, exp_t, num_bins, profile_type="beta"):
    """
    Generates mock surface brightness profiles for N clusters with a given exposure time and number of bins.

    Parameters:
    -----------
    N : int
        Number of clusters to generate profiles for.
    exp_t : float
        Exposure time in ks.
    num_bins : int
        Number of bins in the profile.

    Returns:
    --------
    profiles : array_like
        Array of mock profiles with shape (N, num_bins).
    bin_centers : array_like
        Array of bin centers with shape (N, num_bins).
    bgsub_prof : array_like
        Array of background subtracted profiles with shape (N, num_bins).
    """
    exp_t=exp_t*1e3 #s
    
    if profile_type == "beta":
        # grid of parameters
        S0=np.random.uniform(1e-5,2e-5,N) #counts/s/arcsec^2
        r_c=np.random.uniform(150,350,N)    #arcsec
        beta=np.random.uniform(0.5,0.7,N)   #dimensionless
        cxb=np.random.uniform(5e-8,1e-7,N) #counts/s/arcsec^2
    elif profile_type == "doublebeta":
        # grid of parameters
        S0=np.random.uniform(1e-5,2e-5,N)
        r_c=np.random.uniform(150,350,N)
        beta=np.random.uniform(0.3,0.6,N)
        cxb=np.random.uniform(8e-8,2e-7,N)
        S02=np.random.uniform(1e-6,4e-6,N)
        r_c2=np.random.uniform(500,800,N)
        beta2=np.random.uniform(0.9,1.2,N)

    elif profile_type == "vikhlinin":
        # grid of parameters
        S0=np.random.uniform(1e-5,2e-5,N)
        r_c=np.random.uniform(150,350,N)
        beta=np.random.uniform(0.5,0.7,N)
        cxb=np.random.uniform(5e-8,1e-7,N)
        alpha=np.random.uniform(0.5,2.0,N)
        gamma=np.random.uniform(2.5,4.0,N)
        epsilon=np.random.uniform(2.0,6.0,N)
        r_s=np.random.uniform(500,2400,N)

    else:
        raise ValueError("Invalid profile type. Choose either 'beta', 'doublebeta', or 'vikhlinin'.")
    
    # grid of bin centers
    threer200=np.random.uniform(10000,18000,N) #arcsec
    # grid of profiles
    profiles = np.zeros((N, num_bins), dtype=float) #blank arrays
    bin_centers= np.zeros((N, num_bins), dtype=float)
    bgsub_prof= np.zeros((N, num_bins), dtype=float)

    for i in range(N):
        r=np.linspace(0,threer200[i],num_bins+1)
        bin_edges = r[1:]
        bin_centers[i,:] = r[:-1]+np.diff(r)/2
        area_bin=np.pi*(bin_edges)**2-np.pi*(np.insert(bin_edges[:-1],0,0))**2  #arcsec^2
        if profile_type == "beta":
            pre_profile=betaprof(bin_centers[i,:],S0[i],r_c[i],beta[i],cxb[i])
        elif profile_type == "doublebeta":
            pre_profile=doublebeta(bin_centers[i,:],S0[i],r_c[i],beta[i],S02[i],r_c2[i],beta2[i],cxb[i])
        elif profile_type == "vikhlinin":
            pre_profile=vikhlinin_profile(bin_centers[i,:],S0[i],r_c[i],beta[i],alpha[i],r_s[i],gamma[i],epsilon[i],cxb[i])
    
        counts=np.ceil(pre_profile*area_bin*exp_t)
        poisson_counts=np.random.poisson(counts)
        profiles[i,:]=poisson_counts/(area_bin*exp_t)
        bgsub_prof[i,:]=profiles[i,:]-cxb[i]

    return profiles, bin_centers, bgsub_prof

@staticmethod
def fitter(x,y,yerr,P0,profile_type="beta"):
    """
    Fit a surface brightness profile with different models.

    Parameters:
    -----------
    x : array_like
        Array of bin centers.
    y : array_like
        Array of surface brightness values.
    yerr : array_like
        Array of surface brightness errors.
    P0 : array_like
        Initial guess for the fit parameters.
    profile_type : str
        Type of profile to fit. Options are 'beta', 'doublebeta', and 'vikhlinin'.

    Returns:
    --------
    bestfit : array_like
        Best-fit surface brightness profile.
    popt : array_like
        Best-fit parameters.
    pcov : array_like
        Covariance matrix.
    """
    if profile_type == "beta":
        popt,pcov=curve_fit(betaprof,x,y,sigma=yerr,p0=P0)
        bestfit=betaprof(x,*popt)
    elif profile_type == "doublebeta":
        popt,pcov=curve_fit(doublebeta,x,y,sigma=yerr,p0=P0)
        bestfit=doublebeta(x,*popt)
    elif profile_type == "vikhlinin":
        popt,pcov=curve_fit(vikhlinin_profile,x,y,sigma=yerr,p0=P0)
        bestfit=vikhlinin_profile(x,*popt)
    else:
        raise ValueError("Invalid profile type. Choose either 'beta', 'doublebeta', or 'vikhlinin'.")    
    return bestfit, popt, pcov


class SBprofile:
    def __init__(self,S_0,r_c,beta,r,r500,CXB=None,SBvals=None,SBerrors=None,radbins=None,bin_halfwidth=None):
        """
        Initialize the SBprofile class.

        Parameters:
        -----------
        S_0 : float
            Central surface brightness. 
        r_c : float
            Core radius.
        beta : float
            Beta parameter.
        r : float
            Any radius of the profile.
        r500 : float
            R500 of the cluster
        CXB : float
            Cosmic X-ray background level.
        SBvals : array_like
            Array of surface brightness values.
        SBerrors : array_like
            Array of surface brightness errors.
        radbins : array_like
            Array of bin centers.
        """
        self.S_0=S_0
        self.r_c=r_c
        self.beta=beta
        self.r=r
        self.r500=r500
        self.SBvals = None
        self.SBerrors = None
        if SBvals is not None and SBerrors is not None and CXB is not None and radbins is not None:
            self.CXB=CXB
            self.SBvals=SBvals
            self.SBerrors=SBerrors
            self.radbins=radbins
            self.bin_halfwidth=bin_halfwidth
        

        

    def sb_plotter(self,arcsectokpc,bgsub=False):
        """
        Plot the surface brightness profile of the cluster.

        Parameters:
        -----------
        arcsectokpc : float
            Conversion factor from arcsec to kpc.
        bgsub : bool
            If True, plot the background subtracted profile.
        """
        SB=self.SBvals
        SBerr=self.SBerrors
        S_0=self.S_0
        r_c=self.r_c
        beta=self.beta
        r500=self.r500
        bgall=self.CXB
        bin_center=self.radbins
        bin_halfwidth=self.bin_halfwidth

        if bgsub==False:
            sb_full=SB
            sb_fullerr=SBerr
        else:
            sub=SB-bgall
            sb_full=noms(sub)
            sb_fullerr=stds(sub)

        rad_bin=bin_center
        differ=bin_halfwidth
        r=bin_center
        if bgsub==False:
            bestf_bmodel=betaprof(bin_center,S_0,r_c,beta,bgall)
        else:       
            bestf_bmodel=betaprof(bin_center,S_0,r_c,beta)
        r200=r500/0.65 #arcsec

        #figure
        fig = plt.figure(figsize=(7,7))

        # Define the grid
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1],hspace=0)

        # Create the upper subplot on the grid
        ax1 = plt.subplot(gs[0])

        # Create the lower subplot on the grid
        ax2 = plt.subplot(gs[1])

        # Plot data on the subplots
        ax1.errorbar(rad_bin,sb_full,yerr=sb_fullerr,xerr=differ,fmt='.',capsize=3,c='#0047ab',label='Surface Brightness')
        ax1.plot(r,noms(bestf_bmodel),label=r'Best-fit $\beta$-Model',color='#f59920',lw='2')
        ax1.axhline(noms(bgall),color='#0066ff',label='CXB',linestyle='--',linewidth=2)
        ax1.fill_between(np.arange(0,np.max(rad_bin)+2000),noms(bgall)-stds(bgall),noms(bgall)+stds(bgall),color='#0066ff',alpha=0.2)
        ax1.axvline(r500,linestyle='dotted',color='r',linewidth=2)
        ax1.axvline(r200,linestyle='dotted',color='r',linewidth=2)
        ax1.text(r500-680,1e-5,r'$R_{500}$',color='r',rotation=90,fontsize=13)
        ax1.text(r200-1040,1e-5,r'$R_{200}$',color='r',rotation=90,fontsize=13)
        ax1.set_xlim(8, np.max(rad_bin)+2000)
        ax1.set_ylabel(r'Surface Brightness [$\mathrm{cts/s/arcsec^2}$]')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # Remove x-axis ticks and labels
        ax1.legend(fontsize=12)
        ax1_top=ax1.twiny()
        ax1_top.plot(rad_bin*arcsectokpc,sb_full,alpha=0)
        ax1_top.set_xlim(8*arcsectokpc,  (np.max(rad_bin)+2000)*arcsectokpc)
        ax1_top.set_xlabel('Radius [kpc]')
        ax1_top.set_xscale('log')



        ax2.scatter(rad_bin, (sb_full-noms(bestf_bmodel))/sb_fullerr, marker='.',c='#692f8f')
        ax2.fill_between(np.arange(0,np.max(rad_bin)+2000), -1, 1, color='#0066ff', alpha=0.2, label='1$\sigma$')
        ax2.set_xlabel('Radius [arcsec]')
        ax2.set_ylabel(r'Residuals [$\sigma$]')
        ax2.set_xscale('log')
        ax2.set_ylim(-5, 5)
        ax2.set_xlim(8, np.max(rad_bin)+2000)
        ax2.axhline(0, color='k', linestyle='--',linewidth=2)
        ax2.yaxis.set_major_locator(MaxNLocator(4))
        ax2.legend(fontsize=12)


        # Display the figure
        plt.tight_layout()
        plt.show()

    def sb_significance(self):
        """
        Compute the significance of the surface brightness profile.

        Returns:
        --------
        significance : float
            Significance of the surface brightness profile above the average CXB 
            level in the unit of sigma.
        """
        sb=self.SBvals
        sberr=self.SBerrors
        bgall=self.CXB
        sig=(sb-noms(bgall))/(sberr**2 + stds(bgall)**2)**0.5
        return sig
  
    def massprofile(self,T,D_a,plot=False):
        """
        Compute the total mass profile of the cluster using an isothermal beta model.

        Parameters:
        -----------
        T : float
            Temperature of the cluster in keV.
        D_a : float
            Angular diameter distance to the cluster in m.
        plot : bool
            If True, plot the mass profile.

        Returns:
        --------
        mass_prof : array_like
            Total mass profile.
        """
        k = 1.381e-23  # J/K
        G = 6.67e-11   # Nm^2/Kg^2
        mu = 0.6
        m_p = 1.67e-27  # kg
        T_k = T * 11.606e6  # K
        r_ang=np.linspace(0,self.r,10000)
        r=np.deg2rad(r_ang/3600)*D_a
        rc_nom=np.deg2rad(noms(self.r_c)/3600)*D_a
        rc_err=np.deg2rad(stds(self.r_c)/3600)*D_a
        rc=unp.uarray(rc_nom,rc_err)
        mass_prof=((3 * self.beta * k * T_k * r**3) / (G * mu * m_p * (r**2 + rc**2))) / 1.9884e30
        
        if plot == True:
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(r_ang/60, noms(mass_prof))
            ax.fill_between(r_ang/60, noms(mass_prof)-stds(mass_prof), noms(mass_prof)+stds(mass_prof), alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Radius [arcmin]',fontsize=14)
            ax.set_ylabel('Total Mass [$M_{\odot}$]',fontsize=14)
            plt.tight_layout()
            plt.show()
        else:
            return mass_prof



class SBcalc:
    def __init__(self, path, filename, singlefile=False,custom_expo=None):
        """
        Class to calculate the surface brightness of a cluster using the output from funcnts.
        
        Parameters:
        -----------
        path : str
            Path to the directory containing the output files.
        filename : str
            Name of the output file.
        singlefile : bool
            If True, the input is from a single file (default is False).
        custom_expo : str
            Custom exposure key for the exposure file (default is None).
        """
        self.path = path
        self.filename = filename
        self.singlefile = singlefile

        if singlefile==False:
            cts_array = []
            exp_array = []
            PIB_array = []
            if custom_expo == None:
                for i in filename:
                    cts=np.loadtxt(f'{path}/{i}_cts_Jul2121.txt',skiprows=1)
                    exp=np.loadtxt(f'{path}/{i}_exp_Jul2121.txt',skiprows=1)
                    PIB=np.loadtxt(f'{path}/{i}_PIB_Jul2121.txt',skiprows=1)

                    cts_array.append(cts)
                    exp_array.append(exp)
                    PIB_array.append(PIB)
            else:
                key=custom_expo
                for i in filename:
                    cts=np.loadtxt(f'{path}/{i}_cts_Jul2121.txt',skiprows=1)
                    exp=np.loadtxt(f'{path}/{i}_{key}_exp_Jul2121.txt',skiprows=1)
                    PIB=np.loadtxt(f'{path}/{i}_PIB_Jul2121.txt',skiprows=1)

                    cts_array.append(cts)
                    exp_array.append(exp)
                    PIB_array.append(PIB)

            cts_array=np.array(cts_array)
            exp_array=np.array(exp_array)
            PIB_array=np.array(PIB_array)            

            self.reg_cts=cts_array[:,1]
            self.reg_cts_err=cts_array[:,2]
            self.reg_area=cts_array[:,3]

            self.reg_exp=exp_array[:,1]
            self.reg_experr=exp_array[:,2]
            
            self.PIB=PIB_array[:,1]
            self.PIBerr=PIB_array[:,2]

        else:
            if custom_expo == None:
                cts=np.loadtxt(f'{path}/{filename}_cts_Jul2121.txt',skiprows=1)
                exp=np.loadtxt(f'{path}/{filename}_exp_Jul2121.txt',skiprows=1)
                PIB=np.loadtxt(f'{path}/{filename}_PIB_Jul2121.txt',skiprows=1)
            else:
                key=custom_expo
                cts=np.loadtxt(f'{path}/{filename}_cts_Jul2121.txt',skiprows=1)
                exp=np.loadtxt(f'{path}/{filename}_{key}_exp_Jul2121.txt',skiprows=1)
                PIB=np.loadtxt(f'{path}/{filename}_PIB_Jul2121.txt',skiprows=1)

            if cts.shape == (6,):
                self.reg_cts=cts[1]
                self.reg_cts_err=cts[2]
                self.reg_area=cts[3]

                self.reg_exp=exp[1]
                self.reg_experr=exp[2]
                
                self.PIB=PIB[1]
                self.PIBerr=PIB[2]
                
            else:
                self.reg_cts=cts[:,1]
                self.reg_cts_err=cts[:,2]
                self.reg_area=cts[:,3]

                self.reg_exp=exp[:,1]
                self.reg_experr=exp[:,2]
                
                self.PIB=PIB[:,1]
                self.PIBerr=PIB[:,2]


    def SBcalculator(self,method,group,pix_size=None,ecf=None):
        """
        Calculate the surface brightness of the cluster.

        Parameters:
        -----------
        method : str
            Method to calculate the surface brightness. Options are 'default' which uses gaussian propagation to calculate uncertainties, 
            'cts_err' which only takes into account the counts error from the PIB and photon image, and 'std_err' which calculates uncertainties on average values as the
            average of the individual nominal values of the regions involved (used for CXB estimation).
        group : bool
            If True, group the regions in the region file together.
        pix_size : float
            Pixel size in arcsec (default is 4 for eROSITA).
        ecf : float
            Energy conversion factor to convert count-rate to flux units (default is 1).

        Returns:
        --------
        SB_an_nom : float
            Surface brightness value.
        SB_an_er : float
            Surface brightness error.
        """
        if pix_size==None:
        	pixel_size=4  #arcsec
        else:
            pixel_size=pix_size

        if ecf==None:
            factor=1
        else:
            factor=ecf
                
        net_cts=unp.uarray(self.reg_cts, self.reg_cts_err)
        exp_time=unp.uarray(self.reg_exp/(self.reg_area/pixel_size**2), self.reg_experr/(self.reg_area/pixel_size**2))
        PIB=unp.uarray(self.PIB, self.PIBerr)

        if self.singlefile == False:
            if group == True:
                if method == "default":
                    SB_an = ((net_cts - PIB)*factor / exp_time / self.reg_area)
                    SB_an_nom = np.mean(noms(SB_an))
                    SB_an_er = np.mean(stds(SB_an))
                    return SB_an_nom, SB_an_er
                
                elif method == "cts_err":
                    SB_an = ((net_cts - noms(PIB))*factor / noms(exp_time) / self.reg_area)
                    SB_an_nom = np.mean(noms(SB_an))
                    SB_an_er = np.mean(stds(SB_an))
                    return SB_an_nom, SB_an_er
                
                elif method == "std_err":
                    SB_an = ((net_cts - PIB)*factor / exp_time / self.reg_area)
                    SB_an_nom = np.mean(noms(SB_an))
                    SB_an_er = np.std(noms(SB_an))
                    return SB_an_nom, SB_an_er
                
            else:
                if method == "default":
                    SB_an = ((net_cts - PIB)*factor / exp_time / self.reg_area)
                    SB_an_nom = noms(SB_an)
                    SB_an_er = stds(SB_an)
                    return SB_an_nom, SB_an_er
                
                elif method == "cts_err":
                    SB_an = ((net_cts - noms(PIB))*factor / noms(exp_time) / self.reg_area)
                    SB_an_nom = noms(SB_an)
                    SB_an_er = stds(SB_an)
                    return SB_an_nom, SB_an_er
                
                else:
                    raise AttributeError("This method can't be used.")
                
        else:
            if group == True:
                if method == "default":
                    SB_an = ((net_cts - PIB)*factor / exp_time / self.reg_area)
                    SB_an_nom = np.mean(noms(SB_an))
                    SB_an_er = np.mean(stds(SB_an))
                    return SB_an_nom, SB_an_er
                
                elif method == "cts_err":
                    SB_an = ((net_cts - noms(PIB))*factor / noms(exp_time) / self.reg_area)
                    SB_an_nom = np.mean(noms(SB_an))
                    SB_an_er = np.mean(stds(SB_an))
                    return SB_an_nom, SB_an_er
                
                elif method == "std_err":
                    SB_an = ((net_cts - PIB)*factor / exp_time / self.reg_area)
                    SB_an_nom = np.mean(noms(SB_an))
                    SB_an_er = np.std(noms(SB_an))
                    return SB_an_nom, SB_an_er
                
            else:
                if method == "default":
                    SB_an = ((net_cts - PIB)*factor / exp_time / self.reg_area)
                    SB_an_nom = noms(SB_an)
                    SB_an_er = stds(SB_an)
                    return SB_an_nom, SB_an_er
                
                elif method == "cts_err":
                    SB_an = ((net_cts - noms(PIB))*factor / noms(exp_time) / self.reg_area)
                    SB_an_nom = noms(SB_an)
                    SB_an_er = stds(SB_an)
                    return SB_an_nom, SB_an_er
                
                else:
                    raise AttributeError("This method can't be used.")
                

class RedshiftDistribution:
    """
    Class to analyze any redshift distribution of galaxies.
    """
    def __init__(self,distribution=None,filename=None):
        if filename is not None and distribution is None:
            self.filename=filename
            self.data=pd.read_csv(filename)
            self.z=np.array(self.data['Redshift'])
        elif filename is None and distribution is not None:
            self.z=distribution
        else:
            raise ValueError("Either provide a redshift distribution or a filename.")       
    
    def clipandplot(self,zcutL,zcutH):
        """
        Method to clip and plot a redshift distribution.
        
        Parameters:
        -----------
        zcutL : float
            Lower redshift cut.
        zcutH : float
            Upper redshift cut.

        Returns:
        --------
        z : array_like
            Clipped redshift distribution.
        freq : array_like
            Frequency of redshifts in each bin.
        bin_cent : array_like
            Bin centers.
        """
        z=self.z
        z = z[(z <= zcutH) & (z >= zcutL)]
        freq, bin_st=np.histogram(z,bins=20)
        bin_cent=(bin_st[1:]+bin_st[:-1])/2
        plt.hist(z,bins=20)
        plt.scatter(bin_cent,freq,c='black',s=20)
        plt.xlabel("Redshift")
        plt.ylabel("Frequency")
        plt.show()
        return z, freq, bin_cent
    
    def clipper(self,zcutL,zcutH,iterations,sigma,method="nsigma"):
        """
        Redshift determination method that iteratively clips the redshift distribution using clipping algorithms.
        
        Parameters:
        -----------
        zcutL : float
            Lower redshift cut.
        zcutH : float
            Upper redshift cut.
        iterations : int
            Number of iterations to perform.
        sigma : float
            Sigma value for clipping.
        method : str
            Method to use for clipping. Choose either 'nsigma' or 'adaptive_nsigma'.
        
        Returns:
        --------
        z : array_like
            Clipped redshift distribution.
        res_mean : array_like
            Mean values of the redshift distribution at each iteration.
        res_std : array_like
            Standard deviation of the redshift distribution at each iteration.
        res_median : array_like
            Median values of the redshift distribution at each iteration.
        res_mad : array_like
            Median absolute deviation of the redshift distribution at each iteration.

        """
        self.z=self.z[~np.isnan(self.z)]
        z=self.z
        z = z[(z <= zcutH) & (z >= zcutL)]
        if method=="nsigma":
            res_mean, res_std,res_median, res_mad=[],[],[],[]
            for _ in range(iterations):
                mean = np.mean(z)
                median=np.median(z)
                std_dev = np.std(z)
                med_abs_dev=mad(z)
                low_cut= mean - (sigma * std_dev)
                if low_cut > 0:
                    z = z[(z > low_cut) & (z < mean + sigma * std_dev)] #clips both ends of the distribution
                else:
                    z = z[z < mean + sigma * std_dev] #only clips the upper end as the lower end is already below zero
                print(f"Iteration {_+1}: Mean {mean:.4f}, Std. Dev. {std_dev:.4f}")
                res_mean.append(mean), res_std.append(std_dev),res_median.append(median), res_mad.append(med_abs_dev) 
            return z, res_mean, res_std, res_median, res_mad
        
        elif method=='adaptive_nsigma':
            res_mean, res_std=[],[]
            it_counter=0
            while sigma >= 1.0:
                it_counter+=1
                mean = np.mean(z)
                std_dev = np.std(z)
                low_cut= mean - (sigma * std_dev)
                if low_cut > 0:
                    z = z[(z > low_cut) & (z < mean + sigma * std_dev)]
                else:
                    z = z[z < mean + sigma * std_dev]
                res_mean.append(mean), res_std.append(std_dev)
                if len(res_mean) > 3 and np.isclose(res_mean[-1],res_mean[-2],rtol=1e-06,atol=1e-5) and np.isclose(res_mean[-1],res_mean[-2],rtol=1e-05,atol=1e-05):
                    sigma-=0.5
                if sigma <= 1 and (np.isclose(res_mean[-1],res_mean[-2],atol=1e-4)):
                    break
                print(f"Iteration {it_counter}, Sigma {sigma}: Mean {mean:.4f}, Std. Dev. {std_dev:.4f}")
                
            return z, res_mean, res_std
        

    def threeDDistribution(self, min_z, max_z, sep_RA=None, sep_Dec=None, redshift_type='spectroscopic'):
        """
        Method to plot a 3D distribution of galaxies in RA, DEC, and redshift space.
        
        Parameters:
        -----------
        min_z : float
            Minimum redshift cut.
        max_z : float
            Maximum redshift cut.
        sep_RA : array_like (if filename is not provided)
            Array of RA values.
        sep_Dec : array_like (if filename is not provided)
            Array of DEC values.
        redshift_type : str
            Type of redshift to plot. Choose either 'all', 'spectroscopic', or 'photometric'.

        Returns:
        --------
        3D plot of the galaxy distribution.    
        """
        if hasattr(self, 'filename'):
            data=pd.read_csv(self.filename)
            ra=np.array(data['RA'])
            dec=np.array(data['DEC'])
            z=self.z
            if redshift_type=='all':
                z=self.z
                idx_cut = np.argwhere((min_z < z) & (z < max_z)).flatten()
                z_plot = z[idx_cut]
                ra_plot = ra[idx_cut]
                dec_plot = dec[idx_cut]

            elif redshift_type=='spectroscopic':
                z_plot = z[np.where((data['Redshift Flag'].str.strip() == 'SLS') | 
                                    (data['Redshift Flag'].str.strip() == 'SUN'))[0]]
                ra_plot = ra[np.where((data['Redshift Flag'].str.strip() == 'SLS') | 
                                      (data['Redshift Flag'].str.strip() == 'SUN'))[0]]
                dec_plot = dec[np.where((data['Redshift Flag'].str.strip() == 'SLS') |
                                         (data['Redshift Flag'].str.strip() == 'SUN'))[0]]
                idx_cut_spec = np.argwhere((min_z < z_plot) & (z_plot < max_z)).flatten()
                z_plot = z_plot[idx_cut_spec]
                ra_plot = ra_plot[idx_cut_spec]
                dec_plot = dec_plot[idx_cut_spec]

            elif redshift_type=='photometric':
                z_plot = z[np.where((data['Redshift Flag'].str.strip() == 'PUN'))[0]]
                ra_plot = ra[np.where((data['Redshift Flag'].str.strip() == 'PUN'))[0]]
                dec_plot = dec[np.where((data['Redshift Flag'].str.strip() == 'PUN'))[0]]
                idx_cut_photo = np.argwhere((min_z < z_plot) & (z_plot < max_z)).flatten()
                z_plot = z_plot[idx_cut_photo]
                ra_plot = ra_plot[idx_cut_photo]
                dec_plot = dec_plot[idx_cut_photo]

            else:
                raise ValueError("Invalid redshift type. Choose either 'all', 'spectroscopic', or 'photometric'.")
            
        else:
            ra=sep_RA
            dec=sep_Dec
            z=self.z
            idx_cut = np.argwhere((min_z < z) & (z < max_z)).flatten()
            z_plot = z[idx_cut]
            ra_plot = ra[idx_cut]
            dec_plot = dec[idx_cut]

        fig = go.Figure(data=[go.Scatter3d(
            x=ra_plot,
            y=dec_plot,
            z=z_plot,
            mode='markers',
            marker=dict(
                size=2,
                color=z_plot,                # set color to an array/list of values
                colorscale='Viridis',      # choose a colorscale
                opacity=1.0
            )
        )])

        fig.update_layout(
            scene=dict(
                xaxis_title='RA',
                yaxis_title='DEC',
                zaxis_title='Redshift'
            ),
            coloraxis_colorbar=dict(
                title='Redshift'
            ),
            width=700,
            height=700
        )

        fig.show()
        
def substructured_density(grid_size=300, rad=4000, n0L=1e-3, n0H=1e-1, rcL=200, rcH=700, betaL=0.4, betaH=0.9, num_profiles=None, random_seed=None):
    """
    Generates an image with multiple beta models slightly offset to create substructures.
    
    Parameters:
    -----------
    grid_size : int
        Size of the square image (NxN pixels).
    rad : float
        Maximum radial extent (arcsec).
    n0L : float
        Minimum central density of the beta model in cm^-3.
    n0H : float
        Maximum central density of the beta model in cm^-3.
    rcL : float
        Minimum core radius of the beta model in arcsec.
    rcH : float
        Maximum core radius of the beta model in arcsec.
    betaL : float
        Minimum beta parameter of the beta model.
    betaH : float
        Maximum beta parameter of the beta model.        
    num_profiles : int or None
        Number of beta profiles to sum (if None, randomly choose 2, 3, or 4).
    random_seed : int or None
        Seed for reproducibility.

    Returns:
    --------
    image : 2D numpy array
        The final generated image with substructures.
    """
    def beta_model(r, ne, r_c, beta):
        return ne * (1 + (r / r_c)**2)**(-3 * beta/2)
    
    if random_seed is not None:
        np.random.seed(random_seed)

    # 2D grid
    x = np.linspace(-rad, rad, grid_size)
    y = np.linspace(-rad, rad, grid_size)
    x_grid, y_grid = np.meshgrid(x, y)
    r_grid = np.sqrt(x_grid**2 + y_grid**2)

    # Randomly decide number of components 
    if num_profiles is None:
        num_profiles = np.random.choice([2, 3, 4])

    image = np.zeros((grid_size, grid_size))

    # Generate beta model components
    for _ in range(num_profiles):
        # Randomly shift centers slightly
        shift=np.random.uniform(0.08,0.2)
        x_shift = np.random.uniform(-shift * rad, shift * rad)
        y_shift = np.random.uniform(-shift * rad, shift * rad)
        r_shifted = np.sqrt((x_grid - x_shift) ** 2 + (y_grid - y_shift) ** 2)
        
        # Random beta model parameters
        ne = np.random.uniform(n0L,n0H) 
        r_c = np.random.uniform(rcL,rcH)  
        beta = np.random.uniform(betaL, betaH)  

        # Generate beta profile and add it to the image
        image += beta_model(r_shifted, ne, r_c, beta)

    return image
        



