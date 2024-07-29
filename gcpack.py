"""
Version 1.0
Author: Alpish Srivastava
Date: 29 July 2021

"""


import numpy as np
import matplotlib.pyplot as plt
from uncertainties import unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm

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
        if cxb==None:
            return S0*(1 + (r / rc) ** 2) ** (-3 * beta + 0.5)
        else:
            return S0*(1 + (r / rc) ** 2) ** (-3 * beta + 0.5) + cxb

@staticmethod
def beta_model_fit(x,y,yerr,P0):
    popt,pcov= curve_fit(betaprof,x,y,yerr=yerr,p0=P0)
    bestfit=betaprof(x,*popt)
    return bestfit, popt, pcov


class SBprofile:
    def __init__(self,S_0,r_c,beta,r,r500,CXB=None,SBvals=None,SBerrors=None,radbins=None):
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

        

    def beta_model(self,r=None,values=True, errors=True):
        if r is not None:
            rad=r
        else:
            rad = np.linspace(0, self.r, 1000)
        if values and errors:
            return self.S_0 * (1 + (rad / self.r_c) ** 2) ** (-3 * self.beta + 0.5)
        elif values and not errors:
            return noms(self.S_0 * (1 + (rad / self.r_c) ** 2) ** (-3 * self.beta + 0.5))
        else:
            raise AttributeError("The beta model can't return only uncertainties!")
        
  
    def plotter(self,innerbin):

        SB_vals=unp.uarray(self.SBvals,self.SBerrors)-self.CXB
        SB=noms(SB_vals)
        SBerr=stds(SB_vals)
        r_full=np.linspace(innerbin, self.r, len(self.SBvals))

        
        fig = plt.figure(figsize=(7,7))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1],hspace=0)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax1.errorbar(self.radbins, SB, yerr=SBerr, fmt='.',barsabove=True,capsize=3,c='#692f8f',label='Surface Brightness')
        ax1.plot(np.linspace(innerbin,self.r,1000),self.beta_model(values=True,errors=False),label=r'$\beta$ model',color='#f59920',lw='2')
        ax1.axhline(noms(self.CXB),linestyle='--',color='green',label='CXB')
        ax1.set_ylabel(r'Surface Brightness [$\mathrm{cts/s/arcsec^2}$]')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # Remove x-axis ticks and labels
        ax1.legend(fontsize=12)
        ax1_top=ax1.twiny()
        ax1_top.plot(r_full*0.2435,np.zeros(len(r_full)),alpha=0)
        ax1_top.set_xlabel('Radius [kpc]')
        ax1_top.set_xscale('log')
        ax2.scatter(self.radbins, (SB-self.beta_model(r=self.radbins,errors=False))/SBerr, marker='.',c='#692f8f')
        ax2.fill_between(np.linspace(innerbin, self.r, 100), -1, 1, color='grey', alpha=0.2, label='1$\sigma$')
        ax2.set_xlabel('Radius [arcsec]')
        ax2.set_ylabel(r'Residuals [$\sigma$]')
        ax2.set_xscale('log')
        ax2.set_ylim(-5, 5)
        ax2.axhline(0, color='r', linestyle='--')
        ax2.yaxis.set_major_locator(MaxNLocator(4))
        ax2.legend(fontsize=12)

        # Display the figure
        plt.tight_layout()
        plt.show()

  
    def massprofile(self,T,D_a,plot=False):
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
            ax.set_ylabel('Integrated Total Mass [$M_{\odot}$]',fontsize=14)
            plt.tight_layout()
            plt.show()
        else:
            return mass_prof



class SBcalc:
    def __init__(self, path, filename, singlefile=False):
        self.path = path
        self.filename = filename
        self.singlefile = singlefile

        if singlefile==False:
            cts_array = []
            exp_array = []
            PIB_array = []
            for i in filename:
                cts=np.loadtxt(f'{path}/{i}_cts_Jul2121.txt',skiprows=1)
                exp=np.loadtxt(f'{path}/{i}_exp_Jul2121.txt',skiprows=1)
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
            cts=np.loadtxt(f'{path}/{filename}_cts_Jul2121.txt',skiprows=1)
            exp=np.loadtxt(f'{path}/{filename}_exp_Jul2121.txt',skiprows=1)
            PIB=np.loadtxt(f'{path}/{filename}_PIB_Jul2121.txt',skiprows=1)

            self.reg_cts=cts[:,1]
            self.reg_cts_err=cts[:,2]
            self.reg_area=cts[:,3]

            self.reg_exp=exp[:,1]
            self.reg_experr=exp[:,2]
            
            self.PIB=PIB[:,1]
            self.PIBerr=PIB[:,2]


    def SBcalculator(self,method,group):
        pixel_size=4  #arcsec
        net_cts=unp.uarray(self.reg_cts, self.reg_cts_err)
        exp_time=unp.uarray(self.reg_exp/(self.reg_area/pixel_size**2), self.reg_experr/(self.reg_area/pixel_size**2))
        PIB=unp.uarray(self.PIB, self.PIBerr)

        if self.singlefile == False:
            if group == True:
                if method == "default":
                    SB_an = ((net_cts - PIB) / exp_time / self.reg_area)
                    SB_an_nom = np.mean(noms(SB_an))
                    SB_an_er = np.mean(stds(SB_an))
                    return SB_an_nom, SB_an_er
                
                elif method == "cts_err":
                    SB_an = ((net_cts - noms(PIB)) / noms(exp_time) / self.reg_area)
                    SB_an_nom = np.mean(noms(SB_an))
                    SB_an_er = np.mean(stds(SB_an))
                    return SB_an_nom, SB_an_er
                
                elif method == "std_err":
                    SB_an = ((net_cts - PIB) / exp_time / self.reg_area)
                    SB_an_nom = np.mean(noms(SB_an))
                    SB_an_er = np.std(noms(SB_an))
                    return SB_an_nom, SB_an_er
                
            else:
                if method == "default":
                    SB_an = ((net_cts - PIB) / exp_time / self.reg_area)
                    SB_an_nom = noms(SB_an)
                    SB_an_er = stds(SB_an)
                    return SB_an_nom, SB_an_er
                
                elif method == "cts_err":
                    SB_an = ((net_cts - noms(PIB)) / noms(exp_time) / self.reg_area)
                    SB_an_nom = noms(SB_an)
                    SB_an_er = stds(SB_an)
                    return SB_an_nom, SB_an_er
                
                else:
                    raise AttributeError("This method can't be used.")
                
        else:
            if group == True:
                if method == "default":
                    SB_an = ((net_cts - PIB) / exp_time / self.reg_area)
                    SB_an_nom = np.mean(noms(SB_an))
                    SB_an_er = np.mean(stds(SB_an))
                    return SB_an_nom, SB_an_er
                
                elif method == "cts_err":
                    SB_an = ((net_cts - noms(PIB)) / noms(exp_time) / self.reg_area)
                    SB_an_nom = np.mean(noms(SB_an))
                    SB_an_er = np.mean(stds(SB_an))
                    return SB_an_nom, SB_an_er
                
                elif method == "std_err":
                    SB_an = ((net_cts - PIB) / exp_time / self.reg_area)
                    SB_an_nom = np.mean(noms(SB_an))
                    SB_an_er = np.std(noms(SB_an))
                    return SB_an_nom, SB_an_er
                
            else:
                if method == "default":
                    SB_an = ((net_cts - PIB) / exp_time / self.reg_area)
                    SB_an_nom = noms(SB_an)
                    SB_an_er = stds(SB_an)
                    return SB_an_nom, SB_an_er
                
                elif method == "cts_err":
                    SB_an = ((net_cts - noms(PIB)) / noms(exp_time) / self.reg_area)
                    SB_an_nom = noms(SB_an)
                    SB_an_er = stds(SB_an)
                    return SB_an_nom, SB_an_er
                
                else:
                    raise AttributeError("This method can't be used.")
