import sympy as smp
import mpmath as mp
import glob
import os
from os.path import exists
import pandas as pd
from decimal import *
import scipy.integrate as integrate
import numpy as np
from matplotlib import pyplot as plt
import math
import random
import time
import types
from skhep.math.vectors import LorentzVector, Vector3D
from scipy import interpolate
from matplotlib import gridspec
from numba import jit
from particle import Particle
from cycler import cycler





class Utility():

    ###############################
    #  Hadron Masses, lifetimes etc
    ###############################

    def charges(self, pid):
        try:
            charge = Particle.from_pdgid(int(pid)).charge
        except:
            charge = 0.0
        return charge if charge!=None else 0.0

    def masses(self,pid,mass=0):
        pidabs = abs(int(pid))
        #Treat select entries separately
        if   pidabs==0: return mass
        elif pidabs==4: return 1.5   #GeV, scikit-particle returns 1.27 for c quark
        elif pidabs==5: return 4.5   #GeV, scikit-particle returns 4.18 for b quark
        #General case: fetch values from scikit-particle
        else:
            mret = Particle.from_pdgid(pidabs).mass   #MeV
            return mret*0.001 if mret!=None else 0.0  #GeV

    def ctau(self,pid):
        pidabs = abs(int(pid))
        ctau = 0.0
        try:
            ctau = Particle.from_pdgid(pidabs).ctau
        except:
            print('WARNING '+str(pid)+' ctau not obtained from scikit-particle')
        if pidabs in [2212]: ctau=8.51472e+48  #Avoid inf return value in code
        return ctau*0.001

    def widths(self, pid):
        width = 0.0
        try:
            width = Particle.from_pdgid(int(pid)).width
        except:
            print('WARNING '+str(pid)+' width not obtained from scikit-particle, returning 0')
        return width*1e-6 if width!=None else 0.0

    ###############################
    #  Utility Functions
    ###############################

    #function that reads a table in a .txt file and converts it to a numpy array
    def readfile(self,filename):
        array = []
        with open(filename) as f:
            for line in f:
                if line[0]=="#":continue
                words = [float(elt.strip()) for elt in line.split( )]
                array.append(words)
        return np.array(array)

    def integrate_3body_br(self, br, mass, m0, m1, m2, coupling=1, nsample=100, integration="dq2dE"):

        if m0<m1+m2+mass: return 0 
        if integration == "dq2dE":
            return self.integrate_3body_br_3body_dq2dE(br, coupling, m0, m1, m2, mass, nsample)
        if integration == "dE":
            return self.integrate_3body_br_3body_dE(br, coupling, m0, m1, m2, mass, nsample)
        if integration == 'dq2dm122':
            return self.integrate_3body_br_3body_dq2dm122(br, coupling, m0, m1, m2, mass, nsample)

    
    def integrate_3body_br_3body_dq2dE(self,br, coupling, m0, m1, m2, mass, nsample):

        #integration boundary
        m3 = mass
        q2min,q2max = (m2+m3)**2,(m0-m1)**2

        #numerical integration
        integral=0
        for i in range(nsample):
            # sample q2
            q2 = random.uniform(q2min,q2max)
            q  = math.sqrt(q2)
            # sample energy
            E2st = (q**2 - m2**2 + m3**2)/(2*q)
            E3st = (m0**2 - q**2 - m1**2)/(2*q)
            m232min = (E2st + E3st)**2 - (np.sqrt(E2st**2 - m3**2) + np.sqrt(E3st**2 - m1**2))**2
            m232max = (E2st + E3st)**2 - (np.sqrt(E2st**2 - m3**2) - np.sqrt(E3st**2 - m1**2))**2
            ENmax = (m232max + q**2 - m2**2 - m1**2)/(2*m0)
            ENmin = (m232min + q**2 - m2**2 - m1**2)/(2*m0)
            energy = random.uniform(ENmin,ENmax)
            #branching fraction
            integral += eval(br)*(q2max-q2min)*(ENmax-ENmin)/float(nsample)

        return integral


    def integrate_3body_br_3body_dq2dm122(self,br, coupling, m0, m1, m2, mass, nsample):
            #integration boundary
            m3 = mass
            m12sqmin, m12sqmax = (m1+m2)**2, (m0-m3)**2
            #numerical integration
            integral=0
            for i in range(nsample):
                    # sample q2
                    m12sq = random.uniform(m12sqmin,m12sqmax)

                    # sample energy
                    E2st = (m12sq - m1**2 + m2**2)/(2*np.sqrt(m12sq))
                    E3st = (m0**2 - m12sq - m3**2)/(2*np.sqrt(m12sq))
                    m232min = (E2st + E3st)**2 - (np.sqrt(E2st**2 - m2**2) + np.sqrt(E3st**2 - m3**2))**2
                    m232max = (E2st + E3st)**2 - (np.sqrt(E2st**2 - m2**2) - np.sqrt(E3st**2 - m3**2))**2
                    #q2 is m23sq
                    q2 = random.uniform(m232min,m232max)
                    #branching fraction
                    #print(eval(br))
                    integral += eval(br)*(m232max-m232min)*(m12sqmax-m12sqmin)/float(nsample)

            return integral

    
    def integrate_3body_br_3body_dE(self, br, coupling, m0, m1, m2, mass, nsample):

        #integration boundary
        m3 = mass
        emin, emax = m3, (m0**2+m3**2-(m1+m2)**2)/(2*m0)

        #numerical integration
        integral=0
        for i in range(nsample):
            #sample energy
            energy = random.uniform(emin,emax)
            #branching fraction
            integral += eval(br) * (emax-emin)/float(nsample)

        return integral 

    #initializes plot
    def initialize_plot(x_label, y_label, title, xlims, ylims, scale = 'log'):
        #setup figures
        fig,ax = plt.subplots(1,1)

        #define custom color cycler
        custom_cycler = (cycler(color=['tab:blue','tab:orange','tab:purple', 'y', 'r','green','c'])* 
                        cycler(ls=['-', '--', '-.', 'dotted']))

        ax.set_prop_cycle(custom_cycler)

        fig.set_size_inches(18,7, forward=True)

        ax.set_title(rf"{title}",fontsize = 16)

        if scale == 'log': ax.set(xscale = 'log', yscale = 'log',xlim=xlims,ylim = ylims)

        ax.tick_params(axis='both', which='major',direction='in',top=True,right=True)

        ax.tick_params(axis='both', which='minor',direction='in',top=True,right=True)

        ax.set_xlabel(x_label,fontsize=15)
        ax.set_ylabel(y_label,fontsize=15)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        return(fig,ax)



class HNLCalc(Utility):

    ###############################
    #  Initiate
    ###############################

    def __init__(self, ve=1, vmu=0, vtau=0):
        
        mag = np.sqrt(ve**2 + vmu**2 + vtau**2)
        
        
        self.vcoupling = {"11": ve/mag, "13":vmu/mag, "15": vtau/mag}   #HNL coupling to the electron, the muon and the tau lepton
        self.lepton = {"11": "e", "13":"mu", "15": "tau"}
        self.hadron = {"211": "pi", "321": "K", "213": "rho"}
        self.generators_light = None
        self.generators_heavy = None
        
        self.HNL_Decay_init(couplings = (self.vcoupling['11'],self.vcoupling['13'],self.vcoupling['15']))

        self.mass_543 = 6.400
    
    #decay constants
    def fH(self,pid):
        if   pid in ["211","-211","111"]: return 0.1303
        elif pid in ["221","-221"]: return 0.0784   
        elif pid in ["213","-213"]: return 0.220  
        elif pid in ["113"]       : return 0.220       
        elif pid in ["223","-223"]: return 0.195    
        elif pid in ["333"]: return 0.229               
        elif pid in ["313","-313","323","-323"]: return 0.204 
        elif pid in ["311","-311","321","-321"]: return 0.1564 
        elif pid in ["331","-331"]: return -0.0957
        elif pid in ["421","-421","411","-411"]: return 0.2226 
        elif pid in ["443"]: return 0.409         
        elif pid in ["431","-431"]: return 0.2801
        elif pid in ["511", "511","521","-521"]: return 0.190
        elif pid in ["513","-513"]: return 1.027*0.190
        elif pid in ["531","-531"]: return 0.230  
        elif pid in ["523","-523"]: return 1.027*0.190
        elif pid in ["533","-533"]: return 1.028*0.230        
        elif pid in ["541","-541"]: return 0.480
        elif pid in ["413","-413","423","-423"]: return 1.097*0.2226 
        elif pid in ["433","-433"]: return 1.093*0.2801

    # Lifetimes
    def tau(self,pid):
        if   pid in ["2112","-2112"]: return 10**8
        elif pid in ["15","-15"    ]: return 290.3*1e-15
        elif pid in ["2212","-2212"]: return 10**8
        elif pid in ["211","-211"  ]: return 2.603*10**-8
        elif pid in ["323","-323"  ]: return 1.425*10**-23 
        elif pid in ["321","-321"  ]: return 1.2380*10**-8
        elif pid in ["411","-411"  ]: return 1040*10**-15
        elif pid in ["421","-421"  ]: return 410*10**-15
        elif pid in ["423", "-423" ]: return 3.1*10**-22
        elif pid in ["431", "-431" ]: return 504*10**-15
        elif pid in ["511", "-511" ]: return 1.519*10**-12
        elif pid in ["521", "-521" ]: return 1.638*10**-12
        elif pid in ["531", "-531" ]: return 1.515*10**-12
        elif pid in ["541", "-541" ]: return 0.507*10**-12
        elif pid in ["310" ,"-310"        ]: return 8.954*10**-11
        elif pid in ["130" ,"-130"        ]: return 5.116*10**-8
        elif pid in ["3122","-3122"]: return 2.60*10**-10
        elif pid in ["3222","-3222"]: return 8.018*10**-11
        elif pid in ["3112","-3112"]: return 1.479*10**-10
        elif pid in ["3322","-3322"]: return 2.90*10**-10
        elif pid in ["3312","-3312"]: return 1.639*10**-10
        elif pid in ["3334","-3334"]: return 8.21*10**-11
        elif pid in ["4122","-4122"]: return 201.5*10**-15
        elif pid in ["5122","-5122"]: return 1.471*10**-12
        elif pid in ["4132","-4132"]: return 151.9*10**-15
        elif pid in ["5232","-5232"]: return 1.48*10**-12
        elif pid in ["5332","-5332"]: return 1.64*10**-12

    # CKM matrix elements
    def VH(self,pid):
        if   pid in ["211","-211","213","-213"]: return 0.97373 #Vud
        elif pid in ["321","-321","323","-323"]: return 0.2243 #Vus
        elif pid in ["213","-213"]: return 0.97373
        elif pid in ["411","-411"]: return 0.221
        elif pid in ["431","-431","433","-433"]: return 0.975 #Vcs
        elif pid in ["541","-541"]: return 40.8E-3
        elif pid in ["411","-411","413","-413"]: return 0.221 #Vcd
        elif pid in ["541","-541"]: return 40.8E-3 #Vcb
        elif pid in ["521","-521","523","-523"]: return 3.82E-3 #Vub
        elif pid in []: return 8.6E-3 #Vtd
        elif pid in []: return 41.5E-3 #Vts
        elif pid in []: return 1.014 #Vtb

    #symbol for a given pid
    #originally created to analyze HNL decays
    def symbols(self,pid):
        #quarks
        if   pid in ["1"]: return "d"
        elif pid in ["2"]: return "u"
        elif pid in ["3"]: return "s"
        elif pid in ["4"]: return "c"
        elif pid in ["5"]: return "b"
        elif pid in ["6"]: return "t"

        #leptons
        elif pid in ["11"]: return "e"
        elif pid in ["12"]: return r"$\nu_e$"
        elif pid in ["13"]: return r"$\mu$"
        elif pid in ["14"]: return r"$\mu_e$"
        elif pid in ["15"]: return r"$\tau$"
        elif pid in ["16"]: return r"$\nu_{\tau}$"

        #neutral pseudoscalars
        elif pid in ["111"]: return r"$\pi^0$"
        elif pid in ["221"]: return r"$\eta$"
        elif pid in ["311"]: return r"$K^0$"
        elif pid in ["331"]: return r"$\eta^{'}$"
        elif pid in ["421"]: return r"$D^0$"

        #charged pseudoscalars
        elif pid in ["211"]: return r"$\pi^+$"
        elif pid in ["321"]: return r"$K^+$"
        elif pid in ["411"]: return r"$D^+$"
        elif pid in ["431"]: return r"$D^+_s$"
        elif pid in ["521"]: return r"$B^+$"

        #neutral vectors
        elif pid in ["113"]: return r"$\rho^0$"
        elif pid in ["223"]: return r"$\omega$"
        elif pid in ["313"]: return r"$K^{*0}$"
        elif pid in ["333"]: return r"$\phi$"
        elif pid in ["423"]: return r"$D^{*0}$"
        elif pid in ["443"]: return r"$J/\psi$"

        #charged vectors
        elif pid in ["213"]: return r"$\rho^+$"
        elif pid in ["323"]: return r"$K^{*+}$"
        elif pid in ["413"]: return r"$D^{*+}$"
        elif pid in ["433"]: return r"$D^{*+}_s$"

    #for HNL decays to neutral vector mesons
    def kV(self,pid):
        xw=0.23121
        if pid in ["313","-313"]: return (-1/4+(1/3)*xw)
        elif pid in ["423","-423","443"]: return (1/4-(2/3)*xw)
    
    def GF(self):
        return 1.1663788*10**(-5)
        
    def set_generators(self,generators_light, generators_heavy):
        self.generators = {
            211: generators_light,
            321: generators_light,
            310: generators_light,
            130: generators_light,
            411: generators_heavy,
            421: generators_heavy,
            431: generators_heavy,
            511: generators_heavy,
            521: generators_heavy,
            531: generators_heavy,
            541: generators_heavy,
            15:  generators_heavy,
        }

    ###############################
    #  2-body decays
    ###############################
    # Branching fraction
    #pid0 is parent meson, pid1 is daughter meson
    def get_2body_br(self,pid0,pid1):
        pid0=str(pid0)
        pid1=str(pid1)
        #read constant
        mH, mLep, tauH = self.masses(pid0), self.masses(pid1), self.tau(pid0)
        vH, fH = self.VH(pid0), self.fH(pid0)
        SecToGev=1./(6.582122*pow(10.,-25.))
        tauH=tauH*SecToGev
        GF=1.166378*10**(-5) #GeV^(-2)
        #calculate br
        prefactor=(tauH*GF**2*fH**2*vH**2)/(8*np.pi)
        prefactor*=self.vcoupling[str(abs(int(pid1)))]**2
        br=str(prefactor)+"*coupling**2*mass**2*"+str(mH)+"*(1.-(mass/"+str(mH)+")**2 + 2.*("+str(mLep)+"/"+str(mH)+")**2 + ("+str(mLep)+"/mass)**2*(1.-("+str(mLep)+"/"+str(mH)+")**2)) * np.sqrt((1.+(mass/"+str(mH)+")**2 - ("+str(mLep)+"/"+str(mH)+")**2)**2-4.*(mass/"+str(mH)+")**2)"
        return br

    def get_2body_br_tau(self,pid0,pid1):
        pid0 = str(pid0)
        pid1 = str(pid1)
        if pid1 in ['213','-213','323','-323']:
            if pid1 in ['213','-213']: grho = 0.102
            if pid1 in ['323', '-323']: grho = 0.217*self.masses(pid1)
            VH, tautau = self.VH(pid1), self.tau(pid0)
            Mtau, Mrho=self.masses(pid0), self.masses(pid1)
            SecToGev=1./(6.582122*pow(10.,-25.))
            tautau=tautau*SecToGev
            GF=1.166378*10**(-5) #GeV^(-2)
            prefactor=(tautau*grho**2*GF**2*VH**2*Mtau**3/(8*np.pi*Mrho**2))
            prefactor*=(self.vcoupling[str(abs(int(pid0)))]**2)
            br=f"{prefactor}*coupling**2*((1-(mass**2/{Mtau}**2))**2+({Mrho}**2/{Mtau}**2)*(1+((mass**2-2*{Mrho}**2)/{Mtau}**2)))*np.sqrt((1-(({Mrho}-mass)**2/{Mtau}**2))*(1-(({Mrho}+mass)**2/{Mtau}**2)))"
        #for daughter pseudoscalars
        else:
            SecToGev=1./(6.582122*pow(10.,-25.))
            tautau=self.tau(pid0)
            tautau=tautau*SecToGev
            GF=1.166378*10**(-5) #GeV^(-2)
            VH=self.VH(pid1)
            fH=self.fH(pid1)
            Mtau=self.masses(pid0)
            M1=self.masses(pid1)
            prefactor=(tautau*GF**2*VH**2*fH**2*Mtau**3/(16*np.pi))
            prefactor*=(self.vcoupling[str(abs(int(pid0)))]**2)
            br=f"{prefactor}*coupling**2*((1-(mass**2/{Mtau}**2))**2-({M1}**2/{Mtau}**2)*(1+(mass**2/{Mtau}**2)))*np.sqrt((1-(({M1}-mass)**2/{Mtau}**2)*(1-(({M1}+mass)**2/{Mtau}**2))))"
        return (br)

    ###############################
    #  3-body decays
    ###############################

    #VHH in 3-body decays - CKM matrix elements
    def VHHp(self,pid0,pid1):     
        Vud=0.97373
        Vus=0.2243
        Vub=3.82E-3
        Vcd=0.221
        Vcs=0.975
        Vcb=40.8E-3
        Vtd=8.6E-3
        Vts=41.5E-3
        Vtb=1.014
        V21=Vud
        V23=Vus
        V25=Vub
        V41=Vcd
        V43=Vcs
        V45=Vcb

        if   pid0 in ["2","-2","1","-1"    ] and pid1 in ["1","-1","2","-2" ]: return 0.97373 #Vud
        if   pid0 in ["2","-2","3","-3"] and pid1 in ["3","-3","2","-2"    ]: return 0.2243 #Vus
        if   pid0 in ["4","-4","1","-1"    ] and pid1 in ["1","-1","4","-4"   ]: return 0.221 #Vcd
        if   pid0 in ["4","-4","3","-3"    ] and pid1 in ["3","-3" ,"4","-4"   ]: return 0.975 #Vcs
        if   pid0 in ["4","-4","5","-5"    ] and pid1 in ["4","-4","5","-5"    ]: return 40.8E-3 #Vcb
        if   pid0 in ["2","-2","5","-5"    ] and pid1 in ["2","-2","5","-5"    ]: return 3.82E-3 #Vub
        if   pid0 in ["6","-6","1","-1"    ] and pid1 in ["6","-6","1","-1"    ]: return 8.6E-3 #Vtd
        if   pid0 in ["6","-6","3","-3"    ] and pid1 in ["6","-6","3","-3"    ]: return 41.5E-3 #Vts
        if   pid0 in ["6","-6","5","-5"    ] and pid1 in ["6","-6","5","-5"    ]: return 1.014 #Vtb

        elif pid0 in ['411','-411'] and pid1 in ['311','-311']: return 0.975
        elif pid0 in ['421','-421'] and pid1 in ['321','-321']: return 0.975
        elif pid0 in ['521','-521'] and pid1 in ['421','-421']: return 40.8E-3
        elif pid0 in ['511','-511'] and pid1 in ['411','-411']: return 40.8E-3
        elif pid0 in ['531','-531'] and pid1 in ['431','-431']: return 40.8E-3
        elif pid0 in ['541','-541'] and pid1 in ['511','-511']: return 0.221
        elif pid0 in ['541','-541'] and pid1 in ['531','-531']: return 0.975
        elif pid0 in ['421','-421'] and pid1 in ['323','-323']: return 0.975
        elif pid0 in ['521','-521'] and pid1 in ['423','-423']: return 40.8E-3
        elif pid0 in ['511','-511'] and pid1 in ['413','-413']: return 40.8E-3
        elif pid0 in ['531','-531'] and pid1 in ['433','-433']: return 40.8E-3
        elif pid0 in ['541','-541'] and pid1 in ['513','-513']: return 0.221
        elif pid0 in ['541','-541'] and pid1 in ['533','-533']: return 0.975
        elif pid0 in ['130','310' ,"-130","-310"] and pid1 in ['211','-211']: return 0.2243

        #new channels pseudo
        elif pid0 in ['321','-321'] and pid1 in ['111','-111']: return V23
        elif pid0 in ['431','-431'] and pid1 in ['221','-221']: return V43
        elif pid0 in ['431','-431'] and pid1 in ['331','-331']: return V43
        elif pid0 in ['521','-521'] and pid1 in ['111','-111',"221","-221","331","-331"]: return V25 
        elif pid0 in ['541','-541'] and pid1 in ['421','-421']: return V25
        elif pid0 in ['541','-541'] and pid1 in ['441','-441']: return V45
        elif pid0 in ['421','-421'] and pid1 in ['211','-211']: return V41
        elif pid0 in ['411','-411'] and pid1 in ['111','-111',"221","-221","331","-331"]: return V41 
        elif pid0 in ['431','-431'] and pid1 in ['311','-311']: return V41
        elif pid0 in ['511','-511'] and pid1 in ['211','-211']: return V25
        elif pid0 in ['531','-531'] and pid1 in ['321','-321']: return V25

        #new channels vector
        elif pid0 in ['521','-521'] and pid1 in ['113','-113',"223","-223"]: return V25
        elif pid0 in ['541','-541'] and pid1 in ['443','-443']: return V45
        elif pid0 in ['421','-421'] and pid1 in ['213','-213']: return V41
        elif pid0 in ['411','-411'] and pid1 in ['113','-113',"223","-223"]: return V41 
        elif pid0 in ['411','-411'] and pid1 in ['313','-313']: return V43
        elif pid0 in ['431','-431'] and pid1 in ['313','-313']: return V41
        elif pid0 in ['431','-431'] and pid1 in ['333','-333']: return V43
        elif pid0 in ['511','-511'] and pid1 in ['213','-213']: return V25
        elif pid0 in ['531','-531'] and pid1 in ['323','-323']: return V25
        elif pid0 in ['541','-541'] and pid1 in ['423','-423']: return V25

        #Baryons
        elif pid0 in ["4122","-4122","4132","-4132"] and pid1 in ["3122", "-3122", "3312", "-3312"]: return V43  #Vcs
        elif pid0 in ["5122", "-5122", "5232","-5232", "5332", "-5332"] and pid1 in ["4122", "-4122", "4232", "-4232", "4332", "-4332"]: return V45   #Vcb


    #3-body differential branching fraction dBr/(dq^2dE) for decay of pseudoscalar to pseudoscalar meson
    #pid0 is parent meson pid1 is daughter meson pid2 is lepton pid3 is HNL
    def get_3body_dbr_pseudoscalar(self,pid0,pid1,pid2):
        pid0 = str(pid0)
        pid1 = str(pid1)
        pid2 = str(pid2)

        # read constant
        mH, mHp, mLep = self.masses(pid0), self.masses(pid1), self.masses(pid2)
        VHHp, tauH = self.VHHp(pid0,pid1), self.tau(pid0)
        SecToGev=1./(6.582122*pow(10.,-25.))
        tauH=tauH*SecToGev
        GF=1.166378*10**(-5) #GeV^(-2)

        #accounts for quark content of decay; only relevant for neutral mesons with several quark configurations
        cp=1

        #form factor parameters
        #D+ -> K0
        if pid0 in ["411","-411"] and pid1 in ["311","-311"]:
            pidV, pidS = "433", "431"
            f00, MV, MS = .747, self.masses(pidV), self.masses(pidS)      
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #D0 -> K+
        if pid0 in ["421","-421"] and pid1 in ["321","-321"]:
            pidV, pidS = "433", "431"
            f00, MV, MS = .747, self.masses(pidV), self.masses(pidS)      
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #Ds+ -> K0
        if pid0 in ["431","-431"] and pid1 in ["311","-311"]:
            pidV, pidS = "413", "411"
            f00, MV, MS = .747, self.masses(pidV), self.masses(pidS)     
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #B0 -> D+
        if pid0 in ["511","-511"] and pid1 in ["411","-411"]:
            pidV, pidS = "543", "541"
            f00, MV, MS = 0.66, self.mass_543, self.masses(pidS)      
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #B+ -> D0
        if pid0 in ["521","-521"] and pid1 in ["421", "-421"]:
            pidV, pidS = "543", "541"
            f00, MV, MS = 0.66, self.mass_543, self.masses(pidS)       
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #B0s -> Ds-
        if pid0 in ["531","-531"] and pid1 in ["431","-431"]:
            pidV, pidS = "543", "541"
            f00, MV, MS = -0.65, self.mass_543, self.masses(pidS)          
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)" 
        #Bc+ -> B0
        if pid0 in ["541","-541"] and pid1 in ["511","-511"]:
            pidV, pidS = "413", "411"
            f00, MV, MS = -0.58, self.masses(pidV), self.masses(pidS)     
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #Bc+ -> B0s
        if pid0 in ["541","-541"] and pid1 in ["531","-531"]:
            pidV, pidS = "433", "431"
            f00, MV, MS = -0.61, self.masses(pidV), self.masses(pidS)      
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #Bc+ -> D0
        if pid0 in ["541","-541"] and pid1 in ["421","-421"]:
            pidV, pidS = "523", "521"
            f00, MV, MS = 0.69, self.masses(pidV), self.masses(pidS)     
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #K0L,K0s -> pi+
        if pid0 in ["130","310"] and pid1 in ["211","-211"]:
            cp=(1/2)
            f00, MV, MS = .9636, .878, 1.252        
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #K+ to pi0; assuming same form factors as pi+
        if pid0 in ["321","-321"] and pid1 in ["111","-111"]:
            cp=(1/2)
            pidV, pidS = "313", "311"
            f00, MV, MS = 0.970, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #D^+_s \to \eta; assuming same form factors as pi+
        if pid0 in ["431","-431"] and pid1 in ["221","-221"]:
            pidV, pidS = "433", "431"
            f00, MV, MS = 0.495, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #D^+_s \to \eta'; assuming same form factors as pi+
        if pid0 in ["431","-431"] and pid1 in ["331","-331"]:
            pidV, pidS = "433", "431"
            f00, MV, MS = 0.557, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #B^+ \to \pi^0; assuming same form factors as pi+
        if pid0 in [ "521","-521"] and pid1 in ["111","-111"]:
            cp=1/2
            pidV, pidS = "513","511"
            f00, MV, MS = 0.29, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #B_c^+ \to \eta_c
        if pid0 in ["541","-541"] and pid1 in ["441","-441"]:
            pidV, pidS = "543","541"
            f00, MV, MS = 0.76, self.mass_543, self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #D^0 \to \pi^+
        if pid0 in ["421","-421"] and pid1 in ["211","-211"]:
            pidV, pidS = "413", "411"
            f00, MV, MS = 0.69, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #D^- \to \pi^0 
        if pid0 in ["411","-411"] and pid1 in ["111","-111"]: 
            cp=(1/2)
            pidV, pidS = "413", "411"
            f00, MV, MS = 0.69, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #D^- \to \eta, \eta', used pi+ form factors and corrected by cp
        if pid0 in ["411","-411"] and pid1 in ["221","-221","331","-331"]:
            theta=-11.5*np.pi/180
            #eta; corrects for the fact that eta and etap are rotations of eta1 and eta8
            if pid1 in ["221","-221"]:
                cp = ((np.cos(theta)/np.sqrt(6))-(np.sin(theta)/np.sqrt(3)))**2
            #etap
            if pid1 in ["331","-331"]:
                cp = ((np.sin(theta)/np.sqrt(6)) + (np.cos(theta)/np.sqrt(3)))**2
            pidV, pidS = "413", "411"
            f00, MV, MS = 0.69, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #D_s^+ \to K^0
        if pid0 in ["431","-431"] and pid1 in ["311","-311"]:
            pidV, pidS = "413", "411"
            f00, MV, MS = 0.72, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #B^0 \to \pi^+
        if pid0 in ["511","-511"] and pid1 in ["211","-211"]:
            pidV, pidS = "523", "521"
            f00, MV, MS = 0.29, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #B- -> eta, etap
        if pid0 in ["521","-521"] and pid1 in ["221","-221","331","-331"]:
            theta=-11.5*np.pi/180
            #eta; corrects for the fact that eta and etap are rotations of eta1 and eta8
            if pid1 in ["221","-221"]:
                cp = ((np.cos(theta)/np.sqrt(6))-(np.sin(theta)/np.sqrt(3)))**2
            #etap
            if pid1 in ["331","-331"]:
                cp = ((np.sin(theta)/np.sqrt(6)) + (np.cos(theta)/np.sqrt(3)))**2
            pidV, pidS = "523", "521"
            f00, MV, MS = 0.29, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #B_s^0 \to K^+
        if pid0 in ["531","-531"] and pid1 in ["321","-321"]:
            pidV, pidS = "523", "521"
            f00, MV, MS = 0.31, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        # prefactor
        prefactor=(cp)*tauH*VHHp**2*GF**2/(64*np.pi**3*mH**2)
        prefactor*=self.vcoupling[str(abs(int(pid2)))]**2
        fm="("+f0+"-"+fp+")*("+str(mH)+"**2-"+str(mHp)+"**2)/q**2" 
        #putting all terms together
        term1="("+fm+")**2*(q**2*(mass**2+"+str(mLep)+"**2)-(mass**2-"+str(mLep)+"**2)**2)"
        term2=f"2*("+fp+")*("+fm+")*mass**2*(2*"+str(mH)+"**2-2*"+str(mHp)+"**2-4*energy*"+str(mH)+"-"+str(mLep)+"**2+mass**2+q**2)"
        term3=f"(2*("+fp+")*("+fm+")*"+str(mLep)+"**2*(4*energy*"+str(mH)+"+ "+str(mLep)+"**2-mass**2-q**2))"
        term4=f"("+fp+")**2*(4*energy*"+str(mH)+"+"+str(mLep)+"**2-mass**2-q**2)*(2*"+str(mH)+"**2-2*"+str(mHp)+"**2-4*energy*"+str(mH)+"-"+str(mLep)+"**2+mass**2+q**2)"
        term5=f"-("+fp+")**2*(2*"+str(mH)+"**2+2*"+str(mHp)+"**2-q**2)*(q**2-mass**2-"+str(mLep)+"**2)"
        bra=str(prefactor)  + "* coupling**2 *(" + term1   + "+(" + term2  + "+" + term3 + ")+("  + term4   + "+" + term5 + "))"
        return(bra)

    #3-body differential branching fraction dBr/(dq^2dE) for decay of pseudoscalar to vector meson
    #pid0 is parent meson pid1 is daughter meson pid2 is lepton pid3 is HNL
    def get_3body_dbr_vector(self,pid0,pid1,pid2):
        pid0 = str(pid0)
        pid1 = str(pid1)
        pid2 = str(pid2)
        
        tauH=self.tau(pid0)
        SecToGev=1./(6.58*pow(10.,-25.))
        tauH=tauH*SecToGev
        GF=1.1663787*10**(-5)
        VHV=self.VHHp(pid0,pid1)
        m0=self.masses(pid0)
        m1=self.masses(pid1)
        m2=self.masses(pid2)
        #accounts for quark content of decay; only relevant for neutral mesons with several quark configurations
        cv=1
        #'D^0 -> K*^- + e^+ + N' form factors
        if pid0 in ['-421','421'] and pid1 in ['323','-323']:
            pidp = "431"
            pidV ="433"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=.76; s1A0=.17; s2A0=0; V0=1.03; s1V=.27; s2V=0; A10=.66; s1A1=.3     
            s2A1=.2*0; A20=.49; s1A2=.67; s2A2=.16*0        
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"

        #'D^- -> K^{0*} + e^- + N' form factors
        if pid0 in ['-411','411'] and pid1 in ['313','-313']:
            pidp = "431"
            pidV = "433"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=.76; s1A0=.17; s2A0=0; V0=1.03; s1V=.27; s2V=0; A10=.66; s1A1=.3    
            s2A1=.2*0; A20=.49; s1A2=.67; s2A2=.16*0        
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"

        #'B^+ -> \bar{D}*^0 + e^+ + N'
        if (pid0 in ['521','-521'] and pid1 in ['423','-423']):
            pidp = "541"
            pidV = "543"
            Mp = self.masses(pidp); MV = self.mass_543
            A00=0.69; s1A0=0.58; s2A0=0; V0=0.76; s1V=0.57; s2V=0; A10=0.66; s1A1=0.78     
            s2A1=0; A20=0.62; s1A2=1.04; s2A2=0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #'B^0 -> D*^- + e^+ + N' form factors
        if (pid0 in ['511','-511'] and pid1 in ['413','-413']):
            pidp = "541"
            pidV = "543"
            Mp = self.masses(pidp); MV = self.mass_543
            A00=0.69; s1A0=0.58; s2A0=0; V0=0.76; s1V=0.57; s2V=0; A10=0.66; s1A1=0.78      
            s2A1=0; A20=0.62; s1A2=1.04; s2A2=0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #'B^0_s -> D^*_s^- + e^+ + N' form factors
        if pid0 in ['531','-531'] and pid1 in ['433','-433']:
            pidp = "541"
            pidV = "543"
            Mp = self.masses(pidp); MV = self.mass_543
            A00=0.67; s1A0=0.35; s2A0=0; V0=0.95; s1V=0.372         
            s2V=0; A10=0.70; s1A1=0.463; s2A1=0; A20=0.75; s1A2=1.04; s2A2=0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #'B^+_c -> B*^0 + e^+ + N' form factors
        if pid0 in ['541','-541'] and pid1 in ['513','-513']:
            A00=-.27; mfitA0=1.86; deltaA0=.13; V0=3.27; mfitV=1.76; deltaV=-.052       
            A10=.6; mfitA1=3.44; deltaA1=-1.07; A20=10.8; mfitA2=1.73; deltaA2=-0.09
            A0=f"({A00}/(1-(q**2/{mfitA0}**2)-{deltaA0}*(q**2/{mfitA0}**2)**2))"
            V=f"({V0}/(1-(q**2/{mfitV}**2)-{deltaV}*(q**2/{mfitV}**2)**2))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-(q**2/{mfitA1}**2)-{deltaA1}*(q**2/{mfitA1}**2)**2))"
            A2=f"({A20}/(1-(q**2/{mfitA2}**2)-{deltaA2}*(q**2/{mfitA2}**2)**2))"
        #'B^+_c -> B^*_s^0+ e^+ + N' form factors
        if pid0 in ['541','-541'] and pid1 in ['533','-533']:
            A00=-.33; mfitA0=1.86; deltaA0=.13; V0=3.25; mfitV=1.76; deltaV=-.052      
            A10=.4; mfitA1=3.44; deltaA1=-1.07; A20=10.4; mfitA2=1.73; deltaA2=-0.09
            A0=f"({A00}/(1-(q**2/{mfitA0}**2)-{deltaA0}*(q**2/{mfitA0}**2)**2))"
            V=f"({V0}/(1-(q**2/{mfitV}**2)-{deltaV}*(q**2/{mfitV}**2)**2))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-(q**2/{mfitA1}**2)-{deltaA1}*(q**2/{mfitA1}**2)**2))"
            A2=f"({A20}/(1-(q**2/{mfitA2}**2)-{deltaA2}*(q**2/{mfitA2}**2)**2))"
        #new modes
        #B+ to rho^0 or omega
        if pid0 in ["521","-521"] and pid1 in ["113","-113","223","-223"]:       
            cv=(1/2)
            pidp = "521"
            pidV = "523"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.30; s1A0=0.54; s2A0=0; V0=0.31; s1V=0.59         
            s2V=0; A10=0.26; s1A1=0.73; s2A1=0.1; A20=0.29; s1A2=1.4; s2A2=0.5
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #Bc+ to J/psi
        if pid0 in ['541','-541'] and pid1 in ['443','-443']:
            A00=0.68; mfitA0=8.20; deltaA0=1.40; V0=0.96; mfitV=5.65; deltaV= 0.0013      
            A10=0.68; mfitA1=5.91; deltaA1=0.052; A20=-0.004; mfitA2=5.67; deltaA2=-0.004
            A0=f"({A00}/(1-(q**2/{mfitA0}**2)-{deltaA0}*(q**2/{mfitA0}**2)**2))"
            V=f"({V0}/(1-(q**2/{mfitV}**2)-{deltaV}*(q**2/{mfitV}**2)**2))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-(q**2/{mfitA1}**2)-{deltaA1}*(q**2/{mfitA1}**2)**2))"
            A2=f"({A20}/(1-(q**2/{mfitA2}**2)-{deltaA2}*(q**2/{mfitA2}**2)**2))"
        #\bar{D^0} to \rho^+
        if pid0 in ["421","-421"] and pid1 in ["213","-213"]:      
            pidp = "411"
            pidV = "413"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.66; s1A0=0.36; s2A0=0; V0=0.90; s1V=0.46         
            s2V=0; A10=0.59; s1A1=0.50; s2A1=0; A20=0.49; s1A2=0.89; s2A2=0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #D^- to \rho^0
        if pid0 in ["411","-411"] and pid1 in ["113","-113"]:      
            cv=(1/2)
            pidp = "411"
            pidV = "413"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.66; s1A0=0.36; s2A0=0; V0=0.90; s1V=0.46         
            s2V=0; A10=0.59; s1A1=0.50; s2A1=0; A20=0.49; s1A2=0.89; s2A2=0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #D^- to \omega
        if pid0 in ["411","-411"] and pid1 in ["223","-223"]:      
            cv=(1/2)
            pidp = "411"
            pidV = "413"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.66; s1A0=0.36; s2A0=0; V0=0.90; s1V=0.46         
            s2V=0; A10=0.59; s1A1=0.50; s2A1=0; A20=0.49; s1A2=0.89; s2A2=0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"

        #D_s^- \to \bar{K^{*0}}
        if pid0 in ["431","-431"] and pid1 in ["313","-313"]:      
            pidp = "411"
            pidV = "413"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.67; s1A0=0.2; s2A0=0; V0=1.04; s1V=0.24        
            s2V=0; A10=0.57; s1A1=0.29; s2A1=0.42; A20=0.42; s1A2=0.58; s2A2=0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #D_s^- \to \phi
        if pid0 in ["431","-431"] and pid1 in ["333","-333"]:       
            pidp = "431"
            pidV = "433"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.73; s1A0=0.10; s2A0=0; V0=1.10; s1V=0.26         
            s2V=0; A10=0.64; s1A1=0.29; s2A1=0; A20=0.47; s1A2=0.63; s2A2=0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #B^0 \to \rho^-
        if pid0 in ["511","-511"] and pid1 in ["213","-213"]:      
            pidp = "521"
            pidV = "523"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.30; s1A0=0.54; s2A0=0; V0=0.31; s1V=0.59         
            s2V=0; A10=0.26; s1A1=0.54; s2A1=0.1; A20=0.24; s1A2=1.40; s2A2=0.50
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #B^- \to \omega; used rho- form factors
        if pid0 in ["511","-511"] and pid1 in ["223","-223"]:      
            cv=(1/2)
            pidp = "511"
            pidV = "513"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.30; s1A0=0.54; s2A0=0; V0=0.31; s1V=0.59         
            s2V=0; A10=0.26; s1A1=0.54; s2A1=0.1; A20=0.24; s1A2=1.40; s2A2=0.50
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #B_s^0 \to K^*-
        if pid0 in ["531","-531"] and pid1 in ["323","-323"]:       
            pidp = "521"
            pidV = "523"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.37; s1A0=0.60; s2A0=0.16; V0=0.38; s1V=0.66         
            s2V=0.30; A10=0.29; s1A1=0.86; s2A1=0.6; A20=0.26; s1A2=1.32; s2A2=0.54
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #B_c^+ \to D^{*0} 
        if pid0 in ["541","-541"] and pid1 in ["423","-423"]:
            pidp = "521"
            pidV = "523"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.56; s1A0=0; s2A0=0; V0=0.98; s1V=0        #not sure if this is correct
            s2V=0; A10=0.64; s1A1=1; s2A1=0; A20=-1.17; s1A2=1; s2A2=0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))" 
        #form factors
        f1=f"({V}/({m0}+{m1}))"
        f2=f"(({m0}+{m1})*{A1})"
        f3=f"(-{A2}/({m0}+{m1}))"
        f4=f"(({m1}*(2*{A0}-{A1}-{A2})+{m0}*({A2}-{A1}))/q**2)"
        f5=f"({f3}+{f4})"
        #s1A0 is sigma_1(A0) etc.
        omegasqr=f"({m0}**2-{m1}**2+m3**2-{m2}**2-2*{m0}*energy)"
        Omegasqr=f"({m0}**2-{m1}**2-q**2)"
        prefactor=f"(({cv})*({tauH}*coupling**2*{VHV}**2*{GF}**2)/(32*np.pi**3*{m0}**2))*{self.vcoupling[str(abs(int(pid2)))]}**2"
        term1=f"({f2}**2/2)*(q**2-m3**2-{m2}**2+{omegasqr}*(({Omegasqr}-{omegasqr})/{m1}**2))"
        term2=f"({f5}**2/2)*(m3**2+{m2}**2)*(q**2-m3**2+{m2}**2)*(({Omegasqr}**2/(4*{m1}**2))-q**2)"
        term3=f"2*{f3}**2*{m1}**2*(({Omegasqr}**2/(4*{m1}**2))-q**2)*(m3**2+{m2}**2-q**2+{omegasqr}*(({Omegasqr}-{omegasqr})/{m1}**2))"
        term4=f"2*{f3}*{f5}*(m3**2*{omegasqr}+({Omegasqr}-{omegasqr})*{m2}**2)*(({Omegasqr}**2/(4*{m1}**2))-q**2)"
        term5=f"2*{f1}*{f2}*(q**2*(2*{omegasqr}-{Omegasqr})+{Omegasqr}*(m3**2-{m2}**2))"
        term6=f"({f2}*{f5}/2)*({omegasqr}*({Omegasqr}/{m1}**2)*(m3**2-{m2}**2)+({Omegasqr}**2/{m1}**2)*{m2}**2+2*(m3**2-{m2}**2)**2-2*q**2*(m3**2+{m2}**2))"
        term7=f"{f2}*{f3}*({Omegasqr}*{omegasqr}*(({Omegasqr}-{omegasqr})/{m1}**2)+2*{omegasqr}*({m2}**2-m3**2)+{Omegasqr}*(m3**2-{m2}**2-q**2))"
        term8=f"{f1}**2*({Omegasqr}**2*(q**2-m3**2+{m2}**2)-2*{m1}**2*(q**4-(m3**2-{m2}**2)**2)+2*{omegasqr}*{Omegasqr}*(m3**2-q**2-{m2}**2)+2*{omegasqr}**2*q**2)"
        bra=str(prefactor) + "*(" + term1 + "+" + term2 + "+" + term3 + "+" + term4 + "+" + term5 + "+" + term6 + "+" + term7 + "+" + term8 + ")"
        return(bra)

    #pid0 is tau, pid1 is produced lepton and pid2 is the neutrino
    #3-body differential branching fraction dBr/(dE) for 3-body leptonic decay of tau lepton
    def get_3body_dbr_tau(self,pid0,pid1,pid2):
        pid0 = str(pid0)
        pid1 = str(pid1)
        pid2 = str(pid2)

        m0=self.masses(pid0)
        m1=self.masses(pid1)
        if pid2=='16' or pid2=='-16':
            SecToGev=1./(6.582122*pow(10.,-25.))
            tautau=self.tau(pid0)*SecToGev
            GF=1.166378*10**(-5) #GeV^(-2)
            prefactor=f"({tautau}*{GF}**2*coupling**2*{m0}**2*energy/(2*np.pi**3))*{self.vcoupling[str(abs(int(pid1)))]}**2"
            dbr=f"{prefactor}*(1+((mass**2-{m1}**2)/{m0}**2)-2*(energy/{m0}))*(1-({m1}**2/({m0}**2+mass**2-2*energy*{m0})))*np.sqrt(energy**2-mass**2)"
        else:
            SecToGev=1./(6.582122*pow(10.,-25.))
            tautau=self.tau(pid0)*SecToGev
            GF=1.166378*10**(-5) #GeV^(-2)
            prefactor=f"({tautau}*{GF}**2*coupling**2*{m0}**2/(4*np.pi**3))*{self.vcoupling[str(abs(int(pid0)))]}**2"
            dbr=f"{prefactor}*(1-{m1}**2/({m0}**2+mass**2-2*energy*{m0}))**2*np.sqrt(energy**2-mass**2)*(({m0}-energy)*(1-(mass**2+{m1}**2)/{m0}**2)-(1-{m1}**2/({m0}**2+mass**2-2*energy*{m0}))*(({m0}-energy)**2/{m0}+((energy**2-mass**2)/(3*{m0}))))"
        return(dbr)
    
    def get_3body_dbr_baryon(self,pid0,pid1,pid2):
        pid0 = str(pid0)
        pid1 = str(pid1)
        pid2 = str(pid2)

        GF = 1.166378*10**(-5)
        m0 = self.masses(pid0)
        m1 = self.masses(pid1)
        m2 = self.masses(pid2)
        Vckm = self.VHHp(pid0,pid1)
        #parent baryon lifetime
        tauB = self.tau(pid0)   #seconds
        SecToGev=1./(6.582122*pow(10.,-25.))
        tauB = tauB*SecToGev    #1/GeV
        Ulx = self.vcoupling[str(abs(int(pid2)))]**2

        #\Lambda_c^+ \to \Lambda^0
        #4122 to 3122
        if str(abs(int(pid0))) == '4122':
            #Vcs = 0.975
            #tauB = 201.5*10**(-15)
            #m0 = 2.28646
            #m1 = 1.115683
            #form factor masses
            MV = 2.11
            MA = 2.54
            #form factor coefficients
            f10 = 0.29
            f20 = 0.14
            f30 = 0.03
            g10 = 0.38
            g20 = 0.03
            g30 = 0.19

        #\Lambda_b^0 \to \Lambda_c^+
        #5122 to 4122
        if str(abs(int(pid0))) == '5122':
            #Vcs = 41*10**(-3)
            #tauB = 1.471*10**(-12)
            #m0 = 5.6196
            #m1 = 2.28646
            #form factor masses
            MV = 6.34
            MA = 6.73
            #form factor coefficients
            f10 = 0.53
            f20 = 0.12
            f30 = 0.02
            g10 = 0.58
            g20 = 0.02
            g30 = 0.13

        #\Xi^0 \to \Xi^-
        #4132 to 3312
        if str(abs(int(pid0))) == '4132':
            #c->s for ckm and pole masses
            #Vcs = 0.975
            #tauB = 1.53*10**(-13)
            #m0 = 2.47090
            #m1 = 1.32171
            #form factor masses
            MV = 2.11
            MA = 2.54
            #form factor coefficients
            f10 = 0.31
            f20 = 0.19
            f30 = 0.04
            g10 = 0.39
            g20 = 0.06
            g30 = 0.24

        #\Xi^0_b \to \Xi_c^+
        #5232 to 4232
        if str(abs(int(pid0))) == '5232':
            #b->c for ckm and pole masses
            #Vcs = 40.8*10**(-3)
            #tauB = 1.48*10**(-12)
            #m0 = 5.7919
            #m1 = 2.46794
            #form factor masses
            MV = 6.34
            MA = 6.73
            #form factor coefficients
            f10 = 0.54
            f20 = 0.14
            f30 = 0.02
            g10 = 0.58
            g20 = 0.03
            g30 = 0.16

        #\Omega_b^- \to \Omega_c^0
        #5332 to 4332
        if str(abs(int(pid0))) == '5332':
            #b->c for ckm and pole masses
            #Vcs = 40.8*10**(-3)
            #tauB = 1.64*10**(-12)
            #m0 = 6.0461
            #m1 = 2.6952
            #form factor masses
            MV = 6.34
            MA = 6.73
            #form factor coefficients
            f10 = 0.72
            f20 = -0.68
            f30 = 0.36
            g10 = -0.20
            g20 = -0.01
            g30 = -0.06


        kNkl = f"(q2 - {m2}**2 - mass**2)*(1/2)"
        qkl = f"(-({m2}**2 + " + str(kNkl) + "))"
        qkN = "(-(" + str(kNkl) + f"+mass**2))"
        #m12sq = f"({m0}**2 + mass**2 - 2*{m0}*energy)"
        #PBpkl = "(" + str(m12sq) + f"-{m1}**2-{m2}**2)*(1/2)"
        PBpkl = "(m12sq" + f"-{m1}**2-{m2}**2)*(1/2)"
        energy = f"(-(1/(2*{m0}))*(m12sq-{m0}**2-mass**2))"
        PBpkN = f"({m0}*{energy}-" + str(kNkl) + f"-mass**2)"
        PBpq = "(-(" + str(PBpkl) + "+" + str(PBpkN) + "))"



        #form factors
        f1 = f"({f10}/(1-q2/{MV}**2)**2)"
        f2 = f"({f20}/(1-q2/{MV}**2)**2)"
        f3 = f"({f30}/(1-q2/{MV}**2)**2)"
        g1 = f"({g10}/(1-q2/{MA}**2)**2)"
        g2 = f"({g20}/(1-q2/{MA}**2)**2)"
        g3 = f"({g30}/(1-q2/{MA}**2)**2)"

        #spin averaged amplitude squared
        Msq1 = f"{tauB}*8*{GF}**2*{Vckm}**2*{Ulx}**2*coupling**2*((" + str(f1) + "**2+" + str(g1) + "**2)*(4*" + str(PBpkN) + "*" + str(PBpkl) + "-2*" + str(PBpkN) + "*" + str(qkl) + "-2*" + str(PBpkl) + "*" + str(qkN) + ")"
        Msq2 = f"-2*(" + str(f1) + "**2-" + str(g1) + f"**2)*{m1}*{m0}*" + str(kNkl)
        Msq3 = f"+(2*{m1}/{m0})*(" + str(f1) + "*" + str(f2) + "+" + str(g1) + "*" + str(g2) + ")*(" + str(kNkl) + "*(" + str(PBpq) + "-q2)+" + str(qkN) + "*(" + str(PBpkl) + "-" + str(qkl) + ")+" + str(qkl) + "*(" + str(PBpkN) + "-" + str(qkN) + "))"
        Msq4 = f"-2*(" + str(f1) + "*" + str(f2) + "-" + str(g1) + "*" + str(g2) + ")*(" + str(PBpkl) + "*" + str(qkN) + "+" + str(PBpkN) + "*" + str(qkl) + "+" + str(PBpq) + "*" + str(kNkl) + ")"
        Msq5 = f"-((" + str(f2) + "**2+" + str(g2) + f"**2)/{m0}**2)*(4*" + str(PBpkN) + "*" + str(PBpkl) + "*q2-4*" + str(PBpq) + "*(" + str(PBpkl) + "*" + str(qkN) + "+" + str(PBpkN) + "*" + str(qkl) + ")"
        Msq6 = f"-2*" + str(kNkl) + "*(2*(" + str(PBpq) + f")**2-q2*{m1}**2-" + str(PBpq) + "*q2)+2*" + str(qkN) + "*" + str(qkl) + "*(" + str(PBpq) + f"+{m1}**2)"
        Msq7 = f"-3*{m1}**2*q2*" + str(kNkl) + "+4*(" + str(PBpq) + ")**2*" + str(kNkl) + "-" + str(PBpq) + "*q2*" + str(kNkl) + ")"
        Msq8 = f"-({m1}/{m0})*(" + str(f2) + "**2-" + str(g2) + "**2)*(q2*" + str(kNkl) + "+2*" + str(qkN) + "*" + str(qkl) + ")"
        Msq9 = f"+({m1}/{m0})*(" + str(f1) + "*" + str(f3) + "+" + str(g1) + "*" + str(g3) + f")*(2*" + str(qkN) + "*(" + str(PBpkl) + "-" + str(qkl) + ")+2*" + str(qkl) + "*(" + str(PBpkN) + "-" + str(qkN) + ")-2*" + str(kNkl) + "*(" + str(PBpq) + "-q2))"
        Msq10 = f"+(" + str(f1) + "*" + str(f3) + "-" + str(g1) + "*" + str(g3) + ")*(2*" + str(PBpkl) + "*" + str(qkN) + "+2*" + str(PBpkN) + "*" + str(qkl) + "-2*" + str(PBpq) + "*" + str(kNkl) + ")"
        Msq11 = f"+2*((" + str(f2) + "*" + str(f3) + "+" + str(g2) + "*" + str(g3) + f")/{m0}**2)*(q2*(" + str(PBpkl) + "*" + str(qkN) + "+" + str(PBpkN) + "*" + str(qkl) + ")-2*" + str(PBpq) + "*" + str(qkl) + "*" + str(qkN) + ")"
        Msq12 = f"-(" + str(f3) + f"**2/{m0}**2)*(2*" + str(qkN) + "*" + str(qkl) + "-q2*" + str(kNkl) + ")*(" + str(PBpq) + f"-{m1}*({m0}+{m1}))"
        Msq13 = f"-(" + str(g3) + f"**2/{m0}**2)*(2*" + str(qkN) + "*" + str(qkl) + "-q2*" + str(kNkl) + ")*(" + str(PBpq) + f"+{m1}*({m0}-{m1}))"
        Msq14 = f"+4*(" + str(PBpkl) + "*" + str(qkN) + "-" + str(PBpkN) + "*" + str(qkl) + ")"
        Msq15 = f"*(" + str(g1) + "*" + str(f1) + "+(" + str(g2) + "*" + str(f2) + f"/{m0}**2)*(2*" + str(PBpq) + "-q2)-" + str(f1) + "*" + str(g2) + f"*(1-({m1}/{m0}))+" + str(f2) + "*" + str(g1) + f"*(1+({m1}/{m0}))))"

        Msq = Msq1+Msq2+Msq3+Msq4+Msq5+Msq6+Msq7+Msq8+Msq9+Msq10+Msq11+Msq12+Msq13+Msq14+Msq15
 
        dbr = f"(1/((2*np.pi)**3*32*{m0}**3))*" + Msq
        return(dbr)

    #finds the total br for a given parent particle and returns masses, br_tot
    #key is the symbol for the parent particle
    def parent_br(self, coupling, masses, key, tf_2body, tf_3body):
        br_tot = np.array([0.0 for n in range(len(masses))])
        #2 body grouping
        if key in self.get_channels_2body()['parent'].keys() and tf_2body==True:
            #add all brs for like parent mesons
            for n in range(len(self.get_channels_2body()['parent'][key])):
                _, pid0, pid1, br, _ = self.get_channels_2body()['parent'][key][n].values()
                #print(br)
                print(pid0)
                #mass=.3
                #print(eval(br))
                for m in range(len(masses)):
                    mass=masses[m]
                    if mass<=self.masses(pid0)-self.masses(pid1):
                        print(mass,eval(br))
                #br_arr = np.array([eval(br) if mass<self.masses(pid0)-self.masses(pid1) else 0 for mass in masses])
                br_arr = np.array([eval(br) for mass in masses])
                br_tot += br_arr

        #3 body grouping
        if key in self.get_channels_3body()['parent'].keys() and tf_3body==True:
            #add all brs for like parent mesons
            for n in range(len(self.get_channels_3body()['parent'][key])):
                label, pid0, pid1, pid2 , br, integration, latex_label = self.get_channels_3body()['parent'][key][n].values()
                m0, m1, m2 = self.masses(pid0), self.masses(pid1), self.masses(pid2)
                br_arr = np.array([self.integrate_3body_br(
                    br, mass, m0, m1, m2, integration=integration) if mass<m0-m1-m2 else 0 for mass in masses])
                br_tot += br_arr
        return masses, br_tot


    #input path to given folder and it removes all files in that folder
    def remove_files_from_folder(self,path):
        files = glob.glob(path)
        for f in files:
            os.remove(f)


    def get_br_and_ctau(self,mpts = np.logspace(-3,1, 401),coupling =1):
        """
        
        Generate Decay Data and save to Decay Data directory
        
        """
        
#         #define coupling tuple
#         coupling = (self.vcoupling['11'],self.vcoupling['13'],self.vcoupling['15'])
        
#         #create HNL_Decay Object
#         Decay = HNL_Decay(couplings = coupling)
        
        #Generate ctau (stored in Decay.ctau)
        self.gen_ctau(mpts,coupling)
        
        #Generate branching ratios (stored in Decay.model_brs)
        self.gen_brs()
        
        #Write ctau, branching ratios, and decay widths to the Decay Width directory
        self.save_data(True,True)
        
        
    def set_brs(self):
        """
        
        Create list of decay modes, final states, and the location of thhe respective branching fraction to be passed to FORESEE
        """
        #define coupling tuple
        
        
        modes =  [] 
        filenames = []
        #iterate over decay modes
        for channel in self.modes_active.keys():
            if channel not in ['nuqq','lud','nuhad','lhad']:
                for mode in self.modes_active[channel]: 
                    
                    modes.append(mode)

                    csv_path = f"model/br/"

                    for p in mode: csv_path += f"{p}-"
                    
                
                
                    
                    filenames.append(csv_path[:-1]+".csv")
        
        finalstates  = []
        
        for mode in modes: 
            finalstate = [pid(i) for i in mode]
            
            finalstates.append(finalstate)
            
        
        return modes,finalstates,filenames
            
    ###############################
    #  return list of productions
    ###############################

    def get_channels_2body(self,):
        lep = {"11":r"e","13":r"\mu","15":r"\tau"}

        dic_2body_mode = {"2body_pseudo": [], "2body_tau": []}
        dic_2body_parent = {r"$\pi$": [], r"$K$": [], r"$D$": [], r"$D_s$": [], r"$B$": [], r"$B_c$": [], r"$\tau$": []}
        
        #last sign is for the lepton
        channels_2body = [
            [r'$D^+ \to l^+ + N$'    , '411', '-'],
            [r'$D^- \to l^- + N$'    ,'-411', '' ],
            [r'$D_s^+ \to l^+ + N$'  , '431', '-'],
            [r'$D_s^- \to l^- + N$'  ,'-431', '' ],
            [r'$B^+ \to + l^+ + N$'  , '521', '-'],
            [r'$B^- \to + l^- + N$'  ,'-521', '' ],
            [r'$B_c^+ \to + l^+ + N$', '541', '-'],
            [r'$B_c^- \to + l^- + N$','-541', '' ],
            [r'$\pi^+ \to + l^+ + N$' , '211', '-'],
            [r'$\pi^- \to + l^- + N$' ,'-211', '' ],
            [r'$K^+ \to + l^+ + N$'  , '321', '-'],
            [r'$K^- \to + l^- + N$'  ,'-321', '' ],
        ]
        
        #last sign is for the pion
        channels_2body_tau = [
            [r'$\tau^- \to \pi^- + N$' , '15','211', '-' ],
            [r'$\tau^+ \to \pi^+ + N$' ,'-15','211', ''  ],
            [r'$\tau^- \to K^- + N$'   , '15','321', '-' ],
            [r'$\tau^+ \to K^+ + N$'   ,'-15','321', ''  ],
            [r'$\tau^- \to \rho^- + N$', '15','213', '-' ],
            [r'$\tau^+ \to \rho^+ + N$','-15','213', ''  ],
            [r'$\tau^- \to K^{*-} + N$', '15','323', '-' ],
            [r'$\tau^+ \to K^{*+} + N$','-15','323', ''  ]
        ]
    
        output=[]
        for description, pid_had, sign_lep in channels_2body:
            for pid_lep in ["11","13","15"]:
                if self.vcoupling[pid_lep] <1e-9: continue
                label= "2body_" + pid_had + "_" + sign_lep+pid_lep
                br = f"hnl.get_2body_br({str(pid_had)}, {str(sign_lep+pid_lep)})"
                dic = {'label': label, 'pid0': pid_had, 'pid1':sign_lep+pid_lep, 'br': br , 'description': description.replace("l",lep[pid_lep])}
                dic_2body_mode["2body_pseudo"].append(dic)
                if str(abs(int(pid_had))) == "211":
                    dic_2body_parent[r"$\pi$"].append(dic)
                if str(abs(int(pid_had))) == "321":
                    dic_2body_parent[r"$K$"].append(dic)
                if str(abs(int(pid_had))) in ["411"]:
                    dic_2body_parent[r"$D$"].append(dic)
                if str(abs(int(pid_had))) in ["431"]:
                    dic_2body_parent[r"$D_s$"].append(dic)
                if str(abs(int(pid_had))) in ["521"]:
                    dic_2body_parent[r"$B$"].append(dic)
                if str(abs(int(pid_had))) in ["541"]:
                    dic_2body_parent[r"$B_c$"].append(dic)


        for description, pid_tau, pid_had, sign_had in channels_2body_tau:
                if self.vcoupling["15"] <1e-9: continue
                label= "2body_tau_" + pid_tau + "_" + sign_had+pid_had
                br = f"hnl.get_2body_br_tau({pid_tau},{sign_had+pid_had})"
                dic = {'label': label, 'pid0': pid_tau, 'pid1':sign_had+pid_had, 'br': br , 'description': description.replace("l",lep[pid_lep])}
                dic_2body_mode["2body_tau"].append(dic)
                dic_2body_parent[r"$\tau$"].append(dic)
        
        dic_2body = {"mode": dic_2body_mode, "parent": dic_2body_parent}

        return dic_2body

    def get_channels_3body(self,):
        
        lep = {"11":r"e","13":r"\mu","15":r"\tau"}

        dic_3body_mode = {"3body_pseudo": [], "3body_vector": [], "3body_baryon": [], "3body_tau": [], "3body_tau_nutau": []}
        dic_3body_parent = {r"$\pi$": [], r"$K$": [], r"$K_S$": [], r"$K_L$": [], r"$D$": [], r"$D_0$": [], r"$D_s$": [], r"$B$": [], r"$B_0$": [], r"$B^0_s$": [], r"$B_c$": [], r"$\tau$": [], r"$\Lambda$": [], r"$\Xi$": [], r"$\Omega$": []}
        
        #last sign is for the lepton
        channels_pseudo = [
            [r'$D^0 \to K^- + l^+ + N$'             , '421' , '-321' , '-'],
            [r'$D^0 \to K^+ + l^- + N$'             , '-421', '321'  , '' ],
            [r'$D^+ \to \bar{K}^0 + l^+ + N$'       , '411' , '-311' , '-'],
            [r'$D^- \to K^0 + l^- + N$'             ,'-411' , '311'  , '' ],
            [r'$B^+ \to \bar{D}^0 + l^+ + N$'       , '521' ,  '-421', '-'],
            [r'$B^- \to D^0 + l^- + N$'             , '-521',  '421' , '' ],
            [r'$B^0 \to D^- + l^+ + N$'             , '511' , '-411' , '-'],
            [r'$B^0 \to D^+ + l^- + N$'             , '-511', '411'  , '' ],
            [r'$B^0_s \to D^-_s + l^+ + N$'         , '531' , '-431' , '-'],
            [r'$B^0_s \to D^+_s + l^- + N$'         , '-531', '431'  , '' ],
            [r'$B^+_c \to B^0 + l^+ + N$'           , '541' ,  '511' , '-'],
            [r'$B^-_c \to \bar{B}^0 + l^- + N$'     , '-541',  '-511', '' ],
            [r'$B^+_c \to B^0_s + l^+ + N$'         , '541' ,  '531' , '-'],
            [r'$B^-_c \to \bar{B}^0_s + l^- + N$'   , '-541',  '-531', '' ],
            [r'$K^0_S \to \pi^+ + l^- + N$'         , '310' , '211'  , '' ], 
            [r'$K^0_S \to \pi^- + l^+ + N$'         , '310' , '-211' , '-'],
            [r'$K^0_L \to \pi^+ + l^- + N$'         , '130' , '211'  , '' ],
            [r'$K^0_L \to \pi^- + l^+ + N$'         , '130' , '-211' , '-'],
            [r'$K^+ \to \pi^0 + l^+ + N$'           , '321' , '111'  , '-'], 
            [r'$K^- \to \pi^0 + l^- + N$'           , '-321', '111'  , '' ], 
            [r'$D_s^+ \to \eta + l^+ + N$'          , '431' , '221'  , '-'],
            [r'$D_s^- \to \eta + l^- + N$'          , '-431', '221'  , '' ],
            [r'$D_s^+ \to \eta\' + l^+ + N$'        , '431' , '331'  , '-'],
            [r'$D_s^- \to \eta\' + l^- + N$'        , '-431', '331'  , '' ],
            [r'$B^+ \to \pi^0 + l^+ + N$'           , '521' , '111'  , '-'],
            [r'$B^- \to \pi^0 + l^- + N$'           , '-521', '111'  , '' ],
            [r'$B^+_c \to D^0 + l^+ + N$'           , '541' , '421'  , '-'],
            [r'$B^-_c \to \bar{D}^0 + l^- + N$'     , '-541', '-421' , '' ],
            [r'$B^+_c \to \eta_c + l^+ + N$'        , '541' , '441'  , '-'],
            [r'$B^-_c \to \eta_c + l^- + N$'        , '-541', '441'  , '' ],
            [r'$D^0 \to \pi^- + l^+ + N$'           , '421' , '-211' , '-'],
            [r'$\bar{D}^0 \to \pi^+ + l^- + N$'     , '-421', '211'  , '' ],
            [r'$D^+ \to \pi^0 + l^+ + N$'           , '411' , '111'  , '-'],
            [r'$D^- \to \pi^0 + l^- + N$'           , '-411', '111'  , '' ],
            [r'$D_s^+ \to K^0 + l^+ + N$'           , '431' , '311'  , '-'],
            [r'$D_s^- \to \bar{K}^0 + l^- + N$'     , '-431', '-311' , '' ],
            [r'$B^0 \to \pi^- + l^+ + N$'           , '511' , '-211' , '-'],
            [r'$\bar{B}^0 \to \pi^+ + l^- + N$'     , '-511', '211'  , '' ],
            [r'$B^0_s \to K^- + l^+ + N$'           , '531' , '-321' , '-'],
            [r'$\bar{B}^0_s \to K^+ + l^- + N$'     , '-531', '321'  , '' ],
            [r'$D^+ \to \eta + l^+ + N$'            , '411' , '221'  , '-'],
            [r'$D^- \to \eta + l^- + N$'            , '-411' , '221'  , ''],
            [r'$D^+ \to \eta\' + l^+ + N$'          , '411' , '331'  , '-'],
            [r'$D^- \to \eta\' + l^- + N$'          , '-411' , '331'  , ''],
            [r'$B^+ \to \eta + l^+ + N$'            , '521' , '221'  , '-'],
            [r'$B^- \to \eta + l^- + N$'            , '-521' , '221'  , ''],
            [r'$B^+ \to \eta\' + l^+ + N$'          , '521' , '331'  , '-'],
            [r'$B^- \to \eta\' + l^- + N$'          , '-521' , '331'  , '']
        ]
        
        #last sign is for the lepton
        channels_vector = [
            [r'$D^0 \to K^{*-} + l^+ + N$'                  ,'421' ,'-323', '-' ],
            [r'$D^0 \to K^{*+} + l^- + N$'                  ,'-421', '323', ''  ],
            [r'$B^+ \to \bar{D}^*0 + l^+ + N$'              ,'521' ,'-423', '-' ],
            [r'$B^- \to D^*0 + l^- + N$'                    ,'-521','423' , ''  ],
            [r'$B^0 \to D^{*-} + l^+ + N$'                  ,'511' ,'-413', '-' ],
            [r'$B^0 \to D^{*+} + l^- + N$'                  ,'-511','413' , ''  ],
            [r'$B^0_s \to D^{*-}_s + l^+ + N$'              ,'531' ,'-433', '-' ],
            [r'$B^0_s \to D^{*+}_s + l^- + N$'              ,'-531','433' , ''  ],
            [r'$B^+_c \to B^{*0} + l^+ + N$'                ,'541' ,'513' , '-' ],
            [r'$B^-_c \to \bar{B}^{*0} + l^- + N$'          ,'-541','-513', ''  ],
            [r'$B^+_c \to B^{*0}_s+ l^+ + N$'               ,'541' ,'533' , '-' ],
            [r'$B^-_c \to \bar{B}^{*0}_s+ l^- + N$'         ,'-541','-533', ''  ],
            [r'$B^+ \to \rho^0 + l^+ + N$'                  , '521', '113', '-' ],
            [r'$B^- \to \rho^0 + l^- + N$'                  ,'-521', '113', ''  ],
            [r'$B^+_c \to J/\psi + l^+ + N$'                , '541', '443', '-' ],
            [r'$B^-_c \to J/\psi + l^- + N$'                ,'-541', '443', ''  ],
            [r'$D^0 \to \rho^- + l^+ + N$'                  , '421','-213', '-' ],
            [r'$\bar{D}^0 \to \rho^+ + l^- + N$'            ,'-421', '213', ''  ],
            [r'$D^+ \to \rho^0 + l^+ + N$'                  , '411', '113', '-' ],
            [r'$D^- \to \rho^0 + l^- + N$'                  ,'-411', '113', ''  ],
            [r'$D^- \to K^{*0} + l^- + N$'                  ,'-411', '313', ''  ],
            [r'$D^+ \to \bar{K}^{*0} + l^+ + N$'            ,'-411', '-313', '-'],
            [r'$D^+_s \to K^{*0} + l^+ + N$'                , '431', '313', '-' ],
            [r'$D^-_s \to \bar{K}^{*0} + l^- + N$'          ,'-431','-313', '-' ],
            [r'$D^+_s \to \phi + l^+ + N$'                  , '431', '333', '-' ],
            [r'$D^-_s \to \phi + l^- + N$'                  ,'-431', '333', ''  ],
            [r'$B^0 \to \rho^- + l^+ + N$'                  , '511','-213', '-' ],
            [r'$\bar{B^0} \to \rho^+ + l^- + N$'            ,'-511', '213', ''  ],
            [r'$B^0_s \to K^{*-} + l^+ + N$'                , '531','-323', '-' ],
            [r'$\bar{B}^0_s \to K^{*+} + l^- + N$'          ,'-531', '323', ''  ],
            [r'$B^+_c \to D^{*0} + l^+ + N$'                , '541', '423', '-' ],
            [r'$B^-_c \to \bar{D}^{*0} + l^- + N$'          ,'-541','-423', ''  ],
            [r'$D^- \to \omega + l^- + N$'                  ,'-411', '223', ''  ],
            [r'$D^+ \to \omega + l^+ + N$'                  , '411', '223', '-' ],
            [r'$B^- \to \omega + l^- + N$'                  ,'-521', '223', ''  ],
            [r'$B^+ \to \omega + l^+ + N$'                  , '521', '223', '-' ],
        ]

        channels_baryon = [
            [r'$\Lambda^-_c \to \Lambda^0 + l^- + N$'  ,'-4122' ,'3122'  ,'' ],
            [r'$\Lambda^+_c \to \Lambda^0 + l^+ + N$'  ,'4122'  ,'3122'  ,'-'],
            [r'$\Lambda^0_b \to \Lambda_c^+ + l^- + N$','5122'  ,'4122'  ,'' ],
            [r'$\Lambda^0_b \to \Lambda_c^- + l^+ + N$','5122'  ,'-4122' ,'-'],
            [r'$\Xi^0_c \to \Xi^+ + l^- + N$'          ,'4132'  ,'3312'  ,'' ],
            [r'$\Xi^0_c \to \Xi^- + l^+ + N$'          ,'4132'  ,'-3312' ,'-'],
            [r'$\Xi^0_b \to \Xi^+_c + l^- + N$'        ,'5232'  ,'4232'  ,'' ],
            [r'$\Xi^0_b \to \Xi^-_c + l^+ + N$'        ,'5232'  ,'-4232' ,'-'],
            [r'$\Omega_b^- \to \Omega_c^0 + l^- + N$'  ,'-5332' ,'4332'  ,'' ],
            [r'$\Omega_b^+ \to \Omega_c^0 + l^+ + N$'  ,'5332'  ,'4332'  ,'-']
        ]

        channels_tau_1 =  [
            [r'$\tau^- \to l^- + \nu_{\tau} + N$','15','16',''],
            [r'$\tau^+ \to l^+ + \bar{\nu}_{\tau} + N$','-15','-16','-'],
        ]

        channels_tau_2 = [
            [r'$\tau^- \to l^- + \bar{\nu}_l + N$','15',''],
            [r'$\tau^+ \to l^+ + \nu_l + N$','-15','-']
        ]
        
        output=[]
        
        #Pseudocalar
        for description, pid_parent, pid_daughter, sign_lep in channels_pseudo: 
            for pid_lep in ["11","13","15"]:
                if self.vcoupling[pid_lep] <1e-9: continue
                integration = "dq2dE"
                label = "3body_pseudo_" + pid_parent + "_" +pid_daughter+ "_" + sign_lep+pid_lep
                br = f"hnl.get_3body_dbr_pseudoscalar({pid_parent},{pid_daughter},{sign_lep+pid_lep})"
                dic = {'label': label, 'pid0': pid_parent,'pid1': pid_daughter,'pid2': sign_lep+pid_lep, 'br': br,'integration': integration, 'description': description.replace("l",lep[pid_lep])} 
                dic_3body_mode["3body_pseudo"].append(dic)
                #group by parent
                #kaons
                if str(abs(int(pid_parent))) in ["321"]:
                    dic_3body_parent[r"$K$"].append(dic)
                if str(abs(int(pid_parent))) in ["310"]:
                    dic_3body_parent[r"$K_S$"].append(dic)
                if str(abs(int(pid_parent))) in ["130"]:
                    dic_3body_parent[r"$K_L$"].append(dic)
                #D mesons
                if str(abs(int(pid_parent))) in ["411"]:
                    dic_3body_parent[r"$D$"].append(dic)
                if str(abs(int(pid_parent))) in ["421"]:
                    dic_3body_parent[r"$D_0$"].append(dic)
                if str(abs(int(pid_parent))) in ["431"]:
                    dic_3body_parent[r"$D_s$"].append(dic)
                #B mesons
                if str(abs(int(pid_parent))) in ["521"]:
                    dic_3body_parent[r"$B$"].append(dic)
                if str(abs(int(pid_parent))) in ["511"]:
                    dic_3body_parent[r"$B_0$"].append(dic)
                if str(abs(int(pid_parent))) in ["531"]:
                    dic_3body_parent[r"$B^0_s$"].append(dic)
                if str(abs(int(pid_parent))) in ["541"]:
                    dic_3body_parent[r"$B_c$"].append(dic)

        #Vector
        for description, pid_parent, pid_daughter, sign_lep in channels_vector: 
            for pid_lep in ["11","13","15"]:
                if self.vcoupling[pid_lep] <1e-9: continue
                integration = "dq2dE"
                label = "3body_vector_" + pid_parent + "_" +pid_daughter+ "_" + sign_lep+pid_lep
                br =  f"hnl.get_3body_dbr_vector({pid_parent},{pid_daughter},{sign_lep+pid_lep})"
                dic = {'label': label, 'pid0': pid_parent,'pid1': pid_daughter,'pid2': sign_lep+pid_lep, 'br': br,'integration': integration, 'description': description.replace("l",lep[pid_lep])} 
                dic_3body_mode["3body_vector"].append(dic)
                #group by parent
                #kaons
                if str(abs(int(pid_parent))) in ["321"]:
                    dic_3body_parent[r"$K$"].append(dic)
                if str(abs(int(pid_parent))) in ["310"]:
                    dic_3body_parent[r"$K_S$"].append(dic)
                if str(abs(int(pid_parent))) in ["130"]:
                    dic_3body_parent[r"$K_L$"].append(dic)
                #D mesons
                if str(abs(int(pid_parent))) in ["411"]:
                    dic_3body_parent[r"$D$"].append(dic)
                if str(abs(int(pid_parent))) in ["421"]:
                    dic_3body_parent[r"$D_0$"].append(dic)
                if str(abs(int(pid_parent))) in ["431"]:
                    dic_3body_parent[r"$D_s$"].append(dic)
                #B mesons
                if str(abs(int(pid_parent))) in ["521"]:
                    dic_3body_parent[r"$B$"].append(dic)
                if str(abs(int(pid_parent))) in ["511"]:
                    dic_3body_parent[r"$B_0$"].append(dic)
                if str(abs(int(pid_parent))) in ["531"]:
                    dic_3body_parent[r"$B^0_s$"].append(dic)
                if str(abs(int(pid_parent))) in ["541"]:
                    dic_3body_parent[r"$B_c$"].append(dic)

        #Baryons
        for description, pid_parent, pid_daughter, sign_lep in channels_baryon: 
            for pid_lep in ["11","13","15"]:
                if self.vcoupling[pid_lep] <1e-9: continue
                integration = "dq2dm122"
                label = "3body_baryon_" + pid_parent + "_" +pid_daughter+ "_" + sign_lep+pid_lep
                br =  f"hnl.get_3body_dbr_baryon({pid_parent},{pid_daughter},{sign_lep+pid_lep})"
                dic = {'label': label, 'pid0': pid_parent,'pid1': pid_daughter,'pid2': sign_lep+pid_lep, 'br': br,'integration': integration, 'description': description.replace("l",lep[pid_lep])} 
                dic_3body_mode["3body_baryon"].append(dic)
                #group by parent
                if str(abs(int(pid_parent))) in ["4122","5122"]:
                    dic_3body_parent[r"$\Lambda$"].append(dic)
                if str(abs(int(pid_parent))) in ["4132","5232"]:
                    dic_3body_parent[r"$\Xi$"].append(dic)
                if str(abs(int(pid_parent))) in ["5332"]:
                    dic_3body_parent[r"$\Omega$"].append(dic)

        #Tau
        for description, pid_parent, pid_nu, sign_lep in channels_tau_1: 
                for pid_lep in ["11","13"]:
                    if self.vcoupling[pid_lep] <1e-9: continue
                    integration = "dE"
                    label = "3body_tau_" + pid_parent + "_" +sign_lep+pid_lep+ "_" + pid_nu
                    br = f"hnl.get_3body_dbr_tau({pid_parent},{sign_lep+pid_lep},{pid_nu})"
                    dic = {'label': label, 'pid0': pid_parent,'pid1': sign_lep+pid_lep,'pid2': pid_nu, 'br': br,'integration': integration, 'description': description.replace("l",lep[pid_lep])} 
                    dic_3body_mode["3body_tau_nutau"].append(dic)
                    dic_3body_parent[r"$\tau$"].append(dic)

        for description, pid_parent, sign_lep in channels_tau_2: 
                for pid_lep in ["11","13"]:
                    pid_nu = str(int(pid_lep)+1)
                    if self.vcoupling["15"] <1e-9: continue
                    integration = "dE"
                    label = "3body_tau_" + pid_parent + "_" +sign_lep+pid_lep+ "_" + pid_nu
                    br =  f"hnl.get_3body_dbr_tau({pid_parent},{sign_lep+pid_lep},{pid_nu})"
                    dic = {'label': label, 'pid0': pid_parent,'pid1': sign_lep+pid_lep,'pid2': pid_nu, 'br': br,'integration': integration, 'description': description.replace("l",lep[pid_lep])} 
                    dic_3body_mode["3body_tau"].append(dic)
                    dic_3body_parent[r"$\tau$"].append(dic)

        dic_3body = {"mode": dic_3body_mode, "parent": dic_3body_parent}

        return dic_3body

    def get_bounds(self):

        coupling =  (self.vcoupling['11'],self.vcoupling['13'],self.vcoupling['15'])

        bounds_100 = [
     ['bounds_100/bounds_atlas_2022.txt'  , 'Atlas \n (2022)'        , 6.05   , 0.00035  ,  -35 ], 
     ['bounds_100/bounds_charm.txt'       , 'CHARM'                  , 0.8  , 0.00019    , -35], 
     ['bounds_100/bounds_bebc_barouki.txt', 'BEBC \n (Barouki et al)', 1.114 , 0.000055 ,  -15 ], 
     ['bounds_100/bounds_belle.txt'       , 'Belle'                  , 1.0 , 0.005  ,  -20 ], 
     ['bounds_100/bounds_delphi_long.txt' , 'Delphi \n (long)'       , 2.17 , 0.0028   ,  -27 ], 
     ['bounds_100/bounds_t2k.txt'         , 'T2K'                    , 0.45 , 2.6e-05 ,  50], 
     ['bounds_100/bounds_cosmo.txt'       , 'BBN'                    , 0.38  , 1.3e-5 , -40], 
     ['bounds_100/bounds_cms_2022.txt'    , 'CMS \n (2022)'          , 6.0   , 0.001   , -35], 
     ['bounds_100/bounds_pienu_2017.txt'  , 'PIENU \n (2017)'        , 0.117, 0.0001  ,  75], 
     ['bounds_100/bounds_na62.txt'        , 'NA62'                   , 0.17 , 4.0e-05 ,   -15]
    ]
        bounds_010 = [
    ['bounds_010/bounds_microboone_higgs.txt', r'$\mu$BooNe'    , 0.144, 0.00038   ,-50],
    ['bounds_010/bounds_t2k.txt'             , 'T2K'            , 0.30, 3.5e-05,-50],
    ['bounds_010/bounds_cosmo.txt'           , 'BBN'            , 0.41, 1.1e-05  ,-40],
    ['bounds_010/bounds_nutev.txt'           , 'NuTeV'          , 0.765, 0.00019   ,-30],
    ['bounds_010/bounds_bebc.txt'            , 'BEBC'           , .9, 0.000374   ,-40],
    ['bounds_010/bounds_na62.txt'            , 'NA62'           , 0.26, 0.000151   ,0  ],
    ['bounds_010/bounds_cms_2022.txt'        , 'CMS \n (2022)'  , 5.0  , 0.000303   ,-30],
    ['bounds_010/bounds_na3.txt'             , 'NA3'            , 1.26, 0.0041    ,-40],
]

        bounds_001 = [
    ['bounds_001/bounds_delphi_short.txt', 'Delphi \n (short)'       , 5.8, 0.0023  ,0  ],
    ['bounds_001/bounds_cosmo.txt'       , 'BBN'                     , 0.14, 1.5e-04,-70],
    ['bounds_001/bounds_bebc_barouki.txt', 'BEBC \n (Barouki et al.)', 0.41, 0.00048,-45 ],
    ['bounds_001/bounds_charm_2021.txt'  , 'CHARM \n (2021)'         , 0.52, 0.0017 , -40],
    ['bounds_001/bounds_delphi_long.txt' , 'Delphi \n (long)'        , 2.162, 0.00236 ,-30 ],
    ['bounds_001/bounds_babar_2022.txt'  , 'BaBar'                   , 0.857, 0.00377 ,-70 ],
    ]
        if coupling == (1,0,0): return bounds_100
        elif coupling ==  (0,1,0): return bounds_010
        elif coupling ==  (0,0,1): return bounds_001
        else: return []

    
    #########################################################################################
    
    ## HNL Decay
    
    #########################################################################################
    
    def HNL_Decay_init(self,couplings):
        
        self.cutoff = self.masses('331')
        
        self.U = {'e':couplings[0], 'anti_e': couplings[0],
                  'mu':couplings[1], 'anti_mu': couplings[1],
                  'tau':couplings[2], 'anti_tau': couplings[2]} 
        
        self.Gf = 1.166378*10**(-5)
        
        self.sin2w = 0.23121 
        
        
        self.GeVtoS = (6.582119569 * 10**-25) 
        
        self.c = 299792458 

        script_dir = os.path.realpath(os.path.dirname(__file__))
        alph_data = pd.read_csv(f'{script_dir}/alph_str.csv')
        
        self.alph_s = interpolate.interp1d( alph_data['m'], alph_data['alph'],bounds_error = False,fill_value = 0) 

        self.plot_labels = {
        #Leptons
        'e': r'$e$',
        anti('e'): r'$e$',
        'mu': r'$\mu$',
        anti('mu'): r'$\mu$',       
        'tau': r'$\tau$',
        anti('tau'): r'$\tau$',
        'nu':r'$\nu$',
        #Pseudos
        'pi+':r'$\pi^+$',
        anti('pi+'):r'$\pi^-$',
        'pi0':r'$\pi^0$',
        'K+': r'$K^+$',
        anti('K+'): r'$K^-$',
        'D+': r'$D^+$',
        anti('D+'): r'$D^-$',
        'Ds+': r'$D_s^+$',
        anti('Ds+'): r'$D_s^-$',
        'eta': r'$\eta$',
        'etap': r'$\eta\prime$',
        #Vectors
        'rho+':r'$\rho^+$',
        anti('rho+'):r'$\rho^-$',
        'rho0':r'$\rho^0$',
        'K+*':r'$K^{*+}$',
        anti('K+*'):r'$K^{*-}$',
        'omega':r'$\omega$',
        'phi':r'$\phi$',
        '3had': r'$(\geq3H)$',
        #Quarks
        'd': r'$d$',
        anti('d'): r'$\overline{d}$',
        'u': r'$u$',
        anti('u'): r'$\overline{u}$',
        'c': r'$c$',
        anti('c'): r'$\overline{c}$',
        's': r'$s$',
        anti('s'): r'$\overline{s}$',
        't': r'$t$',
        anti('t'): r'$\overline{t}$',
        'b': r'$b$',
        anti('b'): r'$\overline{b}$',
        }

        
        self.plot_labels_neut = {
        #Leptons
        'e': r'$e$',
        anti('e'): r'$e$',
        'mu': r'$\mu$',
        anti('mu'): r'$\mu$',
        'tau': r'$\tau$',
        anti('tau'): r'$\tau$',
        've': r'$\nu_e$',
        anti('ve'): r'$\overline{\nu}_e$',
        'vmu': r'$\nu_\mu$',
        anti('vmu'): r'$\overline{\nu}_\mu$',
        'vtau': r'$\nu_\tau$',
        anti('vtau'): r'$\overline{\nu}_\tau$',
        'nu':r'$\nu$',
        'pi+':r'$\pi$',
        anti('pi+'):r'$\pi$',
        'pi0':r'$\pi^0$',
        'K+': r'$K$',
        anti('K+'): r'$K$',
        'eta': r'$\eta$',
        'etap': r'$\eta\prime$',
        'D+': r'$D$',
        anti('D+'): r'$D$',
        'Ds+': r'$D_s$',
        anti('Ds+'): r'$D_s$',
        'rho+':r'$\rho$',
        anti('rho+'):r'$\rho$',
        'rho0':r'$\rho^0$',
        'K+star':r'$K^{*}$',
        anti('K+star'):r'$K^{*}$',
        'omega':r'$\omega$',
        'phi': r'$\phi$',
        '3had': r'$(\geq3H)$',
        'd': r'$d$',
        anti('d'): r'$\overline{d}$',
        'u': r'$u$',
        anti('u'): r'$\overline{u}$',
        'c': r'$c$',
        anti('c'): r'$\overline{c}$',
        's': r'$s$',
        anti('s'): r'$\overline{s}$',
        't': r'$t$',
        anti('t'): r'$\overline{t}$',
        'b': r'$b$',
        anti('b'): r'$\overline{b}$',
            
        }

        
        
        #define particle content
        leptons = ['e','mu','tau']

        vectors = {'charged':['rho+','K+star'], 'neutral': ['rho0','omega','phi'] }

        pseudos = {'charged':['pi+','K+','D+','Ds+'], 'neutral':['pi0','eta','etap'] }
        
        neutrinos = ['nu']
        
        quarks = {'up':['u','c','t'], 'down': ['d','s','b']}

        self.particle_content = {'leptons':leptons,'neutrinos':neutrinos,'vectors':vectors,'pseudos':pseudos,'quarks':quarks}



        ##Compile all allowed decay modes for each decay channel##
        
        #N -> nu_alpha l_beta l_beta
        null = [('nu','e','anti_e'),('nu','mu','anti_mu'),('nu','tau','anti_tau')]

        
        
        #N -> l_alpha l_beta nu
        llnu = [
        ('e',anti('mu'),'nu'), ('mu',anti('e'),'nu'),
        ('e',anti('tau'),'nu'), ('tau',anti('e'),'nu'),
        ('mu',anti('tau'),'nu'), ('tau',anti('mu'),'nu')    
        ] 

        #N -> nu nu nu
        nu3 = [('nu','nu','nu')]
    
        #N -> nu_alpha P    
        nuP = [('nu','pi0'),('nu','eta'),('nu','etap')]

     

        #N -> l_alpha P
        lP = []

        for l in leptons:

            for P in pseudos['charged']:

                mode = (l,P)

                lP.append(mode)
                
                #conjugate mode 
                lP.append(conjugate(mode))
        
        #N -> nu_alpha V
        nuV = []

        

        for V in vectors['neutral']:

            mode = ('nu',V)

            nuV.append(mode)

        #N -> l_alpha V 
        lV = []

        for l in leptons:

            for V in vectors['charged']:

                mode = (l,V)

                lV.append(mode)
                
                #conjugate mode
                lV.append(conjugate(mode))
                    
        #N -> nu_alpha q q 
        nuqq = []
        
        
            
        for q in quarks['up'] + quarks['down']:

            mode = ('nu',q,anti(q))

            nuqq.append(mode)
                
        #N -> l_alpha u d      
        lud = []
        
        for l in leptons: 
            
            for u in quarks['up']:
                
                for d in quarks['down']:
                    
                    mode = (l,u,anti(d))
                    
                    lud.append(mode)
                    
                    lud.append(conjugate(mode))

        lhad = [] 
        
        for l in leptons: 

            mode = (l,'3had')

            conj_mode = (anti(l),'3had')

            lhad.append(mode)

            lhad.append(conj_mode)

        nuhad = [('nu','3had')]
                

        self.modes = {'null':null,'llnu':llnu,'nu3':nu3,'nuP':nuP,'lP':lP,'nuV':nuV, 'lV':lV,'lhad':lhad,'nuhad':nuhad,'lud':lud,'nuqq':nuqq}
        
        self.decay_modes = [] 
        
        self.modes_inactive = {}
        self.modes_active = {}
        
        #Compile all modes that are allowed by couplings and filter out those that are not
        for channel in self.modes.keys():
            
            modes_inactive = []
            modes_active = []
            
            for mode in self.modes[channel]:
            
            
                if channel in ['null','nuhad','nuV','nuP','nu3','nuqq']: 
                    if channel != 'nuqq': self.decay_modes.append(mode)
                    modes_active.append(mode)
                
                elif channel in ['lV','lP','lhad','lud']: 
                    if self.U[mode[0]] != 0: 
                        modes_active.append(mode)
                        if channel != 'lud': self.decay_modes.append(mode)
                    else: modes_inactive.append(mode)
                
                elif channel in ['llnu']:
                    if self.U[mode[0]] != 0 or self.U[mode[1]] != 0: 
                        modes_active.append(mode)
                        self.decay_modes.append(mode)
                    else: modes_inactive.append(mode)
                        
                     
            
            

            
            
            self.modes_inactive[channel] = modes_inactive
            
            self.modes_active[channel] = modes_active
        
        
    def gen_widths(self,mpts):
        
        """
        
        Generate decay widths for all active modes
        
        """
        
        channels = self.modes_active.keys()
        
        self.model_widths = {}
        
        self.mpts = mpts

        Gamma = {'null': self.Gamma_null, 'llnu':self.Gamma_llnu, 'lP': self.Gamma_lP, 'nuP': self.Gamma_nuP ,
         'lV': self.Gamma_lV, 'nuV': self.Gamma_nuV, 'nu3': self.Gamma_nu3,'nuqq': self.Gamma_nuqq,'lud': self.Gamma_lud}
       
        #iterate through each decay channel
        for channel in channels:
            

            if channel not in ['nuhad','lhad']:
            
                for mode in self.modes_active[channel]:
                    
                    
                    #Evaluate the decay width with or without a cutoff
                    
                    
                    gamma_pts = [Gamma[channel](m,mode) for m in self.mpts]
       
                    self.model_widths[mode] = gamma_pts
            
            

        
        
        for mode in self.modes_active['lhad']:

            gamma_quark = np.zeros(len(self.mpts))
            
            for quark_mode in self.modes_active['lud']: 
                
                if quark_mode[0] == mode[0]: gamma_quark += np.array(self.model_widths[quark_mode])

            gamma_had = np.zeros(len(self.mpts))

            for had_mode in self.modes_active['lP']: 

                if had_mode[0] == mode[0]: gamma_had += np.array(self.model_widths[had_mode])
            
            for had_mode in self.modes_active['lV'] : 

                if had_mode[0] == mode[0] : gamma_had += np.array(self.model_widths[had_mode])

            gamma = gamma_quark - gamma_had

            gamma[gamma < 0] = 0
            
            self.model_widths[mode] = list(gamma)
            

        
            
        
        
        channel_widths = {}
        
        for mode in self.modes_active['nuhad']:
            
            gamma_quark = np.zeros(len(self.mpts))
            gamma_had = np.zeros(len(self.mpts))

            for quark_mode in self.modes_active['nuqq']: 
                
                if quark_mode[0] == mode[0]: gamma_quark += np.array(self.model_widths[quark_mode])
            
            for had_mode in self.modes_active['nuP']: 

                if had_mode[0] == mode[0]: gamma_had += np.array(self.model_widths[had_mode])
            
            for had_mode in self.modes_active['nuV']: 

                if had_mode[0] == mode[0] : gamma_had += np.array(self.model_widths[had_mode])

            
            gamma = gamma_quark - gamma_had

            gamma[gamma < 0] = 0
            
            self.model_widths[mode] = list(gamma)
            
        
       
        
    def gen_ctau(self,mpts,coupling =1):

            """

            Generate HNL Lifetime

            """

            self.mpts = mpts

            #generate decay widths
            self.gen_widths(mpts=self.mpts)

            total_width = []

            #iterate over mass points
            for i in range(len(mpts)):

                m = mpts[i]
                
                gamma_T = 0

            

                #sum over individual decay widths 
                for channel in self.modes_active.keys():

                    if m <= 1.0 and channel not in ['nuqq','lud','nuhad','lhad']:

                        for mode in self.modes_active[channel]:
    
    
                            gamma = self.model_widths[mode][i]
    
    
    
                            gamma_T += gamma

            
                    elif m > 1.0 and channel not in ['nuV','lV','nuP','lP','nuhad','lhad']:
                        
                        for mode in self.modes_active[channel]:
    
                            gamma = self.model_widths[mode][i]

                            gamma_T += gamma


                total_width.append(gamma_T)
            
            ctau = [float((Decimal(self.c)/Decimal(g)))*self.GeVtoS/coupling**2 for g in total_width]

            self.total_width = total_width

            self.ctau = ctau

    def gen_brs(self):


            """

            Generate Branching ratios

            """

            


            self.model_brs = {}

            #iterate over decay channels
        
            #sum over branching ratios
            for mode in self.model_widths.keys():

                mode_br = []


                for i in range(len(self.model_widths[mode])):

                    gamma_partial = self.model_widths[mode][i]

                    gamma_total = self.total_width[i]



                    mode_br.append( gamma_partial/gamma_total)


                


                self.model_brs[mode] = mode_br
                
    def save_data(self,save_ctau,save_brs):
        
        """
        
        Save decay data
        
        """
        
        
    
        if save_ctau: 

            path = "model/ctau"
            os.makedirs(path,exist_ok = True)
            
            #save ctau 
            ctau_pts = self.ctau
            
            df_data = {'m': self.mpts,'ctau':ctau_pts}


            df=pd.DataFrame(df_data)

            save_path = f"{path}/ctau.txt"

            df.to_csv(save_path,sep=' ',header=False,index=False)


        if save_brs: 
        
            #clean br directories 
            for channel in self.modes_active.keys():

                
                

                channel_path_br = f"model/br/"

                os.makedirs(channel_path_br,exist_ok = True)
                try:
                    for f in os.listdir(channel_path_br):
                        os.remove(f"{channel_path_br}/{f}")


                except:
                    pass
           
             
           
                
            for mode in self.model_brs.keys():
                
                br_pts = self.model_brs[mode]
                
                df_data = {'m': self.mpts,'br':br_pts}


                df=pd.DataFrame(df_data)
                
               
                save_path = f"model/br/"

                for p in mode: save_path += f"{p}-"
                    
                
                

                df.to_csv(save_path[:-1]+".csv",sep=' ',header=False,index=False)

  

        
        
        
        
# Decay Widths
#N-> l P
    def Gamma_lP(self,m,mode): 

        l,P = mode

        xl = self.masses(pid(l))/m

        xP = self.masses(pid(P))/m

        if m>= self.masses(pid(l)) + self.masses(pid(P)): 
            
            prefactor = self.U[l]**2 * self.Gf**2 * m**3 * self.fH(pid(P))**2 * self.VH(pid(P))**2 / (16*np.pi)
            
            kin = np.sqrt(Lambda(1,xP**2,xl**2))*(1-xP**2-xl**2*(2+xP**2-xl**2))
            
            return prefactor*kin

        else: return 0 
       
            
    
    #N -> nu P 
    def Gamma_nuP(self,m,mode):

        nu,P  = mode
        
        

        xP = self.masses(pid(P))/m

        if m >=  self.masses(pid(P)): 
            
            prefactor =   (self.U['e']**2 + self.U['mu']**2 + self.U['tau']**2) * self.Gf**2 * m**3 * self.fH(pid(P))**2 / (16*np.pi)
            
            kin = (1-xP**2)**2
            
            return prefactor*kin
            
        else:       return 0 
        

        
       

    #N -> lV 
    def Gamma_lV(self,m,mode):

        l,V = mode

        prefactor =   self.U[l]**2*self.Gf**2 * m**3 * self.fH(pid(V))**2 * self.VH(pid(V))**2 / (16*np.pi) 


        xl = self.masses(pid(l))/m

        xV = self.masses(pid(V))/m

        if m >= self.masses(pid(V))+ self.masses(pid(l)):

            prefactor =   self.U[l]**2*self.Gf**2 * m**3 * self.fH(pid(V))**2 * self.VH(pid(V))**2 / (16*np.pi) 

            kin = np.sqrt(Lambda(1,xV**2,xl**2)) * ( (1-xV**2)*(1+2*xV**2) + xl**2*(xV**2+xl**2-2) )

            return prefactor*kin

        else: return 0
        


    #N -> nu_alpha V 
    def Gamma_nuV(self,m,mode):

        nu,V = mode

        k_V = {
               'rho0': 1-2*self.sin2w, 
               'anti_rho0': 1-2*self.sin2w,
               'omega': -2*self.sin2w/3,
               'anti_omega': -2*self.sin2w/3,
               'phi': -np.sqrt(2)*(0.5 - 2*self.sin2w/3),
               'anti_phi': -np.sqrt(2)*(0.5 - 2*self.sin2w/3)
              }

        xV = self.masses(pid(V))/m

        if m>= self.masses(pid(V)): 
            
            prefactor = (self.U['e']**2 + self.U['mu']**2 + self.U['tau']**2) * self.Gf**2 * m**3 * self.fH(pid(V))**2 * k_V[V]**2 / (16*np.pi)
            
            kin = (1+2*xV**2)*(1-xV**2)**2
            
            return prefactor*kin
            
        else: return 0
            

    #N -> l_alpha l_beta nu_beta
    def Gamma_llnu(self,m,mode):

        l1,l2,nu = mode

        xl1 = self.masses(pid(l1))/m
        
        xl2 = self.masses(pid(l2))/m


        #evaluate mass thresholds 
        #if 1 >= yl1 + yl2:
        if m >= self.masses(pid(l1)) + self.masses(pid(l2)):

            prefactor = self.Gf**2 * m**5/ (192 * np.pi**3)

            kin =  ( self.U[l1]**2 * I_1(0, xl1**2, xl2**2) + self.U[l2]**2 * I_1(0,xl1**2,xl2**2) )
            
            return prefactor*kin
            
        else: return 0               


    #N -> nu_alpha l_beta l_beta
    def Gamma_null(self,m,mode):
        nu,l1,l2 = mode

        xl = self.masses(pid(l1))/m

        if m >= 2* self.masses(pid(l1)):
            
            sw2 = self.sin2w

            C1 = (1-4*sw2+8*sw2**2)/4

            C2 = (2*sw2**2-sw2)/2
    
            del_e = delta(int(pid('e')),int(pid(l1)))
            del_mu = delta(int(pid('mu')),int(pid(l1)))
            del_tau = delta(int(pid('tau')),int(pid(l1)))
    
    
            prefactor =  self.Gf**2 * m**5 / (96*np.pi**3)
    
            term_e = self.U['e']**2 * ( (C1 + 2*sw2*del_e)*f1(xl) + (C2 + sw2*del_e)*f2(xl) )

            term_mu = self.U['mu']**2 * ( (C1 + 2*sw2*del_mu)*f1(xl) + (C2 + sw2*del_mu)*f2(xl) )

            term_tau = self.U['tau']**2 * ( (C1 + 2*sw2*del_tau)*f1(xl) + (C2 + sw2*del_tau)*f2(xl) )

            kin = (term_e + term_mu + term_tau)
            
            return prefactor*kin
        
        else: return 0   
        
    #N -> nu nu nu 
    def Gamma_nu3(self,m,mode):
        
        nu,_,_ = mode

        gamma = (self.U['e']**2 + self.U['mu']**2 + self.U['tau']**2)*self.Gf**2 * m**5 / (96*np.pi**3)

        return gamma


    #N -> l_alpha u d
    def Gamma_lud(self,m,mode):

        l,u,d = mode


        V = self.VHHp(pid(u),pid(d))

        xu = self.masses(pid(u))/m
        xd = self.masses(pid(d))/m
        xl = self.masses(pid(l))/m

        alph = float(self.alph_s(m))
        
        qcd = 1 + alph/np.pi + 5.2*alph**2/np.pi**2 + 26.4*alph**3/np.pi**3
        

        if m >= self.masses(pid(u))+self.masses(pid(d))+self.masses(pid(l)) and m > 1.0:
                
            
            prefactor = self.U[l]**2*self.Gf**2 * V**2 * m**5 / (64 * np.pi**3)

            if u == 'u' or anti(u) == 'u': 
                
                if d == 'd' or anti(d) == 'd':
                    if l == 'tau' or anti(l) == 'tau':
                        if   m > self.masses(pid(l)) + self.masses("211")+ self.masses("211"): 

                            
                            kin = qcd*I_1(xl**2,xu**2,xd**2)*np.sqrt(1-(self.masses(pid(l)) + self.masses("211")+ self.masses("211"))**2/m**2)
                            
                            
                        else:
                            kin = 0 
                    else: kin = qcd*I_1(xl**2,xu**2,xd**2)

                elif d == 's' or anti(d) == 's':
                    if l == 'tau' or anti(l) == 'tau':
                        if   m > self.masses(pid(l)) + self.masses("321")+ self.masses("211"):
                            
                            kin = qcd*I_1(xl**2,xu**2,xd**2)*np.sqrt(1-(self.masses(pid(l)) + self.masses("321")+ self.masses("211"))**2/m**2)
                            
                        else: kin = 0 
                    else: kin = qcd*I_1(xl**2,xu**2,xd**2)
                    
                else:
                    kin = I_1(xl**2,xu**2,xd**2)
            else: 
                kin = I_1(xl**2,xu**2,xd**2)

            
            return prefactor*kin
        
        else: return 0

        


    #N -> nu_alpha q q
    def Gamma_nuqq(self,m,mode):

      
        
        nu,q,qbar = mode 
        
        alph = float(self.alph_s(m))

        qcd = 1 + alph/np.pi + 5.2*alph**2/np.pi**2 + 26.4*alph**3/np.pi**3
        

        xq = self.masses(pid(q))/m

        prefactor = (self.U['e']**2 + self.U['mu']**2 + self.U['tau']**2) * self.Gf**2 * m**5  / (32*np.pi**3)

        if m >= 2*self.masses(pid(q)) and m > 1.0:

            sw2 = self.sin2w

            if q  in self.particle_content['quarks']['up'] or qbar in self.particle_content['quarks']['up']:
                
                C1 = (1 - 8*sw2/3 + 32*sw2**2/9)/4
    
                C2 = (4*sw2/3 - 1)*sw2/3

            elif q  in self.particle_content['quarks']['down'] or qbar in self.particle_content['quarks']['down']:
               
                C1 = (1 - 4*sw2/3 + 8*sw2**2/9)/4
    
                C2 = (2*sw2/3 - 1)*sw2/6   
    
            
    
            if q in ['u','d','s'] or qbar in ['u','d','s']: 

               
                
                if q == 's' or qbar == 's': 
                    
                    kin = ( C1 *f1(xq) + C2 *f2(xq) )*qcd*np.sqrt(1-4*self.masses("321")**2 / m**2)
                    

                else: kin = ( C1 *f1(xq) + C2 *f2(xq) )*qcd
            
            else: 
                kin =  C1 *f1(xq) + C2 *f2(xq) 

            return prefactor*kin
        
        else: return 0
    
        
        
        
"""

Kinematic Functions

"""

def Lambda(x,y,z):
    
    return x**2 + y**2 + z**2 - 2*x*y - 2*y*z - 2*x*z

def L(x): 

    x = Decimal(x)

    arg = float( (1-3*x**2-(1-x**2)*np.sqrt(1-4*x**2))/(x**2*(1+np.sqrt(1-4*x**2))))

    return np.log(arg)

def f1(x):

    return (1-14*x**2-2*x**4-12*x**6)*np.sqrt(1-4*x**2) + 12*x**4*(x**4-1)*L(x)

def f2(x):

    return 4*( x**2*(2+10*x**2 - 12*x**4)*np.sqrt(1-4*x**2) + 6*x**4*(1-2*x**2+2*x**4)*L(x) )


def I_1(x,y,z, manual=True):
    
    #changed 1->s
    integrand = lambda s: (1/s)*(s - y - z)*(1 + x - s) * np.sqrt(Lambda(s,z,y)) * np.sqrt(Lambda(1,s,x))
    
    if manual:
        smin, smax, ns = (np.sqrt(y) + np.sqrt(z))**2, (1 - np.sqrt(x))**2, 100
        ds = (smax-smin)/float(ns)
        integral=0
        for s in np.linspace(smin+0.5*ds,smax-0.5*ds,ns):
            integral += integrand(s)
        integral*=ds
    else:
        integral,error = integrate.quad(integrand, (np.sqrt(y) + np.sqrt(z))**2, (1 - np.sqrt(x))**2)
    return 12*integral

 
def delta(l1,l2):
    
    if l1 == l2 or l1 == -1*l2:
        return 1
    else: 
        return 0   

"""

Additional Functions

"""
#returns the anti particle
def anti(x):
    
    if 'anti_' not in x: return 'anti_'+x
        
    elif 'anti_' in x:  return x.replace('anti_','')
    




pid_conversions = {    
        #Leptons
        'e': 11,
        'mu': 13, 
        'tau': 15,

        #Neutrinos
        've': 12,
        'vmu': 14,
        'vtau': 16,
        'nu':12,


        #Pseudos
        'pi+':211,
        'pi0':111,
        'K+':321,
        'eta':221,
        'etap':331,
        'D+': 411,
        'Ds+': 431,
    

        #Vectors
        'rho+':213,
        'rho0':113,
        'K+star':323,
        'omega':223,
        'phi':333,
    
        #Quarks
        'd': 1,
        'u':2,
        's':3,
        'c':4,
        'b':5,
        't':6,

        }


#returns the id given a particle
def pid(x):
      
    if 'anti_' not in x: return str(pid_conversions[x])
    
    elif 'anti_' in x:  return str(-1*pid_conversions[x.replace('anti_','')])
    

#returns the conjugate mode
def conjugate(x):
    
    conj_mode = []
    
    for p in x: conj_mode.append(anti(p))
        
    return tuple(conj_mode)
    
 