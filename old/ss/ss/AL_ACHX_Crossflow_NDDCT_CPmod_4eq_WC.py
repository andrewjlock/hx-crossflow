# -*- coding: utf-8 -*-
"""
Created on Tue May 22 11:37:04 2018

@author: Andrew Lock, with heavy use of code written by Ingo Jahn
""" 
import time
import numpy as np
import CoolProp as CPCP
from CoolProp.CoolProp import PropsSI as CP
import scipy as sci
from scipy import optimize
import matplotlib.pyplot as plt 
from getopt import getopt
from statistics import mean
import sys
from scipy.interpolate import interp1d

"""VARIABLE INPUTS--------------------------------------------------"""

# def variable_inputs(G,F,M):
  
#     G.HX_length = 14.4
#     G.n_rows = 4  # - Number of rows (deep) of heat exchanger
#     G.n_tubesPR = 2
#     G.pitch_longitudal = 0.05022 # - (m) Longitudal tube pitch 
#     G.pitch_transverse =  0.058  # - s(m) Transverse tube pitch
#     G.ID = 0.0234   # (m) ID of HX pipe
#     G.t_wall = 0.001 # (m) Thickness of pipe wall
#     G.dx_i = 0.7 # (m) Initial length of pipe discretisation
#     G.k_wall = 237 # (W/m) Thermal conductivity of pipe wall
#     G.D_fin = 0.05715  # (m) Maximum diameter of fin
#     G.pitch_fin = 0.0028 # (m) Fin pitch
#     G.t_fin = 0.0005 # (m) Mean thickness of pipe fin
#     G.k_fin = 230 # (m) 

#     M.Nu_CorrelationH = 3    # set correlation for heat transfer in H channel
#     M.alpha_CorrelationC = 4 # set correlation for heat transfer in C channel
#     M.f_CorrelationH = 0 # set correlation for friction factor in H channel
#     M.f_CorrelationC = [] # set correlation for friction factor in C channel 
#     M.solver_figure = 1 # (Updating solver residual graph: 1=ON, 0=OFF)
#     M.final_figure = 1

#     F.PF_fluid  = 'H2O'    	
#     F.Amb_fluid  = 'Air'
#     F.T_PF_in  = 273.15+71.1 # (K)
#     F.mdot_PF  = []# 0.37817 #158.9/(25*19) # (kg/s)
#     F.P_PF_in  = 7.96*10**6 # 7.96 * 10**6 # 7.96*10**6 # 7.96*10**6 # pressure (Pa)
#     F.TC_in  = 273.15+20 # (K)
#     #F.vC_in  = 1 # (m/s)
#     F.P_amb_in  = 101325 # pressure (Pa)
#     F.TC_outlet_guess = 273.15+45 # Air side outlet guess for initialisation (K)

"""--------------------MAIN FUNCTION -----------------------------------"""
    
def main(ACHX_inputs,mod=None,v=None,mdot=None,TC_in=None,T0i=[0],HX_L=None,fig_switch=0):
    start = time.time()
    
    # Import the class objects that incorporate the details of the model
    M = Model()
    F = Fluid()
    G = Geometry()

    ACHX_inputs(G,F,M) # Run the function that fills the classes with teh model details
    
    # This is the option for direct-input variables that are used for iterative scripts
    if v:
        F.vC_in = v
        print("Velocity argument  recieved")
    if mdot:
        F.mdot_PF = mdot
        print("PF mass flow rate argument recieved")
    if TC_in:
        F.TC_in = TC_in
        print("Air in temp recieved")
    if HX_L:
        G.HX_length = HX_L
        print("HX length recieved")

    if mod: # If the function exists, run the function that modifies one or more of the heat exchanger parameters. Useful for running a series of tests in which one parameter varies. 
        mod(F)
        mod(M)
        mod(G)

    # Create fluid property objects
    global EosPF
    EosPF = CPCP.AbstractState('HEOS',F.PF_fluid)

    if fig_switch == 1:
        M.solver_figure = 1
        M.final_figure = 1
    else:
        M.solver_figure = 0
        M.final_figure = 0


    #Initialise objects that contain geometry and fluid property details.
    G.micro_init()
    F.micro_init(G)

    # Call the inter_fs function which creates an interpolation matrix for the cold fluid (air) 
    I = inter_fs(F.TC_in,F.T_PF_in,F.P_amb_in,F.Amb_fluid)
   
   # Creates initial solution values from inputs or interpolation function
    if T0i[0]==0:
        T0 = get_T0_iter(G,F,M,I)
    else:
        T0 = T0i

    # Setting up iteration figure if switch is set to 1
    global iteration_figure_switch
    iteration_figure_switch = M.solver_figure
    if iteration_figure_switch == 1:
        plt.ion()
        global fig
        global ax
        fig, ax = plt.subplots()

    # Setting up iteration counter variables
    global iteration_no
    global iteration_count
    global e_max
    global e_av

    iteration_no = 0
    iteration_count = []
    e_max = []
    e_av = []
    args = G,F,M,I,0
    
    # Iteratively solves the set of equations for a solution within the tolerance f_tol
    sol = sci.optimize.newton_krylov(lambda T:equations(T,G,F,M,I,0),T0,method='lgmres',f_tol=1e-4,callback=callback_function)
    T = sol

    # Stops the counter and displays the solution time
    stop = time.time()
    runtime = stop-start
    print("Total runtime: ", runtime)

    # Extracts and prints the solution data from the variable string
    TH_final, TC_final, TW_final,DV = open_T(T,F.T_PF_in,F.TC_in,G.n_cells,G.n_rows,G.n_CPR,G.n_CP,G.n_passes,F.P_PF_in,F.PF_fluid)
    print("Air outlet temperature is: ",TC_final[0])
    print("Process fluid outlet temperature is: ",TH_final[G.n_cells])

    # Executes the equations one last time with the final solution to print figures extract other variables
    error, T, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, alpha_i_final, PH, dp_amb = equations(T,G,F,M,I,1)
    PF_dp = F.P_PF_in-PH[G.n_cells]
    print("Outlet process fluid pressure: ",PH[G.n_cells]/1000,"kPa, Differential: ",(PF_dp)/1000,"kPa")
    
    # Determine the average pressure loss of air over the heat exchanger
    dp_amb_total = sum(dp_amb)
    P_amb_out = F.P_amb_in - dp_amb_total
    print("Outlet air pressure: ",P_amb_out/1000,"kPa, Differential: ", dp_amb_total,'Pa')

    # Determine the energy difference of PF between inlete and outlet
    dE_PF = F.mdot_PF * (CP('H','T',TH_final[0],'P',PH[0],F.PF_fluid) - CP('H','T',TH_final[G.n_cells],'P',PH[G.n_cells],F.PF_fluid))

    # Determine the total energy gain of air over the heat exchanger by summing each cell
    i=0
    q_amb_out = 0 
    for i in range(G.n_CPR):
        qi = CP('H','T',TC_final[i],'P',P_amb_out,F.Amb_fluid)
        q_amb_out = q_amb_out + qi
    Q_amb = (q_amb_out - (G.n_CPR*CP('H','T',F.TC_in,'P',F.P_amb_in,F.Amb_fluid)))*F.mdot_C
    TC_out_ave = mean(TC_final[0:G.n_CPR])
    
    #Determine and print the discrepance of energy balance between PF and air
    deltaQ = Q_amb-dE_PF
    print("Heat rejected: ", (dE_PF/1000), " kW")
    print("Solution discrepance: ", deltaQ/1000, " kW, or: ", (deltaQ/dE_PF)*100, "%")

    # Display warning if energy balance between the two fluids is too large
    Q_PF = dE_PF
    if abs(deltaQ/Q_PF) > 0.001:
        print("Solution error. Suggest smaller element size")
  
    # Plot figure if switch set to 1
    if M.final_figure == 1:
        plot_generator(TH_final, TC_final, TW_final, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, T,G,F,M)
    
    return TC_out_ave, dp_amb_total, T, Q_PF, PF_dp, TH_final, dp_amb, alpha_i_final, 

""" ------------------PROPERTY CLASSES--------------------------------------------------------"""
  
class Model:
    # Create model detail classes  
    def __init__(self):
        self.Nu_CorrelationH = [] # set correlation for heat transfer in H channel
        self.alpha_CorrelationC = [] # set correlation for heat transfer in C channel
        self.f_CorrelationH = [] # set correlation for friction factor in H channel
        self.f_CorrelationC = [] # set correlation for friction factor in C channel    
        self.alpha_o = [] # (W/m2)
        self.iteration_no = 0 # Iteration counter
        self.solver_figure = [] # Switch for solver figure
        self.final_figure = [] # Switch for final figures
        self.consider_bends = [] # Switch for considering bends in pressure loss
        self.bend_loss_coefficient  = [] # Friction factor for pressure loss in bends

class Geometry:
    def __init__(self):
        self.dx_i = [] # number of cells
        self.k_wall = [] # thermal conductivity (W / m)
        self.HX_th_length = [] # (m) 
        self.n_rows = []
        self.n_passes = []
        self.pitch_longitudal = []   # - (m) Streamwise tube pitch 
        self.pitch_transverse = []   # - (m) Normal to stream tube pitch (X,Y)
        self.ID = []   # (m) ID of HX pipe
        self.t_wall = [] # (m) Pipe wall thickness
        self.pitch_fin = [] #(m) Pitch spacing of fins
        self.D_fin = [] # (m) Fin diameter
        self.t_fin = [] # (m) Fin thickness
        self.k_fin = [] # (W/m) Fin matieral thermal conductivity
        self.n_tubesPR = [] # Number of tubes per row
        
    def micro_init(self):
        self.HX_th_length =  self.HX_length   # - (m) Length of heat exchanger (axial pipe direction)
        self.n_CPR= int(self.HX_th_length//self.dx_i+1)
        self.n_CP = int(self.n_rows/self.n_passes) # Number of adjacent co-flow rows
        if self.n_rows/self.n_passes != 0:
            print("Error in pipe arrangement")
            MyError('Incorrect pipework arrangement')
        self.n_cells = self.n_CPR * self.n_rows
        print("Number of cells:",self.n_cells)
        self.dx = self.HX_th_length/self.n_CPR   
        self.A_amb = self.pitch_transverse*self.dx
        self.A_CS = np.pi * (self.ID**2)/4 # (m2) Internal cross sectional area of pipe
        self.OD = self.ID + (2*self.t_wall)
        self.A_WCS = np.pi * (1/4) * ((self.OD**2)-(self.ID**2))
        self.dA_o = float(np.pi * self.OD * self.dx ) 
        self.dA_i = float(np.pi * self.ID * self.dx ) 
        self.n_phi = ((self.D_fin/self.OD)-1)*(1+(0.35*(np.log(self.D_fin/self.OD)))) # Equation 3.3.13 fro Kroger
        self.bend_path_length = np.pi * self.pitch_transverse /2
        if self.D_fin:
            self.A_f = (self.dx/self.pitch_fin)*((((self.D_fin**2)-(self.OD**2))*np.pi/2)+self.D_fin*self.t_fin) # Area of the fins only
            self.A_r = (self.dA_o - ((self.dx/self.pitch_fin)*self.t_fin*self.OD*np.pi)) # Area of the root between fins
            self.A_ft = self.A_f + self.A_r # Total exposed area
            self.A_FTCS = (self.OD*self.dx) + ((self.dx/self.pitch_fin)*self.t_fin*(self.D_fin-self.OD)) # Total maximum cross-section blokage
            self.A_c = self.A_amb - self.A_FTCS # Mininum free flow area
            self.H_f = 0.5*(self.D_fin-self.OD) # fin height
            self.A_rat = (self.A_f + self.A_r)/self.dA_o # Ratio of exposed area to root area

class Fluid:
    def __init__(self):
        self.PF_fluid  = []
        self.Amb_fluid  = []
        self.T_PF_in  = [] # (K)
        self.mdot_PF  = [] # (kg/s)
        self.P_PF_in  = [] # pressure (Pa)
        self.TC_in  = [] # (K)
        self.vC_in  = [] # (m/s)
        self.P_amb_in  = [] # pressure (Pa)
        #self.T0 = []

    def micro_init(self,G):
        self.Vdot_C = G.A_amb * self.vC_in # (m3/s) air volumetric flow rate per cell
        self.mdot_C = self.Vdot_C * CP('D','P',self.P_amb_in,'T',self.TC_in,self.Amb_fluid)
        self.mdot_amb_max = self.mdot_C/G.A_c 


class inter_fs:
    # This function creates an interpolation matrix of fluid properties for the cold (air) side
    def __init__(self,T_l,T_h,P,fluid):
        n=1000
        T = np.linspace(T_l,T_h,n)
     
        V = CP('V','P',P,'T',T[0:n],fluid)
        D = CP('D','P',P,'T',T[0:n],fluid)
        H = CP('H','P',P,'T',T[0:n],fluid)
        K = CP('CONDUCTIVITY','P',P,'T',T[0:n],fluid)
        C = CP('C','P',P,'T',T[0:n],fluid)
        Pr = CP('PRANDTL','P',P,'T',T[0:n],fluid)

        self.mu_f = interp1d(T,V,bounds_error=False,fill_value="extrapolate")
        self.rho_f = interp1d(T,D,bounds_error=False,fill_value="extrapolate")
        self.h_f = interp1d(T,H,bounds_error=False,fill_value="extrapolate")
        self.k_f = interp1d(T,K,bounds_error=False,fill_value="extrapolate")
        self.cp_f = interp1d(T,C,bounds_error=False,fill_value="extrapolate")
        self.Pr_f = interp1d(T,Pr,bounds_error=False,fill_value="extrapolate")

""" ----------------------------------------------------------------------------------------"""

def callback_function(x,r):
    # Function to recieve and print residuals during solver operation 
    global iteration_no
    global e_max
    global e_av
    global iteration_figure_switch
    e_max.append(max(np.absolute(r)))
    e_av.append(np.average(np.absolute(r)))
    iteration_no = iteration_no + 1
    iteration_count.append(iteration_no)
    print("Iteration number: ",iteration_no)
    print("Maximum residual: ", e_max[iteration_no-1], "Average residual: ", e_av[iteration_no-1])

    if  iteration_figure_switch == 1:
        plt.cla()
        plt.xlabel("Iteration number")
        plt.ylabel("Residual")
        ax.set_yscale('log')
        it_line = ax.plot(iteration_count,e_max,label="Maximum")
        it1_line = ax.plot(iteration_count,e_av,label="Average")
        plt.legend(loc=1)
        plt.pause(0.001)
        plt.draw()

def get_T0_iter(G,F,M,I):
    # Function to get initial values for iterative solver 
    try: 
        h_amb_in =  CP('H','T',F.TC_in,'P',F.P_amb_in,F.Amb_fluid)
        h_PF_in =  CP('H','T',F.T_PF_in,'P',F.P_PF_in,F.PF_fluid)
        E_amb_in = h_amb_in * F.mdot_C * G.n_CPR 
        E_PF_in = h_PF_in * F.mdot_PF
        TC_out_guess = F.TC_outlet_guess
        error = 1
        j = 0
        while error > 0.01:
            j = j+1
            h_amb_out =  CP('H','T',TC_out_guess,'P',F.P_amb_in,F.Amb_fluid)
            dh_amb = (h_amb_out-h_amb_in)
            dh_PF = dh_amb*F.mdot_C*G.n_CPR/F.mdot_PF
            h_PF_out = h_PF_in - dh_PF
            T_PF_out = CP('T','H',h_PF_out,'P',F.P_PF_in,F.PF_fluid) # PF temp out from enthalpy balance
            T_PF_av = ((4*T_PF_out) + F.T_PF_in)/5
            T_amb_av = 0.5*(F.TC_in+TC_out_guess) # bulk temperature
            Pr_amb = CP('PRANDTL','P',F.P_amb_in,'T', T_amb_av ,F.Amb_fluid)
            rho_amb = CP('DMASS','P',F.P_amb_in,'T', T_amb_av ,F.Amb_fluid)
            mu_amb = CP('VISCOSITY','P',F.P_amb_in,'T', T_amb_av ,F.Amb_fluid)
            Re_amb = (F.mdot_C/G.A_c) * G.OD / mu_amb
            alpha_o = calc_alpha_amb(G,F,I,Re=Re_amb,Pr=Pr_amb,Tb=T_amb_av,rho_b=rho_amb,Correlation=2)
            b = ((2*alpha_o)/(G.k_fin*G.t_fin))**0.5
            n_f = np.tanh(b*G.OD*G.n_phi/2)/(b*G.OD*G.n_phi/2)
            q = -(T_amb_av - T_PF_av)*alpha_o*G.A_ft*n_f*G.n_cells
            T_PF_out1 = CP('T','H',h_PF_in - (q/F.mdot_PF),'P',F.P_PF_in,F.PF_fluid) # PF temp out from convective heat trasnfer balance
            error = abs(T_PF_out1 - T_PF_out)
            error1 = T_PF_out1 - T_PF_out
            TC_out_guess = TC_out_guess - (0.05*error1)
            if j>1000:
                raise ValueError()
            print("Initialisation iteration number: ", j)
        print("Initialisation complete")
        print("Intiial process fluid outlet temperature: ",T_PF_out)
        print("Initial ambient fluid outlet temperature:", TC_out_guess)

        T0 = np.zeros(3*G.n_cells)
        # Initial process fluid temperatures 
        T0[0:G.n_cells] = F.T_PF_in - (np.arange(G.n_cells)+1)/float(G.n_cells)*(F.T_PF_in - T_PF_out) # Assumes temperature does not drop below critical temperature
        # Initial ambient fluid temperatures
        for i in range(G.n_rows):
            T0[G.n_cells+(i*G.n_CPR):G.n_cells+((i+1)*G.n_CPR)] = F.TC_in + ((TC_out_guess-F.TC_in)*((G.n_rows -i)/G.n_rows))
        # Initial pipe wall temperatures
        T0[2*G.n_cells:3*G.n_cells] =  T0[0:G.n_cells] - ((T0[0:G.n_cells]-T0[G.n_cells:2*G.n_cells])*(alpha_o/5000)*(n_f*G.A_ft/G.dA_i))
       
        return T0
    except ValueError:
        print("Default to simple initialisation")
        T0 = np.zeros(3*G.n_cells) 
        T0[0:G.n_cells] = F.T_PF_in - (np.arange(G.n_cells)+1)/float(G.n_cells)*(F.T_PF_in - F.TC_outlet_guess)
        for i in range(G.n_rows):
            T0[G.n_cells+(i*G.n_CPR):G.n_cells+((i+1)*G.n_CPR)] = F.TC_in + ((F.TC_outlet_guess-F.TC_in)*((G.n_rows -i)/G.n_rows))
        T0[2*G.n_cells:3*G.n_cells] =  T0[0:G.n_cells]
        return T0

def open_T(T,T_PF_in,TC_in,n_cells,n_rows,n_CPR,n_CP,n_passes,P_PF_in,PF_fluid):
    """
    function to unpack the Temperature vector T into the 6 vectors
    TH, TWH, TWC, TC, PH, PC
    """
    TH = np.zeros(n_cells+n_rows)
    TC = np.zeros(n_cells+n_CPR) 
    TW = np.zeros(n_cells)
    
    for i1 in range(n_CP):
        TH[i1*(n_CPR+1)] = T_PF_in
    T_head = []
    dir_vec = []
    for i2 in range(n_passes):
        T_end = []
        for i3 in range(n_CP):

            row_no = (i2*n_CP)+i3
            start_n = row_no*(n_CPR+1)
            end_n = row_no*(n_CPR+1) + n_CPR +1

            if i2%2 == 0:
                dir_vec.append(1)
                TH[start_n+1:end_n] = T[row_no*(n_CPR):((row_no+1)*(n_CPR))]
                T_end.append(TH[end_n-1])
                if i2>0:
                    TH[start_n] = T_head[i2-1]
                    dir_vec.append(1)
            if i2%2 == 1:
                dir_vec.append(-1)
                TH[start_n:end_n-1] = T[row_no*(n_CPR):((row_no+1)*(n_CPR))]
                T_end.append(TH[start_n])
                if row_no<n_rows:
                    TH[end_n-1] = T_head[i2-1]
        h_c = []
        for c in range(n_CP):
            h_c.append(GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P_PF_in,T_end[c]],[CPCP.iHmass]))
        h_ave = sum(h_c)/n_CP
        T_head.append(CP('T','H',h_ave,'P',P_PF_in,PF_fluid))
    
    TC[0:n_cells] = T[n_cells:(2*n_cells)]
    TC[n_cells:n_cells+n_CPR] = TC_in
    TW[0:n_cells] = T[2*n_cells:3*n_cells]

    return TH,TC,TW,dir_vec

def calc_Nu(G,F,Re=0,Pr=0,P=0,Tb=0,Tw=0,rho_b=0,rho_w=0,Correlation=0,K_c=0 ,mu_b = 0):
    """ Function to return Nusselt number for internal pipe fluid flow """
    if Correlation == 1:
        Nu = 0.14 * Re**0.69 * Pr**0.66 # Yoon et al correlation for Nu based on bulk temperature only

    if Correlation == 2:  # Yoon et al corelation that incorporates wall temperature
        
        h_b = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P,Tb],[CPCP.iHmass])
        h_w = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P,Tw],[CPCP.iHmass])
        Cp_w = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P,Tw],[CPCP.iCpmass]) 
        mu_w = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P,Tw],[CPCP.iviscosity])
        k_w = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P,Tw],[CPCP.iconductivity])
        Pr_w = mu_w * Cp_w / k_w
        Re_w = Re * (mu_b/mu_w) * (rho_w/rho_b)
        Cp_bar = (h_w-h_b)/(Tw-Tb)
        f = ((0.79*np.log(Re))-1.64)**-2
        
        Nu_w = ((f/8)*(Re_w)*Pr_w)/(1.07+(12.7*((f/8)**0.5)*((Pr_w**(2/3))-1)))
        Nu = 1.38 * Nu_w * ((Cp_bar/Cp_w)**0.86)* ((rho_w/rho_b)**0.57)

    if Correlation == 3: # Gneilinski
        f = ((0.79*np.log(Re))-1.64)**-2
        Nu = ((f/8)*(Re-1000)*Pr)/(1.07+(12.7*((f/8)**0.5)*((Pr**(2/3))-1))) 

    if Correlation == 4: # Pitla et al
        Cp_w = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P,Tw],[CPCP.iCpmass])
        mu_w = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P,Tw],[CPCP.iviscosity])
        k_w = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P,Tw],[CPCP.iconductivity])
        k_b = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P,Tb],[CPCP.iconductivity])
        Pr_w = mu_w * Cp_w / k_w
        Re_w = Re * (mu_b/mu_w) * (rho_w/rho_b)
        f = ((0.79*np.log(Re))-1.64)**-2
        Nu_b = ((f/8)*(Re-1000)*Pr)/(1.07+(12.7*((f/8)**0.5)*((Pr**(2/3))-1))) 
        Nu_w = ((f/8)*(Re_w)*Pr_w)/(1.07+(12.7*((f/8)**0.5)*((Pr_w**(2/3))-1)))
        Nu = ((Nu_w + Nu_b)/2)*(k_w/k_b)
    
    if Correlation == 5: #Wang et al (UQ)
        Tf = (Tb+Tw)/2
        h_b = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P,Tb],[CPCP.iHmass])
        h_w = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P,Tw],[CPCP.iHmass])
        Cp_bar = (h_w-h_b)/(Tw-Tb)
        Cp_f = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P,Tf],[CPCP.iCpmass])
        Cp_w = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P,Tw],[CPCP.iCpmass])
        mu_f = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P,Tf],[CPCP.iviscosity])
        k_f = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P,Tf],[CPCP.iconductivity])
        Re_f = Re * (mu_b/mu_f)
        f_f = (0.79*np.log(Re_f)-1.64)**-2
        Pr_f = mu_f * Cp_f / k_f
        Nu_iso = ((f_f/8)*(Re-1000)*Pr_f)/(1.07+(12.7*((f_f/8)**0.5)*((Pr_f**(2/3))-1))) 
        Nu = 1.2791 * Nu_iso * ((rho_w/rho_b)**-0.0541) * ((Cp_bar/Cp_w)**-0.0441)

    return Nu

def calc_alpha_amb(G,F,I,Re=0,Pr=0,Tb=0,rho_b=0,Correlation=0,K_c=0):
    """ Function to return air-side finned tube heat transfer coefficient """
    mu = I.mu_f(Tb) 
    k = I.k_f(Tb) 

    if Correlation == 1: # Briggs and Young finned tube correlation
        Re = (F.mdot_C/G.A_c) * G.OD / mu
        Nu = 0.134*(Pr**0.33)*(Re**0.681)*((2*(G.pitch_fin - t_fin)/(G.D_fin - G.OD))**0.2)*(((G.pitch_fin-G.t_fin)/G.t_fin)**0.1134)
        alpha = Nu*k/G.OD

    if Correlation == 2: # Gaugouli finned tube correlation
        Re = (F.mdot_C/G.A_c) * G.OD / mu
        Nu = 0.38 * (Re**0.6) * (Pr**(1/3))*((G.A_rat)**-0.15)
        alpha = Nu*k/G.OD

    if Correlation == 3: # From ASPEN HTFS3-AC
        cp = I.cp_f(Tb) 
        U = F.mdot_C/(rho_b*G.A_amb)
        U_f = K_c * U
        Re_f = rho_b * U_f * G.OD/mu
        u_max = F.mdot_C / (rho_b * G.A_c)
        Re_max = u_max * rho_b * G.OD / mu
        j = 1.207 * (Re_f**0.04) * (Re_max**-0.5094)*(G.A_rat**-0.312)
        alpha = j * cp * F.mdot_amb_max * (Pr**(-2/3))

    if Correlation == 4:
        cp = I.cp_f(Tb) 
        U = F.mdot_C/(rho_b*G.A_amb)
        U_f = K_c * U
        Re_f = rho_b * U_f * G.OD/mu
        u_max = F.mdot_C / (rho_b * G.A_c)
        Re_max = u_max * rho_b * G.OD / mu
        j = 0.29 * (Re_max**-0.367)*(G.A_rat**-0.17)
        alpha = j * cp * F.mdot_amb_max * (Pr**(-2/3))
    return alpha 

def calc_f(Re, P, Tm, Tp, TW, mdot, A, Dh, fluid, Correlation, q,rho_b,rho_w,mu_b, epsilon = 0):
    """ Function to return friction factor for internal pipe flow """
    if Correlation == 0:
        f = 0

    if Correlation == 1:
        f = (0.79*np.log(Re) - 1.64)**-2 # Laminar pipe flow

    if Correlation == 2: 
        mu_w = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P,TW],[CPCP.iviscosity]) 
        s = 0.023 * (abs(q/mdot*A)**0.42)
        f = (((1.82*np.log(Re))-1.64)**-2)*(rho_w/rho_b)*((mu_w/mu_b)**s)

    if Correlation == 3: # Blasius fricion factor
        if Re < 2e4:
            f = 0.316 * Re**(-1/4)
        else:
            f = 0.184 * Re**(-1/5)
    return f

def calc_air_dp(G,n_rows,rho,mu):
    """ Function to return pressure differential per row for air through finned tube banks, from ASPEN HTRI """
    K_tube = 4.75 * n_rows * G.pitch_longitudal * ((mu/rho)**0.3)*(((G.pitch_transverse/G.OD)-1)**-1.86)*(G.OD**-1.3)
    phi = np.pi * ((G.D_fin**2)- (G.OD**2))*(1/G.pitch_fin)*n_rows / (2*G.D_fin)
    B = G.A_FTCS/G.dx
    tau =  G.D_fin / (G.D_fin-B)
    K_fins = 0.0265 * phi * (tau**1.7)
    K_ft = K_tube + K_fins
    N_G = (G.n_rows-1)/G.n_rows
    G_D = (((G.pitch_longitudal**2)+((0.5*G.pitch_transverse)**2))**0.5) - G.D_fin
    G_T = G.pitch_transverse - G.D_fin
    G_A = 0.5*(G.D_fin-G.OD)
    GR_eff = (G_D + G_A)/G_T
    theta = np.arctan(0.5*G.pitch_transverse/G.pitch_longitudal)
    K_gap = N_G * theta * GR_eff
    K_B = K_ft / (((G.D_fin/G.pitch_transverse)+(((K_ft/K_gap)**(1/1.7))*(1-(G.D_fin/G.pitch_transverse))))**1.7)
    return K_B,K_ft

def equations(T,G,F,M,I,flag):
    """ Equations to solve energy balance for each cell """
    error = np.zeros(3*G.n_cells)
    Q1 = []; Q2 = []; Q3 = []; Q4 = [] 
    Q5 = []; Q6 = []; Q7 = []; Q8 = []; Alpha_i=[] 
    PH = [];
    PH.append(F.P_PF_in)

    TH,TC,TW,DV = open_T(T,F.T_PF_in,F.TC_in,G.n_cells,G.n_rows,G.n_CPR,G.n_CP,G.n_passes,F.P_PF_in,F.PF_fluid)
    # Calculate the pressure drop and loss coefficients across each row based on mean air properties for that row. Based off ASPEN HTSF3
    K_c = []
    dp_amb = []
    inlet_cells = []
    outlet_cells = []
    header_inlet_cells = []
    header_outlet_cells = []

    for x in range(G.n_CP):
        inlet_cells.append(x*G.n_CPR)
        outlet_cells.append(G.n_cells - (x*G.n_CPR))
        for y in range(G.n_passes-1):
            header_inlet_cells.append((G.n_CPR*x)+(y*G.n_CPR*G.n_CP)-1+G.n_CPR)
            header_outlet_cells.append(((y+1)*G.n_CPR*G.n_CP)+(x*G.n_CPR))

    for k in range(G.n_rows):
        rho_k = 0.25*(I.rho_f(TC[k*G.n_CPR])  + I.rho_f(TC[((k+1)*G.n_CPR)-1])  + \
               I.rho_f(TC[(k+1)*G.n_CPR]) + I.rho_f(TC[((k+2)*G.n_CPR)-1])) 
        
        mu_k = 0.25*(I.mu_f(TC[k*G.n_CPR])  + I.mu_f(TC[((k+1)*G.n_CPR)-1])  + \
               I.mu_f(TC[(k+1)*G.n_CPR]) + I.mu_f(TC[((k+2)*G.n_CPR)-1]))
        
        U_k = F.mdot_C/(rho_k * G.A_amb)
        K_B,K_ft = calc_air_dp(G,1,rho_k,mu_k)
        K_c.append((K_B/K_ft)**(1/1.7))
        dp_amb.append(1.066 * K_B*rho_k*(U_k**1.65))
    q_pre = 0
    
    for i in range(G.n_cells): 
        
        row = abs(i-1)//G.n_CPR

        # Amb node [i] is the outlet to the current cell
        node_amb_i1=i+G.n_CPR # Ambient fluid inlet node to current cell
        node_amb_i2= i+(2*G.n_CPR) # Ambient node two prior to current cell
        node_amb_i3 = i - G.n_CPR # Ambient node following current cell

        k_PF = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[i],(0.5*(TH[i+row]+TH[i+1+row]))],[CPCP.iconductivity]) 
        km_PF = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[i],TH[i+row]],[CPCP.iconductivity])  
        kp_PF = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[i],TH[i+1+row]],[CPCP.iconductivity])  
        k_amb = I.k_f(0.5*(TC[i]+TC[node_amb_i1]))  
        km_amb = I.k_f(TC[node_amb_i1])  
        kp_amb = I.k_f(TC[i])  

        Tb_PF = 0.5*(TH[i+row]+TH[i+1+row]) # bulk temperature
        Tb_amb = 0.5*(TC[node_amb_i1]+TC[i]) # bulk temperature
        Pr_PF = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[i],Tb_PF],[CPCP.iPrandtl]) 
        Pr_amb = I.Pr_f(Tb_amb)  
        rho_PF_b = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[i],Tb_PF],[CPCP.iDmass]) 
        rho_PF_w = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[i],TW[i]],[CPCP.iDmass]) 
        rho_amb = I.rho_f(Tb_amb) 
        U_PF = abs(F.mdot_PF / (rho_PF_b * G.A_CS))
        mu_PF = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[i],Tb_PF],[CPCP.iviscosity]) 
        Re_PF = rho_PF_b * U_PF * G.ID / mu_PF
        Nu_PF = calc_Nu(G,F,Re=Re_PF,Pr=Pr_PF,P=PH[i],Tb=Tb_PF,Tw=TW[i],rho_b=rho_PF_b,rho_w=rho_PF_w,Correlation=M.Nu_CorrelationH,mu_b = mu_PF)
        alpha_o = calc_alpha_amb(G,F,I,Pr=Pr_amb,Tb=Tb_amb,rho_b=rho_amb,Correlation=M.alpha_CorrelationC,K_c=K_c[row])
        alpha_i = Nu_PF * k_PF / G.ID
        b = ((2*alpha_o)/(G.k_fin*G.t_fin))**0.5
        n_f = np.tanh(b*G.OD*G.n_phi/2)/(b*G.OD*G.n_phi/2)
        
        f_PF = calc_f(Re_PF,PH[i],TH[i+row],TH[i+1+row],TW[i],F.mdot_PF,G.A_CS,G.ID,F.PF_fluid,M.f_CorrelationH,q_pre/G.dA_i,rho_PF_b,rho_PF_w,mu_PF)
        
        if M.consider_bends == 1 and i%(G.n_CPR) == 0 and i != 0:
            dp_PF_friction = rho_PF_b * f_PF * ((G.dx + G.bend_path_length)/G.ID) * (U_PF**2) / 2
            dp_PF_bend = rho_PF_b * M.bend_loss_coefficient * (U_PF**2) / 2
            dp_PF = dp_PF_friction + dp_PF_bend
        else:
            dp_PF = rho_PF_b * f_PF * (G.dx/G.ID) * (U_PF**2) / 2 # calculate friction head loss (Darcy Weisbach formula)
        PH.append(PH[i] - dp_PF)

        # Convective fluid-wall heat trasnfer hot side
        q3 = -(TW[i] - ((TH[i+row]+TH[i+1+row])/2))*alpha_i*G.dA_i
        q_pre = q3
        # Convective fluid-wall heat transfer cold side
        q4 = -(((TC[node_amb_i1]+TC[i])/2) - TW[i])*alpha_o*((G.A_f*n_f)+(G.A_r))

        # PF side bulk convection
        q1_conv = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[i],TH[i+row]],[CPCP.iHmass]) * F.mdot_PF * DV[row]
        q2_conv = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[i+1],TH[i+1+row]],[CPCP.iHmass]) * F.mdot_PF * DV[row]

        if i in inlet_cells or header_outlet_cells:
            q1_cond = - km_PF * G.A_CS/ G.dx/2 * ( 0.5*(TH[i+row] + TH[i+1+row]) - TH[i+row] )
        else:
            q1_cond = - km_PF * G.A_CS/  G.dx * ( 0.5*(TH[i+row] + TH[i+1+row]) - 0.5*(TH[i+row] + TH[i-1+row]) )

        if i in outlet_cells or header_inlet_cells:
            q2_cond = - kp_PF * G.A_CS/ G.dx/2 * (TH[i+1+row]- 0.5*(TH[i+row] + TH[i+1+row]) ) 
        else:
            q2_cond = - kp_PF * G.A_CS/ G.dx * ( 0.5*(TH[i+1+row] + TH[i+2+row]) - 0.5*(TH[i+row] + TH[i+1+row]) )

        q1 = q1_cond + q1_conv
        q2 = q2_cond + q2_conv

        # Amb side bulk convection
        q5_conv = I.h_f(TC[node_amb_i1]) * F.mdot_C 
        q6_conv = I.h_f(TC[i]) * F.mdot_C 

        if i >= (G.n_cells - G.n_CPR):
             q5_cond = - km_amb * G.A_amb/ G.pitch_longitudal/2 * ( 0.5*(TC[node_amb_i1] + TC[i]) - TC[node_amb_i1] )
        else: 
             q5_cond = - km_amb * G.A_amb/ G.pitch_longitudal * ( 0.5*(TC[node_amb_i1] + TC[i]) - 0.5*(TC[node_amb_i1] + TC[node_amb_i2] ))

        if i < G.n_CPR:
            q6_cond = - km_amb * G.A_amb/ G.pitch_longitudal/2 * (TC[i] - 0.5*(TC[node_amb_i1] + TC[i]))
        else:
            q6_cond = - km_amb * G.A_amb/ G.pitch_longitudal * (0.5*(TC[i]+TC[node_amb_i3]) - 0.5*(TC[node_amb_i1] + TC[i]))

        # Wall conduction 
        if i in inlet_cells or header_outlet_cells:
            q7 = 0
        else: 
            q7 = -(TW[i] - TW[i-1])*G.k_wall*G.A_WCS/G.dx
            
        if i in outlet_cells or header_inlet_cells:
            q8 = 0
        else:
            q8 = -(TW[i+1] - TW[i])*G.k_wall*G.A_WCS/G.dx

        q5 = q5_cond + q5_conv
        q6 = q6_cond + q6_conv

        error[i] = (q1 - q2 - q3)
        error[G.n_cells + i] = (q4+q5-q6)
        error[2*G.n_cells + i] = (q3 - q4 + q7 - q8)
        if flag ==1:
            Q1.append(q1); Q2.append(q2); Q3.append(q3); 
            Q4.append(q4); Q5.append(q5);  Q6.append(q6); 
            Q7.append(q7); Q8.append(q8); Alpha_i.append(alpha_i)
    if flag==0:
        return error 
    else:
        return error, T, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Alpha_i, PH, dp_amb

""" --------------------------PLOTTING------------------------------------------------------------- """

def plot_HX(TH,TC,TW,G,F):
    TH_av = [];
    i=0
    for i in range(G.n_cells):
        TH_av.append((TH[i] + TH[i+1])/2)

    TC_in_unsorted = TC[G.n_CPR:G.n_cells+G.n_CPR]
    TC_out = TC[0:G.n_cells]
    TC_in_sorted = np.zeros(G.n_cells)

    i=0
    for i in range(G.n_rows):
        TC_in_sorted[i*G.n_CPR:(i+1)*G.n_CPR] = list(reversed(TC_in_unsorted[i*G.n_CPR:(i+1)*G.n_CPR]))
    fig1 = plt.figure()
    plt.plot(np.linspace(0,1.,num=G.n_cells),TH_av,label="Process Fluid")
    plt.plot(np.linspace(0,1.,num=G.n_cells),TC_in_sorted,label="Air in")
    plt.plot(np.linspace(0,1.,num=G.n_cells),TC_out,label="Air out")
    plt.plot(np.linspace(0,1.,num=G.n_cells),TW,'--',label="Pipe Wall") 

    plt.ylabel('Temperature (K)')
    plt.xlabel('Position along Heat Exchanger (normalised)')
    plt.title('Temperature Distributions in Heat Exchanger')
    plt.legend(loc=1)
    fig1.show()

def plot_HXq(Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, N_cell):
    fig2 = plt.figure()
    plt.plot(np.linspace(0,1.,num=N_cell)            ,np.array(Q1)/1.e3,label="q1")
    plt.plot(np.linspace(0,1.,num=N_cell)           ,np.array(Q2)/1.e3,label="q2")
    plt.plot(np.linspace(0,1.,num=N_cell)           ,np.array(Q3)/1.e3,label="q3")
    plt.plot(np.linspace(0,1.,num=N_cell)              ,np.array(Q4)/1.e3,label="q4")
    plt.plot(np.linspace(0,1.,num=N_cell)       ,np.array(Q5)/1.e3,label="q5")
    plt.plot(np.linspace(0,1.,num=N_cell)       ,np.array(Q6)/1.e3,label="q6")
    plt.plot(np.linspace(0,1.,num=N_cell)       ,np.array(Q7)/1.e3,label="q7")
    plt.plot(np.linspace(0,1.,num=N_cell)       ,np.array(Q8)/1.e3,label="q8")
    plt.ylabel('Heat Flow (kW)')
    plt.xlabel('Position along Heat Exchanger (normalised)')
    plt.title('Energy Fluxes in Heat Exchanger')
    plt.legend(loc=1)
    fig2.show()

def plot_generator(TH_final, TC_final, TW_final,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,T,G,F,M):
        print("Plotting results")
        plot_HX(TH_final,TC_final,TW_final,G,F)
        plot_HXq(Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, G.n_cells)

def GetFluidPropLow(Eos,state,props):
    """
    Low level fluid property call.
    Uses CoolProp low level interface for all supported fluids.
    Computes Shell Thermia D properties using 'HTFOilProps'

    Inputs:
    Eos   - Equation of state object for desired fluid.
    state - 1x3 list that specifies fluid state according to CoolProp
            low level syntax. Takes the form
            [InputPairId,Prop1Value,Prop2Value], for example
            state = [CP.PT_INPUTS,20e6,273.15+150].
    props - List specifying required output properties in CoolProp low
            level syntax. For example, props = [CP.iPrandtl,CP.iDmass]
            will give Prandtl number and density.

    Outputs:
    outputProps - Array containing desired output properties.

    Notes:
    Currently only supports pressure & temperature input pairs.
    """
    try:
        Eos.update(*state)
        outputProps = [Eos.keyed_output(k) for k in props]
        if len(outputProps) == 1:
            outputProps = outputProps[0] 
        return outputProps
    except:
        print("state",state)
        print("props:",props)
        MyError('Invalid fluid specified.')


#shortOptions = ""
#longOptions = ["help", "job=", "noprint"]

class MyError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

if __name__ == "__main__":
#    userOptions = getopt(sys.argv[1:], shortOptions, longOptions)
    uoDict = dict(userOptions[0])


    