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
    
def main(ACHX_inputs,mod=None,v=None,TC_in=None,T0i=[0],HX_L=None,fig_switch=0):
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
    error, T, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, alpha_i_final, PH, dp_amb = equations(T,G,F,M,I,1)
    TH_final, TC_final, TW_H_final, TW_C_final,DV,mdot_vec = open_T(T,F.T_PF_in,F.TC_in,G.n_cells,G.n_rows,G.n_CPR,G.n_CP,G.n_passes,F.P_PF_in,F.PF_fluid)
    h_c = []
    if G.n_passes%2==0:
        for i in range(G.n_CP):
            print(len(TH_final) - ((i+1)*(G.n_CPR+1)))
            T_row_out = TH_final[len(TH_final) - ((i+1)*(G.n_CPR+1))]
            print(T_row_out)
            h_c.append(GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,F.P_PF_in,T_row_out],[CPCP.iHmass]))
        h_ave = sum(h_c)/G.n_CP
        T_out = (CP('T','H',h_ave,'P',F.P_PF_in,F.PF_fluid))
        P_out = PH[len(TH_final) - ((i+1)*(G.n_CPR+1))]
    else:
        for i in range(G.n_CP):
            print(TH_final)
            print(len(TH_final) - ((i)*(G.n_CPR+1)))
            T_row_out = TH_final[len(TH_final) - ((i)*(G.n_CPR+1))-1]
            print(T_row_out)
            h_c.append(GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,F.P_PF_in,T_row_out],[CPCP.iHmass]))
        h_ave = sum(h_c)/G.n_CP
        T_out = (CP('T','H',h_ave,'P',F.P_PF_in,F.PF_fluid))
        P_out = PH[len(TH_final) - ((i)*(G.n_CPR+1))-1]


    print("Air outlet temperature is: ",TC_final[0])
    print("Process fluid outlet temperature is: ",T_out)
    print("Mass flow rate vector is:",mdot_vec)

    # Executes the equations one last time with the final solution to print figures extract other variables
    PF_dp = F.P_PF_in-P_out
    print("Outlet process fluid pressure: ",P_out/1000,"kPa, Differential: ",(PF_dp)/1000,"kPa")
    
    # Determine the average pressure loss of air over the heat exchanger
    dp_amb_total = sum(dp_amb)
    P_amb_out = F.P_amb_in - dp_amb_total
    print("Outlet air pressure: ",P_amb_out/1000,"kPa, Differential: ", dp_amb_total,'Pa')

    # Determine the energy difference of PF between inlete and outlet
    mdot_total = 0
    for i in range(G.n_CP):
        mdot_total = mdot_total + mdot_vec[i]

    dE_PF = G.n_CP * mdot_total * (CP('H','T',TH_final[0],'P',PH[0],F.PF_fluid) - CP('H','T',T_out,'P',PH[G.n_cells],F.PF_fluid))

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
        plot_generator(TH_final, TC_final, TW_H_final, TW_C_final, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, T,G,F,M)
    
    return TC_out_ave, dp_amb_total, T, Q_PF, PF_dp, TH_final, dp_amb, alpha_i_final, TW_H_final, TW_C_final, Q3

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
        self.solver_type = []   # Swtich for solver type  - (0) for pressure differential input, (1) for total mass flow input 

class Geometry:
    def __init__(self):
        self.dx_i = [] # number of cells
        self.k_wall = [] # thermal conductivity (W / m)
        self.HX_th_length = [] # (m) 
        self.n_rows = []
        self.n_passes =[]
        self.pitch_longitudal = []   # - (m) Streamwise tube pitch 
        self.pitch_transverse = []   # - (m) Normal to stream tube pitch (X,Y)
        self.ID = []   # (m) ID of HX pipe
        self.t_wall = [] # (m) Pipe wall thickness
        self.pitch_fin = [] #(m) Pitch spacing of fins
        self.D_fin = [] # (m) Fin diameter
        self.t_fin = [] # (m) Fin thickness
        self.k_fin = [] # (W/m) Fin matieral thermal conductivity
        
    def micro_init(self):
        self.HX_th_length =  self.HX_length  # - (m) Length of heat exchanger (axial pipe direction)
        self.n_CPR= int(self.HX_th_length//self.dx_i+1)
        self.n_cells = self.n_CPR * self.n_rows
        self.n_CP = int(self.n_rows/self.n_passes) # Number of adjacent co-flow rows
        if self.n_rows%self.n_passes != 0:
            print("Error in pipe arrangement")
            MyError('Incorrect pipework arrangement')
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
        self.P_PF_dp = [] # Process fluid pressure drop (Pa)

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
    print("Location of maximum residual",np.argmax(abs(r)))

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
    T0 = np.zeros((4*G.n_cells)+(G.n_rows))
    for i in range(G.n_rows):
        T0[i*G.n_CPR:(i+1)*G.n_CPR] = F.T_PF_in - ((F.T_PF_in - F.TC_outlet_guess)*(((i)/G.n_rows)))
        T0[G.n_cells+(i*G.n_CPR):G.n_cells+((i+1)*G.n_CPR)] = F.TC_in + ((F.TC_outlet_guess-F.TC_in)*((G.n_rows -i)/G.n_rows))
        T0[2*G.n_cells:3*G.n_cells] =  F.T_PF_in - ((F.T_PF_in - F.TC_outlet_guess)*(((i+0.5)/G.n_rows)))
        T0[3*G.n_cells:4*G.n_cells] =  F.T_PF_in - ((F.T_PF_in - F.TC_outlet_guess)*(((i+1)/G.n_rows)))
    T0[4*G.n_cells:(4*G.n_cells)+G.n_rows] = F.mdot_PF/G.n_CP # Initial mass flow rate is total divided by all rows
    return T0

def open_T(T,T_PF_in,TC_in,n_cells,n_rows,n_CPR,n_CP,n_passes,P_PF_in,PF_fluid):
    """
    function to unpack the Temperature vector T into the 6 vectors
    TH, TWH, TWC, TC, PH, PC
    """
    TH = np.zeros(n_cells+n_rows)
    TC = np.zeros(n_cells+n_CPR) 
    TW_H = np.zeros(n_cells)
    TW_C = np.zeros(n_cells)
    mdot_vec = np.zeros(n_rows)

    mdot_vec[0:n_rows] = T[4*n_cells:(4*n_cells)+n_rows]

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
                # if i2>0:
                #     TH[start_n] = T_head[i2-1]
                    
            if i2%2 == 1:
                dir_vec.append(-1)
                TH[start_n:end_n-1] = T[row_no*(n_CPR):((row_no+1)*(n_CPR))]
                T_end.append(TH[start_n])
                # if row_no<n_rows:
                #     # TH[end_n-1] = T_head[i2-1]
        # if i2 < (n_passes -1):
        #     h_c = []
            # for c in range(n_CP):
            #     h_c.append(GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P_headers_init[i2],T_end[c]],[CPCP.iHmass]))
            # h_ave = np.dot(h_c,mdot_vec[(i2*n_CP):((i2+1)*n_CP)])/sum(mdot_vec[(i2*n_CP):((i2+1)*n_CP)])
            # T_head.append(CP('T','H',h_ave,'P',P_headers_init[i2],PF_fluid))

    # print("Header temperatures: ",T_head)
    # print("mdot vectors: ",mdot_vec)
    TC[0:n_cells] = T[n_cells:(2*n_cells)]
    TC[n_cells:n_cells+n_CPR] = TC_in
    TW_H[0:n_cells] = T[2*n_cells:3*n_cells]
    TW_C[0:n_cells] = T[3*n_cells:4*n_cells]

    return TH,TC,TW_H,TW_C,dir_vec,mdot_vec

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
        Nu = 0.134*(Pr**0.33)*(Re**0.681)*((2*(G.pitch_fin - G.t_fin)/(G.D_fin - G.OD))**0.2)*(((G.pitch_fin-G.t_fin)/G.t_fin)**0.1134)
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

    if Correlation == 4: # ASPEN HTFS3
        cp = I.cp_f(Tb) 
        U = F.mdot_C/(rho_b*G.A_amb)
        U_f = K_c * U
        Re_f = rho_b * U_f * G.OD/mu
        u_max = F.mdot_C / (rho_b * G.A_c)
        Re_max = u_max * rho_b * G.OD / mu
        j = 0.29 * (Re_max**-0.367)*(G.A_rat**-0.17)
        alpha = j * cp * F.mdot_amb_max * (Pr**(-2/3))
    return alpha 

def calc_f(Re, P, Tm, Tp, TW, mdot, A, Dh, fluid, Correlation,rho_b,rho_w,mu_b, epsilon = 0):
    """ Function to return friction factor for internal pipe flow """
    if Correlation == 0:
        f = 0

    if Correlation == 1:
        f = (0.79*np.log(Re) - 1.64)**-2 # Laminar pipe flow

    # if Correlation == 2: 
    #     mu_w = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P,TW],[CPCP.iviscosity]) 
    #     s = 0.023 * (abs(q/mdot*A)**0.42)
    #     f = (((1.82*np.log(Re))-1.64)**-2)*(rho_w/rho_b)*((mu_w/mu_b)**s)

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
    error = np.zeros((4*G.n_cells)+G.n_rows)
    Q1 = []; Q2 = []; Q3 = []; Q4 = [] 
    Q5 = []; Q6 = []; Q7 = []; Q8 = []; Q9 = []; Alpha_i=[] 
    PH = np.zeros(G.n_cells + G.n_rows);
    P_headers = [F.P_PF_in]
    T_headers = [F.T_PF_in]
    P_row_ends = []
    T_row_ends = []

    TH,TC,TW_H,TW_C,DV,mdot_vec = open_T(T,F.T_PF_in,F.TC_in,G.n_cells,G.n_rows,G.n_CPR,G.n_CP,G.n_passes,F.P_PF_in,F.PF_fluid)

    # Calculate the pressure drop and loss coefficients across each row based on mean air properties for that row. Based off ASPEN HTSF3
    K_c = []
    dp_amb = []

    inlet_cells = []
    outlet_cells = []
    header_inlet_cells = []
    header_outlet_cells = []

    for x in range(G.n_CP):
        inlet_cells.append(x*G.n_CPR)
        outlet_cells.append(G.n_cells - (x*G.n_CPR)-1)
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

    # Pressure drop calculations
    for a in range(G.n_passes):
        for b in range(G.n_CP):
            row = b + (a*G.n_CP)
            
            if DV[row] == 1:
                # print("Row starting node",row * (G.n_CPR+1))
                PH[row * (G.n_CPR+1)] = P_headers[a]
                TH[row * (G.n_CPR+1)] = T_headers[a]

                for c in range(1,G.n_CPR+1):
                    node = (row * (G.n_CPR+1)) + c
                    cell = (row * (G.n_CPR)) + c - 1
                    # print("node",node)
                    # print("cell",cell)
                    Tb_PF = (TH[node] + TH[node-1])/2
                    rho_PF_b = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[node-1],Tb_PF],[CPCP.iDmass]) 
                    rho_PF_w = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[node-1],TW_H[cell]],[CPCP.iDmass]) 
                    mu_PF = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[node-1],Tb_PF],[CPCP.iviscosity]) 
                    U_PF = abs(mdot_vec[row] / (rho_PF_b * G.A_CS))
                    Re_PF = rho_PF_b * U_PF * G.ID / mu_PF
                    f_PF = calc_f(Re_PF,PH[node-1],TH[node],TH[node-1],TW_H[cell],mdot_vec[row],G.A_CS,G.ID,F.PF_fluid,M.f_CorrelationH,rho_PF_b,rho_PF_w,mu_PF)
                    dp = rho_PF_b * f_PF * (G.dx/G.ID) * (U_PF**2) / 2 # calculate friction head loss (Darcy Weisbach formula)
                    PH[node] = PH[node-1] - dp
                P_row_ends.append(PH[node])
                T_row_ends.append(TH[node])
                # print("Row end node",node)
                
            else:
                PH[((row+1) * (G.n_CPR+1))-1] = P_headers[a]
                TH[((row+1) * (G.n_CPR+1))-1] = T_headers[a]

                # print("Row starting node",((row+1) * (G.n_CPR+1))-1)
                for c in range(G.n_CPR-1,-1,-1):
                    node = (row * (G.n_CPR+1)) + c
                    cell = (row * (G.n_CPR)) + c 
                    # print("node",node)
                    # print("cell",cell)
                    Tb_PF = (TH[node] + TH[node+1])/2
                    rho_PF_b = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[node+1],Tb_PF],[CPCP.iDmass]) 
                    rho_PF_w = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[node+1],TW_H[cell]],[CPCP.iDmass]) 
                    mu_PF = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[node+1],Tb_PF],[CPCP.iviscosity]) 
                    U_PF = abs(mdot_vec[row] / (rho_PF_b * G.A_CS))
                    Re_PF = rho_PF_b * U_PF * G.ID / mu_PF
                    f_PF = calc_f(Re_PF,PH[node+1],TH[node],TH[node+1],TW_H[cell],mdot_vec[row],G.A_CS,G.ID,F.PF_fluid,M.f_CorrelationH,rho_PF_b,rho_PF_w,mu_PF)
                    dp = rho_PF_b * f_PF * (G.dx/G.ID) * (U_PF**2) / 2 # calculate friction head loss (Darcy Weisbach formula)
                    PH[node] = PH[node+1] - dp
                P_row_ends.append(PH[node])
                T_row_ends.append(TH[node])
                # print("Row end note",node)
        P_headers.append(PH[node])
        # Calculation of header temperature from enthalpy balance

        if a < (G.n_passes -1):
            h_c = np.zeros(G.n_CP)

            for x in range(G.n_CP):
                h_c[x] = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P_headers[a+1],T_row_ends[x + (a*G.n_CP)]],[CPCP.iHmass])
            h_ave = np.dot(h_c,mdot_vec[(a*G.n_CP):((a+1)*G.n_CP)])/sum(mdot_vec[(a*G.n_CP):((a+1)*G.n_CP)])
            T_headers.append(CP('T','H',h_ave,'P',P_headers[a+1],F.PF_fluid))

    for d in range(G.n_passes-1):
        for e in range(G.n_CP-1):
            delta_header_P = P_headers[d+1]-P_row_ends[(d*G.n_CP)+e]  
            error[4*G.n_cells + (G.n_passes-1)+ ((d*G.n_CP)+e)] = delta_header_P*1e2  # Error for similar pressure between parallel streams at headers

    for f in range(G.n_CP):
        error[(4*G.n_cells) + (G.n_passes-1)+ ((G.n_passes-1)*(G.n_CP-1))+f] = ((F.P_PF_in - P_row_ends[-1-f]) - F.P_PF_dp) # Error for outlet pressure matching required dP
        print(P_row_ends[-1-f])
    # print(error[(4*G.n_cells)+G.n_rows:(4*G.n_cells)+G.n_rows+3])
    # print(error[(4*G.n_cells)+G.n_rows:(4*G.n_cells)+G.n_rows+(G.n_passes-1)])
    # print(error[(4*G.n_cells) + (G.n_passes-1)+ (G.n_passes*(G.n_CP-1))])
    print(error[160:164])
    # Heat transfer calculations
    for i in range(G.n_cells): 
        # Amb node [i] is the outlet to the current cell
        node_amb_i1=i+G.n_CPR # Ambient fluid inlet node to current cell
        node_amb_i2= i+(2*G.n_CPR) # Ambient node two prior to current cell
        node_amb_i3 = i - G.n_CPR # Ambient node following current cell
        
        row = abs(i)//G.n_CPR

        # print("Cell number",i)
        # print("Row number",row)
        # print("PF temp 1",TH[i+row])
        # print(i+row)
        # print("PF temp 2",TH[i+row+1])
        # print(i+row+1)

        k_PF = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[i],(0.5*(TH[i+row]+TH[i+1+row]))],[CPCP.iconductivity]) 
        km_PF = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[i],TH[i+row]],[CPCP.iconductivity])  
        kp_PF = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[i],TH[i+1+row]],[CPCP.iconductivity])  
        k_amb = I.k_f(0.5*(TC[i]+TC[node_amb_i1]))  
        km_amb = I.k_f(TC[node_amb_i1])  
        kp_amb = I.k_f(TC[i])  

        Tb_PF = 0.5*(TH[i+row]+TH[i+1+row]) # bulk temperature
        Pb_PF = 0.5*(PH[i+row]+PH[i+1+row])
        Tb_amb = 0.5*(TC[node_amb_i1]+TC[i]) # bulk temperature
        Pr_PF = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,Pb_PF,Tb_PF],[CPCP.iPrandtl]) 
        Pr_amb = I.Pr_f(Tb_amb)  
        rho_PF_b = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,Pb_PF,Tb_PF],[CPCP.iDmass]) 
        rho_PF_w = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,Pb_PF,TW_H[i]],[CPCP.iDmass]) 
        rho_amb = I.rho_f(Tb_amb) 
        U_PF = abs(mdot_vec[row] / (rho_PF_b * G.A_CS))
        mu_PF = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,Pb_PF,Tb_PF],[CPCP.iviscosity]) 
        Re_PF = rho_PF_b * U_PF * G.ID / mu_PF
        Nu_PF = calc_Nu(G,F,Re=Re_PF,Pr=Pr_PF,P=Pb_PF,Tb=Tb_PF,Tw=TW_H[i],rho_b=rho_PF_b,rho_w=rho_PF_w,Correlation=M.Nu_CorrelationH,mu_b = mu_PF)
        alpha_o = calc_alpha_amb(G,F,I,Pr=Pr_amb,Tb=Tb_amb,rho_b=rho_amb,Correlation=M.alpha_CorrelationC,K_c=K_c[row])
        alpha_i = Nu_PF * k_PF / G.ID
        b = ((2*alpha_o)/(G.k_fin*G.t_fin))**0.5
        n_f = np.tanh(b*G.OD*G.n_phi/2)/(b*G.OD*G.n_phi/2)
        
        
        # if M.consider_bends == 1 and i%(G.n_CPR) == 0 and i != 0:
        #     dp_PF_friction = rho_PF_b * f_PF * ((G.dx + G.bend_path_length)/G.ID) * (U_PF**2) / 2
        #     dp_PF_bend = rho_PF_b * M.bend_loss_coefficient * (U_PF**2) / 2
        #     dp_PF = dp_PF_friction + dp_PF_bend
        # else:

        # Convective fluid-wall heat trasnfer hot side
        q3 = -(TW_H[i] - ((TH[i+row]+TH[i+1+row])/2))*alpha_i*G.dA_i

        # Convective fluid-wall heat transfer cold side
        q4 = -(((TC[node_amb_i1]+TC[i])/2) - TW_C[i])*alpha_o*((G.A_f*n_f)+(G.A_r))

        # PF side bulk convection

        if DV[row]>0:
            q1_conv = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[i+row],TH[i+row]],[CPCP.iHmass]) * mdot_vec[row] * DV[row]
            q2_conv = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[i+row+1],TH[i+row+1]],[CPCP.iHmass]) * mdot_vec[row] * DV[row] * -1
        
            if i in inlet_cells or header_outlet_cells:
                q1_cond = - km_PF * G.A_CS/ G.dx/2 * ( 0.5*(TH[i+row] + TH[i+1+row]) - TH[i+row] )
            else:
                q1_cond = - km_PF * G.A_CS/  G.dx * ( 0.5*(TH[i+row] + TH[i+1+row]) - 0.5*(TH[i+row] + TH[i-1+row]) )

            if i in outlet_cells or header_inlet_cells:
                q2_cond = - kp_PF * G.A_CS/ G.dx/2 * (TH[i+1+row]- 0.5*(TH[i+row] + TH[i+1+row]) ) 
            else:
                q2_cond = - kp_PF * G.A_CS/ G.dx * ( 0.5*(TH[i+row] + TH[i+1+row]) - 0.5*(TH[i+1+row] + TH[i+2+row]) )

        else:
            q2_conv = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[i+row],TH[i+row]],[CPCP.iHmass]) * mdot_vec[row] * DV[row]
            q1_conv = GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[i+row+1],TH[i+row+1]],[CPCP.iHmass]) * mdot_vec[row] * DV[row] *-1

            if i in inlet_cells or header_outlet_cells:
                q2_cond = - km_PF * G.A_CS/ G.dx/2 * ( 0.5*(TH[i+row] + TH[i+1+row]) - TH[i+row] )
            else:
                q2_cond = - km_PF * G.A_CS/  G.dx * ( 0.5*(TH[i+row] + TH[i+1+row]) - 0.5*(TH[i+row] + TH[i-1+row]) )

            if i in outlet_cells or header_inlet_cells:
                q1_cond = - kp_PF * G.A_CS/ G.dx/2 * (TH[i+1+row]- 0.5*(TH[i+row] + TH[i+1+row]) ) 
            else:
                q1_cond = - kp_PF * G.A_CS/ G.dx * ( 0.5*(TH[i+1+row] + TH[i+2+row]) - 0.5*(TH[i+row] + TH[i+1+row]) )

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
            q7 = -(((TW_H[i]+TW_C[i])/2) - ((TW_H[i-1]+TW_C[i-1])/2))*G.k_wall*G.A_WCS/G.dx
            
        if i in outlet_cells or header_inlet_cells:
            q8 = 0
        else:
            q8 = -(((TW_H[i+1]+TW_C[i+1])/2) - ((TW_H[i]+TW_C[i])/2))*G.k_wall*G.A_WCS/G.dx

        q5 = q5_cond + q5_conv
        q6 = q6_cond + q6_conv

        q9 = - (TW_C[i]-TW_H[i]) * G.k_wall * 2 * np.pi * G.dx / np.log((G.ID+(2*G.t_wall))/G.ID)

        error[i] = (q1 + q2 - q3)
        error[G.n_cells + i] = (q4+q5-q6)
        error[2*G.n_cells + i] = (q3 - q4 + q7 - q8)
        error[3*G.n_cells + i] = q9 - (min(q3,q4))

        if flag ==1:
            Q1.append(q1); Q2.append(q2); Q3.append(q3); 
            Q4.append(q4); Q5.append(q5);  Q6.append(q6); 
            Q7.append(q7); Q8.append(q8); Q9.append(q9); Alpha_i.append(alpha_i)

    # Error values for mass flow rate
    
    for j in range(G.n_passes-1):
        # Error value for continuity between passes
        mdot_header_out = [0]
        mdot_header_in = [0]
        for k in range(G.n_CP):
            mdot_header_out = mdot_header_out + mdot_vec[k + ((j+1)*G.n_CP)]
            mdot_header_in = mdot_header_in + mdot_vec[k + (j*G.n_CP)]
        error[4*G.n_cells + j] = (mdot_header_in - mdot_header_out)*1e6
    # print("Error of interest",error[4*G.n_cells:4*G.n_cells + j+2])

    if flag==0:
        return error 
    else:
        return error, T, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Alpha_i, PH, dp_amb

""" --------------------------PLOTTING------------------------------------------------------------- """

def plot_HX(TH,TC,TW_H,TW_C,G,F):
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
    plt.plot(np.linspace(0,1.,num=G.n_cells),TW_H,'--',label="Pipe Wall Hot Side") 
    plt.plot(np.linspace(0,1.,num=G.n_cells),TW_C,'--',label="Pipe Wall Cold Side") 

    plt.ylabel('Temperature (K)')
    plt.xlabel('Position along Heat Exchanger (normalised)')
    plt.title('Temperature Distributions in Heat Exchanger')
    plt.legend(loc=1)
    fig1.show()

def plot_HXq(Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8,Q9, N_cell):
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

def plot_generator(TH_final, TC_final, TW_H_final,TW_C_final,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,T,G,F,M):
        print("Plotting results")
        plot_HX(TH_final,TC_final,TW_H_final,TW_C_final,G,F)
        plot_HXq(Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8,Q9, G.n_cells)

        plt.pause(10) # <-------
        print('\n \n')
        input("<Hit Enter To Close Figures>")
        plt.close()

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


    