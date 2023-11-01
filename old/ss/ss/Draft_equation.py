import sys
import os 
import AL_ACHX_Crossflow_NDDCT as HX
import numpy as np
from CoolProp.CoolProp import PropsSI as CP
import HX_indirect
import ACHX_and_NDDCT_inputs 
import scipy as sci
from scipy import optimize

class Geometry:
    def __init__(self):
        self.R_Hd = [] # Aspect ratio H5/d3
        self.R_dD = []# Diameter ratio d5/d3
        self.R_AA = [] # Area coverage of heat exchangers Aft/A3
        self.R_hD = [] # Ratio of inlet height to diameter H3/d3
        self.R_sd = []  # Ratio of number of tower supports to d3
        self.R_LH = [] # Ratio of height support to tower height L/H3
        self.D_ts = [] # Diameter of tower supports
        self.C_Dts = [] # Drag coefficeint of tower supports
        self.K_ct = [] # Loss coefficient of cooling tower separation
        self.sigma_c = [] # Sigma_c, per Kroger
        self.A_HX = [] # (m2) Frontal area of each heat exchanger
        self.dA_HX = [] #(m2) Frontal area of section of HX analysed in code
        self.d3 = [] # Initial guess for cooling tower diameter 
        
    def dim_update(self,d3):
        # All dimensions in meters and labelled per Kroger
        self.H5 = self.R_Hd * d3 
        self.d5 = self.R_dD * d3
        self.A3 = np.pi * 0.25 * (d3**2)
        self.A5 = np.pi * 0.25 * (self.d5**2)
        self.A_fr = self.A3 * self.R_AA
        self.H3 = self.R_hD * d3
        self.n_ts = self.R_sd * d3
        self.L_ts = self.R_LH * d3

class Design_point:
    def __init__(self):
        self.T_a1 = [] # (K) Ambient temperature
        self.p_a1 = [] # (Pa) Ambient temperature
        self.PF_fluid = []
        self.T_CO2_in = [] # (K) Design CO2 inlet temperature
        self.T_CO2_out = [] # (K) Design CO2 outlet temperature
        self.mdot_CO2_total = [] # (kg/s) Total CO2 mass flow rate througout system 
        self.HX_PP = [] # (K) CO2 - H2O heat exchanger design pinch point
        self.P_CO2 = [] # (Pa) CO2 fluid pressure

    def micro_init(self):
        self.T_H2O_out, self.mdot_multi = HX_indirect.HX_PP_analysis(self.T_CO2_in,self.T_CO2_out,self.HX_PP)
        self.mdot_H2O_total = self.mdot_CO2_total * self.mdot_multi
        self.Q_design = (CP('H','T',self.T_CO2_in,'P',self.P_CO2,'CO2')-CP('H','T',self.T_CO2_out,'P',self.P_CO2,'CO2'))*self.mdot_CO2_total

    def flow_update(self,CT):
        self.mdot_CO2 = self.mdot_CO2_total / (CT.A_fr / CT.dA_HX)
        self.mdot_H20 = self.mdot_CO2 * self.mdot_multi

class Model_parameters:
    def __init__(self):
        self.fluid_switch = []
        self.Vtolerance = []
        self.Qtolerance = []

CT = Geometry()
DP = Design_point()
MP = Model_parameters()
ACHX_and_NDDCT_inputs.NDDCT_inputs(CT,DP,MP)
DP.micro_init()

d3 = CT.d3i
CT.dim_update(d3)

def T_calc(T_a1,z):
    T = T_a1 - (0.00975 * z)
    return T

def p_calc(p_a1,T_a1,z):
    Tz = T_calc(T_a1,z)
    p = p_a1*((Tz/T_a1)**3.5)
    return p

def Draft_equation(v,CT,DP,MP,mod):
    global T0
    T_heo, dp_HX, T0, q = HX.main(v,DP.mdot_CO2,T0,fig_switch=0,Value_override=mod)
    print("Hot air temperature:",T_heo)
    T_a4 = T_calc(T_heo,CT.H3)
    T_a5 = T_calc(T_a4,CT.H5-CT.H3)
    p_a6 = p_calc(DP.p_a1,DP.T_a1,CT.H5)
    p_a34 = p_calc(DP.p_a1,DP.T_a1,CT.H3)

    rho_a1 = CP('D','T',DP.T_a1,'P',DP.p_a1,'Air')
    rho_a3 = CP('D','T',DP.T_a1,'P',p_a34,'Air')
    rho_a4 = CP('D','T',T_a4,'P',p_a34,'Air')
    rho_a5 = CP('D','T',T_a5,'P',p_a6,'Air')
    rho_a6 = CP('D','T',DP.T_a1,'P',p_a6,'Air')
    rho_a34 = 2/((1/rho_a3)+(1/rho_a4))

    mdot_a = v * CT.A_fr * rho_a3
    Fr_D = ((mdot_a/CT.A5)**2)/(rho_a5*(rho_a6-rho_a5)*9.81*CT.d5)

    K_he = dp_HX * 2 / ((v**2) * (rho_a34**2))
    K_ct = (-18.7 + (8.095*(1/CT.R_hD)) - (1.084 * (1/CT.R_hD**2)) + (0.0575 * (1/CT.R_hD**3)))*(K_he**(0.165 - (0.035*(1/CT.R_hD)))) #from Kroger eq 7.3.6 (p70)

    K_to = (-0.28 * (Fr_D**-1)) + (0.04*(Fr_D**-1.5)) # Tower outlet losses
    K_tshe = CT.C_Dts * CT.L_ts * CT.D_ts * CT.n_ts * (CT.A_fr**2) * (rho_a34/rho_a1)/((np.pi * d3 * CT.H3)**3) # Tower support losses
    K_cthe = K_ct *(rho_a34/rho_a3) * ((CT.A_fr/CT.A3)**2) # Tower losses due to flow detatchment - 
    K_ctche = (1 - (2/CT.sigma_c) + (1/(CT.sigma_c**2)))*(rho_a34/rho_a3) * ((CT.A_fr/CT.A3)**2) # Losses due to contraction through HX bundles
    K_ctehe = (((1-CT.A_fr)/CT.A3)**2)*(rho_a34/rho_a4) * ((CT.A_fr/CT.A3)**2) # Losses due to expansion through HX bundles
    K_hes  = 0.1

    K_sum = K_tshe + K_cthe + K_ctche + K_ctehe + K_hes

    term1 = p_a34 *((1-0.00957*(CT.H5-CT.H3)/T_a4)**3.5)
    term2 = p_a6
    term3 = dp_HX
    term4 = (1/(2*(CT.A_fr**2)*rho_a34))*K_sum * ((1-0.00957*(CT.H5-CT.H3)/T_a4)**3.5)
    term5 = (1/(2*(CT.A5**2)*rho_a5))*(1+K_to)

    mdot12 = (term1 - term2 - term3)/(term4 + term5)
    
    if mdot12<0:
        mdot1 = -(-mdot12)**0.5
    else:
        mdot1 = (mdot12)**0.5

    v_fr = mdot1/(rho_a3*CT.A_fr)
    Verror = (v_fr-v)/v
    # print("K_he: ",K_he)
    # print("K_cthe: ",K_cthe)
    # print("K_to: ",K_to)
    # print("K_tshe: ",K_tshe)
    # print("K_ctche: ",K_ctche)
    # print("K_ctehe: ",K_ctehe)
    # print("K_hes",K_hes)
    return Verror, q

def NDDCT_sizer(X,CT,DP,MP):
    d3 = X[0]
    v = X[1]

    CT.dim_update(d3)
    DP.flow_update(CT)

    Verror, q = Draft_equation(v,CT,DP,MP)
    
    Q = q * DP.mdot_CO2_total/DP.mdot_CO2
    Qerror = (Q-DP.Q_design)/DP.Q_design
    
    print("Trialed diameter",d3)
    print("Trialed velovity",v)
    print("Qerror:",Qerror*100,"%")
    print("Verror:",Verror*100,"%")
    
    E = [Qerror,Verror]
    return E

args = CT,DP,MP
T0 = [0]
d3i = 37
vi = 1.65
X = [d3i,vi]

x = sci.optimize.fsolve(NDDCT_sizer,X,args,xtol=0.000001)
print(x)













# def V_solve(CT,DP,MP,d3,v_fr1,T0):
#     Vtolerance = 0.01
#     Verror = Vtolerance + 1
#     it_no = 0
#     while Verror > MP.Vtolerance:
#     T_heo, dp_HX, T0, Q = HX.main(v_fr1,DP.mdot_CO2,T0,fig_switch=0)
#     print("Hot air temperature:",T_heo)
#     T_a4 = T_calc(T_heo,CT.H3)
#     T_a5 = T_calc(T_a4,CT.H5-CT.H3)
#     p_a6 = p_calc(DP.p_a1,DP.T_a1,CT.H5)
#     p_a34 = p_calc(DP.p_a1,DP.T_a1,CT.H3)

#     rho_a1 = CP('D','T',DP.T_a1,'P',DP.p_a1,'Air')
#     rho_a3 = CP('D','T',DP.T_a1,'P',p_a34,'Air')
#     rho_a4 = CP('D','T',T_a4,'P',p_a34,'Air')
#     rho_a5 = CP('D','T',T_a5,'P',p_a6,'Air')
#     rho_a6 = CP('D','T',DP.T_a1,'P',p_a6,'Air')
#     rho_a34 = 2/((1/rho_a3)+(1/rho_a4))

#     mdot_a = v_fr1 * CT.A_fr * rho_a3
#     Fr_D = ((mdot_a/CT.A5)**2)/(rho_a5*(rho_a6-rho_a5)*9.81*CT.d5)

#     K_he = dp_HX * 2 / ((v_fr1**2) * (rho_a34**2))
#     K_ct = (-18.7 + (8.095*(1/CT.R_hD)) - (1.084 * (1/CT.R_hD**2)) + (0.0575 * (1/CT.R_hD**3)))*(K_he**(0.165 - (0.035*(1/CT.R_hD)))) #from Kroger eq 7.3.6 (p70)


#     K_to = (-0.28 * (Fr_D**-1)) + (0.04*(Fr_D**-1.5)) # Tower outlet losses
#     K_tshe = CT.C_Dts * CT.L_ts * CT.D_ts * CT.n_ts * (CT.A_fr**2) * (rho_a34/rho_a1)/((np.pi * d3 * CT.H3)**3) # Tower support losses
#     K_cthe = K_ct *(rho_a34/rho_a3) * ((CT.A_fr/CT.A3)**2) # Tower losses due to flow detatchment - 
#     K_ctche = (1 - (2/CT.sigma_c) + (1/(CT.sigma_c**2)))*(rho_a34/rho_a3) * ((CT.A_fr/CT.A3)**2) # Losses due to contraction through HX bundles
#     K_ctehe = (((1-CT.A_fr)/CT.A3)**2)*(rho_a34/rho_a4) * ((CT.A_fr/CT.A3)**2) # Losses due to expansion through HX bundles
#     K_hes  = 0.1

#     K_sum = K_tshe + K_cthe + K_ctche + K_ctehe + K_hes

#     term1 = p_a34 *((1-0.00957*(CT.H5-CT.H3)/T_a4)**3.5)
#     term2 = p_a6
#     term3 = dp_HX
#     term4 = (1/(2*(CT.A_fr**2)*rho_a34))*K_sum * ((1-0.00957*(CT.H5-CT.H3)/T_a4)**3.5)
#     term5 = (1/(2*(CT.A5**2)*rho_a5))*(1+K_to)

#     mdot12 = (term1 - term2 - term3)/(term4 + term5)
    
#     if mdot12<0:
#         mdot1 = -(-mdot12)**0.5
#     else:
#         mdot1 = (mdot12)**0.5

#     v_fr = mdot1/(rho_a3*CT.A_fr)
#     print("Iteration no:",it_no)
#     print("Trialed velocity",v_fr1)
#     change = (v_fr-v_fr1)/15
#     Verror = abs(v_fr1-v_fr)
#     v_fr1 = v_fr1 + change
#     print("New velocity",v_fr1)
#     print("Error:",Verror)
#     print("K_he: ",K_he)
#     print("K_cthe: ",K_cthe)
#     print("K_to: ",K_to)
#     print("K_tshe: ",K_tshe)
#     print("K_ctche: ",K_ctche)
#     print("K_ctehe: ",K_ctehe)
#     print("K_hes",K_hes)
#     return v_fr, T_heo, dp_HX, T0, Q

# def Q_solve(CT,DP,MP,d1):
#     d3=d1
#     Qtolerance = 1000
#     Qerror = Qtolerance + 1
#     Q_it = 0
#     T0 = [0]
#     v_fr = 1.6545
#     while Qerror > MP.Qtolerance:
#         CT.dim_update(d3)
#         DP.flow_update(CT)
#         Q_it = Q_it + 1
#         v_fr, T_heo, dp_HX, T0, q = V_solve(CT,DP,MP,d3,v_fr,T0)
#         print("VELOCITY RESULT REACHED. ADJUSTING DIAMETER")
#         print("Final HX inlet velocity: ",v_fr)
#         Q = q * DP.mdot_CO2_total/DP.mdot_CO2
#         print("Q Iteration no:",Q_it)
#         print("Trialed diameter",d3)
#         Dchange = (Q/DP.Q_design)**0.5
#         Qerror = abs(DP.Q_design -Q)
#         d3 = d3 / (Dchange)
#         print("New diameter",d3)
#         print("Q error:",Qerror/1000,"kW")
#         print("New PF flow rate",DP.mdot_CO2)
#     return v_fr,T0,d3
  
# v_fr_final, T0_final = Q_solve(CT,DP,MP,d3)

# T_heo, dp_HX, T0, q = HX.main(v_fr1,DP.mdot_CO2,T0,fig_switch=1,)

# print("Final Diameter:",d3)
# print("Final Air Velocity:",v_fr1)

# print("---------------------------------------------------------")
# print("---------------- FINISHED CALCULATION -------------------")
# print("---------------------------------------------------------")


