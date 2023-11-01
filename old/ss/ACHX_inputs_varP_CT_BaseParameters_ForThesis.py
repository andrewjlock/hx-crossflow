import AL_ACHX_Crossflow_NDDCT_CPmod_4eq_MP_varP_DE

from CoolProp.CoolProp import PropsSI as CP
import numpy as np
import csv
from itertools import zip_longest
import os
import datetime

datestamp = datetime.datetime.now().strftime("%I%M%p_%B_%d_%Y")
d3 = [0]
v = [0]

def main():
   
    TC_out_ave, dp_amb_total, T1, Q_PF, PF_dp, TH_final, dp_amb, alpha_i_final, TW_H_final, TW_C_final, Q3, PH, TC_final, alpha_o_final, dT, mdot_vec, G, d3[0], v[0], Bu_c, Bu_j = AL_ACHX_Crossflow_NDDCT_CPmod_4eq_MP_varP_DE.main(ACHX_inputs,T0i = [0],fig_switch=0)
    
    x = []

    datestamp = datetime.datetime.now().strftime("%I%M%p_%B_%d_%Y")

    dT.append(sum(dT)/len(dT))
    Q3_t = sum(Q3)
    Q3.append(Q3_t/len(Q3))
    Q3.append(Q3_t)

    alpha_i_final.append(sum(alpha_i_final)/len(alpha_i_final))
    alpha_o_final.append(sum(alpha_o_final)/len(alpha_o_final))

    for i in range(G.n_rows):
                TW_H_final = np.insert(TW_H_final,(G.n_CPR*i)+i,0)
                TW_C_final = np.insert(TW_C_final,(G.n_CPR*i)+i,0)
                dT = np.insert(dT,(G.n_CPR*i)+i,0)
                alpha_i_final = np.insert(alpha_i_final,(G.n_CPR*i)+i,0)
                alpha_o_final = np.insert(alpha_o_final,(G.n_CPR*i)+i,0)
                TC_final = np.insert(TC_final,(G.n_CPR*i)+i,0)
                Q3 = np.insert(Q3,(G.n_CPR*i)+i,0)
                Bu_c = np.insert(Bu_c,(G.n_CPR*i)+i,0)
                Bu_j = np.insert(Bu_j,(G.n_CPR*i)+i,0)

                x.extend(np.linspace(0,G.HX_length,G.n_CPR+1))

    Raw_results = [x, TH_final, TW_H_final, TW_C_final, TC_final, dT, PH, dp_amb, alpha_i_final, alpha_o_final, Q3, mdot_vec, d3, v, Bu_c, Bu_j]
    Raw_results_names = ["x","TH_final", "TW_H_final", "TW_C_final", "TC_final", "dT", "PH", "dp_amb", "alpha_i_final", "alpha_o_final", "Q3", "mdot_vec", "Tower inlet diameter", "Air face velocity", "Bu_c", "Bu_j"]

    Results = zip_longest(*Raw_results,fillvalue='')
    Filename = "HX_Results_(2)_"+datestamp+".csv"

    with open("Results\\"+Filename, "w",newline='') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(Raw_results_names)
        for row in Results:
            writer.writerow(row)


def ACHX_inputs(G,F,M,CT):    
    G.HX_length = 14
    G.n_rows = 4  # - Number of rows (deep) of heat exchangerprint(q1)
    G.n_passes = 2
    G.pitch_longitudal = 0.0635*np.cos(np.pi/6) # - (m) Longitudal tube pitch 
    G.pitch_transverse =  0.0635  # - s(m) Transverse tube pitchPp
    G.ID = 0.0212  # (m) ID of HX pipe
    G.t_wall = 0.0021 # (m)c Thickness of pipe wall
    G.dx_i = 0.6001  #0.4 # (m) Initial length of pipe discretisation
    G.k_wall = 50 #14.192 # (W/m) Thermal conductivity of pipe wall
    G.D_fin = 0.05715  # (m) Maximum diameter of fin
    G.pitch_fin = 0.00231 # (m) Fin pitch
    G.t_fin = 0.0004 # (m) Mean thickness of pipe fin
    G.k_fin = 167 # (m)

    M.Nu_CorrelationH = 5    # set correlation for heat transfer in H channel (1)-Yoon f(T_b), (2)-Yoon f(T_b,T_w), (3)-Gnielinski (4)-Pitla (5)-Wang
    M.alpha_CorrelationC = 3 # set correlation for heat transfer in C channel
    M.f_CorrelationH = 2 # set correlation for friction factor in H channel
    M.f_CorrelationC =[] # set correlation for friction factor in C channel 
    M.consider_bends = 1 # Consider bends in pressure loss? 1-yes, 0-no
    M.bend_loss_coefficient = 0 # Bend friction factor (from Chen 2018) (approximate, converted from Fanning to Darcy Weisbach)
    M.solver_type = 2   # Swtich for solver type  - (0) for pressure differential input, (1) for total mass flow input (2) for temperature output
    # Note: For this cooling tower model, M.solver must be 0 or 2.

    F.PF_fluid  = 'CO2'       
    F.Amb_fluid  = 'Air'
    F.T_PF_in  = 273.15+71 #  CO2 inlet temperature (K)
    F.T_PF_out = 273.15+33.5 # CO2 temperature outlet (K) 
    F.P_PF_in  = 8*10**6 # CO2 inlet pressure (Pa)
    
    F.TC_in  = 273.15+30 # Air ground level temperature (K) 
    F.vC_in  = 1.5 # Initialisation air velocity (m/s) - solved during run
    F.P_amb_in  = 101325 # Atmospheric pressure (Pa)
    F.TC_outlet_guess = 273.15+55 # Air side outlet guess for initialisation (K)
    F.P_PF_dp = 12000 # Process fluid pressure drop for cooling tower solver type 0

    F.mdot_PF = [] # Ignore this value in cooling tower model

    CT.R_Hd = 1.4 # Aspect ratio H5/d3
    CT.R_dD = 0.7 # Diameter ratio d5/d3
    CT.R_AA = 0.65 # Area coverage of heat exchangers Aft/A3
    CT.R_hD = 1/6.5 # Ratio of inlet height to diameter H3/d3
    CT.R_sd = 0.724  # Ratio of number of tower supports to d3
    CT.R_LH = 15.78/11 # Ratio of height support to tower height L/H3
    CT.D_ts = 0.5 # Diameter of tower supports
    CT.C_Dts = 2 # Drag coefficeint of tower supports
    CT.K_ct = 0.1 # Loss coefficient of cooling tower separation
    CT.sigma_c = 0.725 # Sigma_c, per Kroger
    CT.dA_width = [] #(m2) Frontal area of section of HX (calculated within code)
    CT.solver_type = 1 # Solver type: (0) - fixed diameter varying mass flow rate, (1) - fixed mass flow rate, varying diameter (can't use const mdot and diameter)
    CT.d3 = 60 # Diameter of cooling tower inlet for initialisation
    CT.mdot_PF_total = 152.69 # Total mass flow rate through the cooling tower

main()

#R_Hd_array = [1.2, 1.3, 1.4]
#Results = np.zeros(len(R_Hd_array))
#
#for i in range(len(R_Hd_array)):
#
#    def mod_3(A):
#        if hasattr(A, 'R_Hd'):
#            A.R_Hd = R_Hd_array[i]
#    Results[i] = main()
#
#print(Results)
#
#main()



