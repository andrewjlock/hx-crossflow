import ACHX_solver

from CoolProp.CoolProp import PropsSI as CP
import numpy as np
import csv
from itertools import zip_longest
import os
import datetime
import matplotlib.pyplot as plt


def main(mod=None,verbosity=1):
# This is the main function that executes a result
# A lot of the details are just for saving results etc.

    TC_out_ave, dp_amb_total, T1, Q_PF, PF_dp, TH_final, dp_amb, alpha_i_final, TW_H_final, TW_C_final, Q3, PH, TC_final, alpha_o_final, dT, mdot_vec, G, F, Bu_c, Bu_j, T_out = ACHX_solver.main(ACHX_inputs,T0i = [0],fig_switch=0, mod=mod,verbosity=verbosity)
    
    

    x = []

    dT_ave = (sum(dT)/len(dT))
    dT.append(dT_ave)
    Q3_t = sum(Q3)
    Q3.append(Q3_t/len(Q3))
    Q3.append(Q3_t)
    Ai = G.HX_length * G.n_rows * G.ID * np.pi
    U = Q3_t/(Ai * dT_ave)
    # print("Overall U: ",U)

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

    # The following code writes a result file as a CSV with a timestamp:

    datestamp = datetime.datetime.now().strftime("%I-%M-%S%p_%B_%d_%Y")

    Raw_results = [x, TH_final, TW_H_final, TW_C_final, TC_final, dT, PH, dp_amb, alpha_i_final, alpha_o_final, Q3, mdot_vec, Bu_c, Bu_j]
    Raw_results_names = ["x","TH_final", "TW_H_final", "TW_C_final", "TC_final", "dT", "PH", "dp_amb", "alpha_i_final", "alpha_o_final", "Q3", "mdot_vec", "Bu_c", "Bu_j"]

    Results = zip_longest(*Raw_results,fillvalue='')
    Filename = "HX_Results_"+datestamp+".csv"

    with open("Results\\"+Filename, "w",newline='') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(Raw_results_names)
        for row in Results:
            writer.writerow(row)
        dT = TH_final[0] - T_out
        dT_C = TC_final[-1] - TC_out_ave

    return T_out 
	
# These parameters are the geometrical parameters of the heat exchanger
def ACHX_inputs(G,F,M):    
    G.HX_length = 12
    G.n_rows = 4  # - Number of rows (deep) of heat exchangerprint(q1)
    G.n_passes = 2
    G.pitch_longitudal = 0.0635*np.cos(np.pi/6) # - (m) Longitudal tube pitch 
    G.pitch_transverse =  0.0635  # - s(m) Transverse tube pitchPp
    G.ID = 0.0212  # (m) ID of HX pipe
    G.t_wall = 0.0021 # (m)c Thickness of pipe wall
    G.dx_i = 1.5001  #0.4 # (m) Initial length of pipe discretisation
    G.k_wall = 40 #14.192 # (W/m) Thermal conductivity of pipe wall
    G.D_fin = 0.05715  # (m) Maximum diameter of fin
    G.pitch_fin = 0.0028 # (m) Fin pitch
    G.t_fin = 0.0004 # (m) Mean thickness of pipe fin
    G.k_fin = 200 # (m)

    # These are the model parameters, such as which correlations etc. 
    M.Nu_CorrelationH = 5    # set correlation for heat transfer in H channel (1)-Yoon f(T_b), (2)-Yoon f(T_b,T_w), (3)-Gnielinski (4)-Pitla (5)-Wang
    M.alpha_CorrelationC = 1 # set correlation for heat transfer in C channel
    M.f_CorrelationH = 1 # set correlation for friction factor in H channel
    M.f_CorrelationC =[] # set correlation for friction factor in C channel 
    M.consider_bends = 1 # Consider bends in pressure loss? 1-yes, 0-no
    M.bend_loss_coefficient = 2 # Bend friction factor (from Chen 2018) (approximate, converted from Fanning to Darcy Weisbach)
    M.solver_type = 2   # Swtich for solver type  - (0) for pressure differential input, (1) for total mass flow input (2) for temperature output

    # These are the fluid properties
    F.PF_fluid  = 'CO2'       
    F.Amb_fluid  = 'Air'
    F.T_PF_in  =354.55# (K)b
    F.P_PF_in  =9*10**6 # 7.96 * 10**6 # 7.96*10**6 # 7.96*10**6 # pressure (Pa)

    F.TC_in  = 273.15+30 # (K)
    F.vC_in  = 3 # (m/s)
    F.P_amb_in  = 101325 # pressure (Pa)
    F.TC_outlet_guess = 273.15+35 # Air side outlet guess for initialisation (K)

    F.P_PF_dp = 1000 # Process fluid pressure drop (Pa)
    F.T_PF_out = 273.15 + 40 # For temperature boundary condition
    F.mdot_PF = 0.2     # Total mass flow rate through the heat exchanger circuits combined (kg/s). Is initial guess if using pressure solver

#main()
if __name__ == "__main__":

    main()

