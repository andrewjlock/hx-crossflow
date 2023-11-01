import AL_ACHX_Crossflow_NDDCT_CPmod_4eq_MP_varP

from CoolProp.CoolProp import PropsSI as CP
import numpy as np
import csv
from itertools import zip_longest
import os
import datetime



def main(mod=None):

    TC_out_ave, dp_amb_total, T1, Q_PF, PF_dp, TH_final, dp_amb, alpha_i_final, TW_H_final, TW_C_final, Q3, PH, TC_final, alpha_o_final, dT, mdot_vec, G, Bu_c, Bu_j, T_out = AL_ACHX_Crossflow_NDDCT_CPmod_4eq_MP_varP.main(ACHX_inputs,T0i = [0],fig_switch=0, mod=mod)
    
    x = []

    datestamp = datetime.datetime.now().strftime("%I-%M-%S%p_%B_%d_%Y")

    dT_ave = (sum(dT)/len(dT))
    dT.append(dT_ave)
    Q3_t = sum(Q3)
    Q3.append(Q3_t/len(Q3))
    Q3.append(Q3_t)
    Ai = G.HX_length * G.n_rows * G.ID * np.pi
    U = Q3_t/(Ai * dT_ave)
    print("Overall U: ",U)

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
        dT_C = TC_out_ave - TC_final[-1]
            
    return dT, dT_C, Q_PF, T_out, TW_H_final[-1]
	

def ACHX_inputs(G,F,M):    
    G.HX_length = 2*4
    G.n_rows = 1  # - Number of rows (deep) of heat exchangerprint(q1)
    G.n_passes = 1 # Number of passes in the heat exchanger for eaach circuit
    G.pitch_longitudal = 0.0635*np.cos(np.pi/6) # - (m) Longitudal tube pitch 
    G.pitch_transverse =  0.0635  # - s(m) Transverse tube pitch
    G.ID = 0.0212  # (m) ID of HX pipe
    G.t_wall = 0.0021 # (m)c Thickness of pipe wall
    G.dx_i = 0.2001  #0.4 # (m) Initial length of pipe discretisation
    G.k_wall = 40 #14.192 # (W/m) Thermal conductivity of pipe wall
    G.D_fin = 0.05715  # (m) Maximum diameter of fin
    G.pitch_fin = 0.0028 # (m) Fin pitch
    G.t_fin = 0.0004 # (m) Mean thickness of pipe fin
    G.k_fin = 200 # (m)

    M.Nu_CorrelationH = 5    # set correlation for heat transfer in H channel (1)-Yoon f(T_b), (2)-Yoon f(T_b,T_w), (3)-Gnielinski (4)-Pitla (5)-Wang
    M.alpha_CorrelationC = 3 # set correlation for heat transfer in C channel
    M.f_CorrelationH = 1 # set correlation for friction factor in H channel
    M.f_CorrelationC =[] # set correlation for friction factor in C channel 
    M.consider_bends = 1 # Consider bends in pressure loss? 1-yes, 0-no
    M.bend_loss_coefficient = 1 # Bend friction factor (from Chen 2018) (approximate, converted from Fanning to Darcy Weisbach)
    M.solver_type = 1   # Swtich for solver type  - (0) for pressure differential input, (1) for total mass flow input (2) for temperature output

    F.PF_fluid  = 'CO2'       
    F.Amb_fluid  = 'Air'
    F.T_PF_in  = [] # (K)b
    F.P_PF_in  = [] # 7.96 * 10**6 # 7.96*10**6 # 7.96*10**6 # pressure (Pa)

    F.TC_in  = 273.15+30 # (K)
    F.vC_in  = [] # (m/s)
    F.P_amb_in  = 101325 # pressure (Pa)
    F.TC_outlet_guess = 273.15+49 # Air side outlet guess for initialisation (K)

    F.P_PF_dp = 1000 # Process fluid pressure drop (Pa)
    F.T_PF_out = 273.15+33 # For temperature boundary condition
    F.mdot_PF = 0.1 # Total mass flow rate through the heat exchanger circuits combined (kg/s). Is initial guess if using pressure solver

P_array = [7.5,8,9,10]
TH_array = [35,37.5,40,45,50,55,60,70]
V_array = [0.5,1,2,3,4]

n=0
P_results = np.zeros(len(P_array)*len(V_array)*len(TH_array))
TH_results = np.zeros(len(P_array)*len(V_array)*len(TH_array))
TW_results = np.zeros(len(P_array)*len(V_array)*len(TH_array))
V_results = np.zeros(len(P_array)*len(V_array)*len(TH_array))
dT_results = np.zeros(len(P_array)*len(V_array)*len(TH_array))
dT_C_results = np.zeros(len(P_array)*len(V_array)*len(TH_array))
Q_results = np.zeros(len(P_array)*len(V_array)*len(TH_array))
dQ_results = np.zeros(len(P_array)*len(V_array)*len(TH_array))
Unc_results = np.zeros(len(P_array)*len(V_array)*len(TH_array))

dRTD = 0.1 # Accuracy of RTD (degC)
dmdot = 0.001 # Accuracy of Coriolis flow meter 0.1% for liquids
mdot = 0.1 # Mass flow rate used in modelling

for a in range(len(P_array)):
    for b in range(len(V_array)):
        for c in range(len(TH_array)):
            
            P_results[n] = P_array[a]
            V_results[n] = V_array[b]
            TH_results[n] = TH_array[c]

            
            def mod_1(A):
                if hasattr(A, 'P_PF_in'):
                    A.P_PF_in = P_array[a] * 10**6
                    A.vC_in = V_array[b]
                    A.T_PF_in = TH_array[c] + 273.15

            dT, dT_C, Q_total,T_out, TW_min = main(mod_1)
            dT_results[n] = dT
            TW_results[n] = TW_min
            dT_C_results[n] = dT_C
            Q_results[n] = Q_total

            h_in = CP('H','T',TH_array[c]+273.15,'P',P_array[a]* 10**6,'CO2')
            h_in_d1 = abs(CP('H','T',TH_array[c]+273.15+dRTD,'P',P_array[a]* 10**6,'CO2') - h_in)
            h_in_d2 = abs(CP('H','T',TH_array[c]+273.15 -dRTD,'P',P_array[a]* 10**6,'CO2') - h_in)
            
            h_out = CP('H','T',T_out,'P',P_array[a]* 10**6,'CO2')
            h_out_d1 = abs(CP('H','T',T_out+dRTD,'P',P_array[a]* 10**6,'CO2') - h_out)
            h_out_d2 = abs(CP('H','T',T_out-dRTD,'P',P_array[a]* 10**6,'CO2') - h_out)
            
            dQ_1 = ( ((mdot*h_in_d1)**2) + ((mdot*h_out_d2)**2) + (((h_in - h_out)*mdot*dmdot)**2))**0.5
            dQ_2 = ( ((mdot*h_in_d2)**2) + ((mdot*h_out_d1)**2) + (((h_in - h_out)*mdot*dmdot)**2))**0.5
            dQ = max(dQ_1,dQ_2)
            Unc = dQ/Q_total
            
            dQ_results[n] = dQ
            Unc_results[n] = Unc

            n = n+1

Final_results = [P_results,TH_results,V_results,dT_results, TW_results, dT_C_results, Q_results, dQ_results, Unc_results]
Final_names = ["P_results","TH_results","V_results","dT_results","TW_results","dT_C_results","Q_results", "dQ_results", "Unc_results"]

Results = zip_longest(*Final_results,fillvalue='')

datestamp = datetime.datetime.now().strftime("%I-%M-%S%p_%B_%d_%Y")
Filename = "Final_Results_"+datestamp+".csv"

with open("Results\\"+Filename, "w",newline='') as results_file:
    writer = csv.writer(results_file)
    writer.writerow(Final_names)
    for row in Results:
        writer.writerow(row)

print(dT_results)
print(Q_results)

#
#def mod_2(A):
#    if hasattr(A, 'HX_length'):
#        A.HX_length = 6
#        A.n_passes = 4
#        A.dx_i = 0.3001 
#
#def mod_3(A):
#    if hasattr(A, 'HX_length'):
#        A.HX_length = 10
#        A.n_passes = 1
#        A.dx_i = 0.2001 
#
#def mod_4(A):
#    if hasattr(A, 'HX_length'):
#        A.HX_length = 10
#        A.n_passes = 4
#        A.dx_i = 0.5001 
#
#main(mod_1)
#main(mod_2)
#main(mod_3)
#main(mod_4)
