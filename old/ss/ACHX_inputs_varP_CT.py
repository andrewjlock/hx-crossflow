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

def main(T0=[0],mod=None):
# This is the main function that executes a result

    TC_out_ave, dp_amb_total, T1, Q_PF, PF_dp, TH_final, dp_amb, alpha_i_final, TW_H_final, TW_C_final, Q3, PH, TC_final, alpha_o_final, dT, mdot_vec, G, d3[0], v[0], Bu_c, Bu_j, T_out = AL_ACHX_Crossflow_NDDCT_CPmod_4eq_MP_varP_DE.main(ACHX_inputs,T0i = T0,fig_switch=0, mod=mod)
    
    x = []

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

    # The following code writes a result file as a CSV with a timestamp:
    datestamp = datetime.datetime.now().strftime("%I%M%p_%B_%d_%Y")
    
    Raw_results = [x, TH_final, TW_H_final, TW_C_final, TC_final, dT, PH, dp_amb, alpha_i_final, alpha_o_final, Q3, mdot_vec, d3, v, Bu_c, Bu_j]
    Raw_results_names = ["x","TH_final", "TW_H_final", "TW_C_final", "TC_final", "dT", "PH", "dp_amb", "alpha_i_final", "alpha_o_final", "Q3", "mdot_vec", "Tower inlet diameter", "Air face velocity", "Bu_c", "Bu_j"]

    Results = zip_longest(*Raw_results,fillvalue='')
    Filename = "HX_Results_(2)_"+datestamp+".csv"

    with open("Results\\"+Filename, "w",newline='') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(Raw_results_names)
        for row in Results:
            writer.writerow(row)
#    print (T_out)
    return  T_out, v[0], TC_out_ave, T1 #d3[0], T1 

def ACHX_inputs(G,F,M,CT):    
    # This is the heat exchanger geometric inputs
    G.HX_length = 12
    G.n_rows = 4  # - Number of rows (deep) of heat exchangerprint(q1)
    G.n_passes = 2
    G.pitch_longitudal = 0.0635*np.cos(np.pi/6) # - (m) Longitudal tube pitch 
    G.pitch_transverse =  0.0635  # - s(m) Transverse tube pitchPp
    G.ID = 0.0212  # (m) ID of HX pipe
    G.t_wall = 0.0021 # (m)c Thickness of pipe wall
    G.dx_i = 0.6001  #0.4 # (m) Initial length of pipe discretisation
    G.k_wall = 40 #14.192 # (W/m) Thermal conductivity of pipe wall
    G.D_fin = 0.05715  # (m) Maximum diameter of fin
    G.pitch_fin = 0.0028 # (m) Fin pitch
    G.t_fin = 0.0004 # (m) Mean thickness of pipe fin
    G.k_fin = 200 # (m)

    # These are the model inputs (correlations, etc). 
    M.Nu_CorrelationH = 5    # set correlation for heat transfer in H channel (1)-Yoon f(T_b), (2)-Yoon f(T_b,T_w), (3)-Gnielinski (4)-Pitla (5)-Wang
    M.alpha_CorrelationC = 1 # set correlation for heat transfer in C channel
    M.f_CorrelationH = 1 # set correlation for friction factor in H channel
    M.f_CorrelationC =[] # set correlation for friction factor in C channel 
    M.consider_bends = 1 # Consider bends in pressure loss? 1-yes, 0-no
    M.bend_loss_coefficient = 1 # Bend friction factor (from Chen 2018) (approximate, converted from Fanning to Darcy Weisbach)
    M.solver_type = 1  # Swtich for solver type  - (0) for pressure differential input, (1) for total mass flow input (2) for temperature output
    # Note: For this cooling tower model, M.solver must be 0 or 2.

    # These are the fluid properties. Note that TC_in is the ambient
    # temperature for the cooling tower model. 
    F.PF_fluid  = 'CO2'       
    F.Amb_fluid  = 'Air'
    F.T_PF_in  = 386.1437822 #  CO2 inlet temperature (K)
    F.T_PF_out = 273.15+40 # CO2 temperature outlet (K) 
    F.P_PF_in  = 7.96*10**6 # CO2 inlet pressure (Pa)
    
    F.TC_in  = 273.15+37 # Air ground level temperature (K) 
    F.vC_in  = 1.5 # Initialisation air velocity (m/s) - solved during run
    F.P_amb_in  = 101325 # Atmospheric pressure (Pa)
    F.TC_outlet_guess = 273.15+50 # Air side outlet guess for initialisation (K)
    F.P_PF_dp = 2000 # Process fluid pressure drop for cooling tower solver type 0

    F.mdot_PF = [] # Ignore this value in cooling tower model

    # These are the cooling tower inputs. Refer to my paper (and Sam Duniam's
    # paper/Kroger) for explanation of the paremters. 

    CT.R_Hd = 1.2 # Aspect ratio H5/d3
    CT.R_dD = 0.7 # Diameter ratio d5/d3
    CT.R_AA = 0.65 # Area coverage of heat exchangers Aft/A3
    CT.R_hD = 1/6.5 # Ratio of inlet height to diameter H3/d3
    CT.R_sd = (60//82.93)  # Ratio of number of tower supports to d3
    CT.R_LH = 15.78/13.67 # Ratio of height support to tower height L/H3
    CT.D_ts = 0.2 # Diameter of tower supports
    CT.C_Dts = 2 # Drag coefficeint of tower supports
    CT.K_ct = 0.1 # Loss coefficient of cooling tower separation
    CT.sigma_c = 0.725 # Sigma_c, per Kroger
    CT.dA_width = [] #(m2) Frontal area of section of HX (calculated within code)
    CT.solver_type = 0# Solver type: (0) - fixed diameter and HX solver BC, (1) - fixed mass flow rate, varying diameter (must use deltaT HX BC)
    CT.d3 = [] # Diameter of cooling tower inlet for initialisation
    CT.mdot_PF_total = 300.3046*0.711  # Total mass flow rate through the cooling tower

# ---------------------------------------------------------------------------
#
d3s = [52.86961232, 46.03457388, 41.72325061, 38.60877406, 36.18955625, 34.22272824, 32.57264609]

def perf_table(d3):
    mdot_array = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    Tamb_array = [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25]
    
    T_out_results = np.zeros(len(Tamb_array)*len(mdot_array))
    v_air_results = np.zeros(len(Tamb_array)*len(mdot_array))
    mdot_results = np.zeros(len(Tamb_array)*len(mdot_array))
    T_amb_results = np.zeros(len(Tamb_array)*len(mdot_array))
    
    vi = 1 ##Starting air velocity for initial case. Will be updated with solution after first run of model
    mdot_design = 300.3046*0.711
    k = 0
    T0 = [0]
    TC_out_i = 273.15+65.1
    T_PF_out_i = 273.15 + 50
    for j in range(len(mdot_array)):
        T0 = [0]
        TC_out_i = 273.15+60.1
        T_PF_out_i = 273.15 + 42.1
        for i in range(len(Tamb_array)):
            def mod(A):
                if hasattr(A, 'TC_in'):
                    A.TC_in = 273.15 + Tamb_array[i]
                if hasattr(A, 'vC_in'):
                    A.vC_in = vi
                if hasattr(A, 'TC_outlet_guess'):
                    A.TC_outlet_guess = TC_out_i
                if hasattr(A, 'mdot_PF_total'):
                    A.mdot_PF_total = mdot_design*mdot_array[j]
                if hasattr(A, 'd3'):
                    A.d3 = d3
            
            T_out, v_out, TC_out, T0_out = main(T0,mod)
            T_out_results[k] = T_out
            v_air_results[k] = v_out
            mdot_results[k] = mdot_array[j]*mdot_design
            T_amb_results[k] = Tamb_array[i]
            k = k+1
        
            vi= v_out
            TC_out_i = TC_out
            T0 = T0_out
        
    datestamp = datetime.datetime.now().strftime("%I%M%p_%B_%d_%Y")
    
    Result_data = [mdot_results, T_amb_results, T_out_results, v_air_results]
    Result_names = ["mdot", "Tamb", "T_out", "v_air"]
    
    Results = zip_longest(*Result_data,fillvalue='')
    Filename = "Cooling_tower_results"+str(d3)+datestamp+".csv"
    
    with open("Results\\"+Filename, "w",newline='') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(Result_names)
        for row in Results:
            writer.writerow(row)
main()

# ---------------------------------------------------------------------------            
# The below is an example of how you can run the model as a batch varying any
# parameters you want using nested "for" loops. The way to run as a batch is to
# create a small function called "mod()" and pass it to the main() function. This
# "mod()" function overwrites the values you want.

#Tamb_array = [38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25]
#d3_results = np.zeros(len(Tamb_array))
#T0 = [0]
#
#for i in range(len(Tamb_array)):
#    def mod(A):
#        if hasattr(A, 'TC_in'):
#            A.TC_in = Tamb_array[1] + 273.15
#    d3i = main(T0,mod)
#    d3_results[i] = d3i
#print(d3_results)

