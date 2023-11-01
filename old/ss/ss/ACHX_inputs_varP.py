
import AL_ACHX_Crossflow_NDDCT_CPmod_4eq_MP_const_P
import AL_ACHX_Crossflow_NDDCT_CPmod_4eq_MP_varP

from CoolProp.CoolProp import PropsSI as CP
import numpy as np
import csv
from itertools import zip_longest


def main():
   
    TC_out_ave, dp_amb_total, T1, Q_PF1, PF_dp, TH_final, dp_amb, alpha_i_final, TW_H_final, TW_C_final, Q3 = AL_ACHX_Crossflow_NDDCT_CPmod_4eq_MP_varP.main(ACHX_inputs,T0i = [0],fig_switch=0)

    Raw_results = [TH_final,TW_H_final,TW_C_final,Q3]

    Results = zip_longest(*Raw_results,fillvalue=' ')


    with open('Results\\Results5.csv', "w",newline='') as results_file:
        writer = csv.writer(results_file)
        for row in Results:
            writer.writerow(row)

    # def mod1(A):
    #     if hasattr(A, 'Nu_CorrelationH'):
    #         A.Nu_CorrelationH = 5

    # TC_out_ave, dp_amb_total, T1, Q_PF1, alpha_i1 = AL_ACHX_Crossflow_NDDCT_CPmod.main(ACHX_inputs,T0i = [0],fig_switch=0)

    # TC_out_ave1, dp_amb_total1, T11, Q_PF11, alpha_i11 = AL_ACHX_Crossflow_NDDCT_CPmod.main(ACHX_inputs,mod=mod1,T0i = [0],fig_switch=0)

    # print("Difference:",100*((Q_PF11-Q_PF1)/Q_PF1))

#     def mod2(A):
#         if hasattr(A, 'Nu_CorrelationH'):
#             A.Nu_CorrelationH = 2

#     def mod3(A):
#         if hasattr(A, 'Nu_CorrelationH'):
#             A.Nu_CorrelationH = 3

#     def mod4(A):
#         if hasattr(A, 'Nu_CorrelationH'):
#             A.Nu_CorrelationH = 4

#     TC_out_ave, dp_amb_total, T1, Q_PF1, alpha_i1 = AL_ACHX_Crossflow_NDDCT_CPmod.main(ACHX_inputs,mod=mod1,fig_switch=1)
#     TC_out_ave, dp_amb_total, T2, Q_PF2, alpha_i2 = AL_ACHX_Crossflow_NDDCT_CPmod.main(ACHX_inputs,mod=mod2,fig_switch=1)
#     TC_out_ave, dp_amb_total, T3, Q_PF3, alpha_i3 = AL_ACHX_Crossflow_NDDCT_CPmod.main(ACHX_inputs,mod=mod3,fig_switch=1)
#     TC_out_ave, dp_amb_total, T4, Q_PF4, alpha_i4 = AL_ACHX_Crossflow_NDDCT_CPmod.main(ACHX_inputs,mod=mod4,fig_switch=1)

#     print("Max alpha_i Yoon simple:", max(alpha_i1))
#     print("Max alpha_i Yoon full:", max(alpha_i2))
#     print("Max alpha_i Gneilinski:", max(alpha_i3))
#     print("Max alpha_i Pitla et al:", max(alpha_i4))

#     print("Average alpha_i Yoon_WC1_WC1_WC1 simple:", np.average(alpha_i1))
#     print("Average alpha_i Yoon full:", np.average(alpha_i2))
#     print("Average alpha_i Gneilinski:", np.average(alpha_i3))
#     print("Average alpha_i Pitla et al:", np.average(alpha_i4))

#     print("Tout Yoon simple:", T1[-1])
#     print("Tout Yoon full:", T2[-1])
#     print("Tout Gneilinski:", T3[-1])
#     print("Tout Pitla et al:", T4[-1])

#     print("Qout Yoon simple:",Q_PF1)
#     print("Qout Yoon full:", Q_PF2)
#     print("Qout Gneilinski:", Q_PF3)
#     print("Qout Pitla et al:", Q_PF4)


def ACHX_inputs(G,F,M):    
    G.HX_length = 10
    G.n_rows = 4  # - Number of rows (deep) of heat exchangerprint(q1)
    G.n_passes = 2
    G.pitch_longitudal = 0.04993 # - (m) Longitudal tube pitch 
    G.pitch_transverse =  0.05765  # - s(m) Transverse tube pitchPp
    G.ID = 0.0222  # (m) ID of HX pipe
    G.t_wall = 0.0016 # (m)c Thickness of pipe wall
    G.dx_i = 1.0001  #0.4 # (m) Initial length of pipe discretisation
    G.k_wall = 16 #14.192 # (W/m) Thermal conductivity of pipe wall
    G.D_fin = 0.05715  # (m) Maximum diameter of fin
    G.pitch_fin = 0.00245 # (m) Fin pitch
    G.t_fin = 0.0004 # (m) Mean thickness of pipe fin
    G.k_fin = 234 # (m)

    M.Nu_CorrelationH = 5    # set correlation for heat transfer in H channel (1)-Yoon f(T_b), (2)-Yoon f(T_b,T_w), (3)-Gnielinski (4)-Pitla (5)-Wang
    M.alpha_CorrelationC = 4 # set correlation for heat transfer in C channel
    M.f_CorrelationH = 1 # set correlation for friction factor in H channel
    M.f_CorrelationC =[] # set correlation for friction factor in C channel 
    M.consider_bends = 0 # Consider bends in pressure loss? 1-yes, 0-no
    M.bend_loss_coefficient = 0.5 # Bend friction factor (from Chen 2018) (approximate, converted from Fanning to Darcy Weisbach)
    M.solver_type = 0   # Swtich for solver type  - (0) for pressure differential input, (1) for total mass flow input 

    F.PF_fluid  = 'CO2'       
    F.Amb_fluid  = 'Air'
    F.T_PF_in  = 273.15+85 # (K)b
    F.P_PF_in  = 8*10**6 # 7.96 * 10**6 # 7.96*10**6 # 7.96*10**6 # pressure (Pa)
    F.P_PF_dp = 1000 # Process fluid pressure drop (Pa)
    F.mdot_PF = 0.1 # Total mass flow rate through the heat exchanger circuits combined (kg/s). Is initial guess if using pressure solver
    F.TC_in  = 273.15+25 # (K)
    F.vC_in  = 2 # (m/s)
    F.P_amb_in  = 101325 # pressure (Pa)
    F.TC_outlet_guess = 273.15+35 # Air side outlet guess for initialisation (K)

main()