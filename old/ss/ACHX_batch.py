import ACHX_inputs
import ACHX_solver
from CoolProp.CoolProp import PropsSI as CP
import numpy as np
import csv
from itertools import zip_longest
import os
import datetime
import multiprocessing as mp
from scipy import optimize
from scipy.interpolate import interp1d


mdot_total_design = 344.4*0.643
mdot_tuberow_design = 0.29091

Tamb_array = np.array([30, 32 , 34, 36, 38, 40])+273.15
mdot_array = np.array([0.6, 0.8, 1, 1.2, 1.4])*mdot_total_design
Tin_array = np.array([70, 80, 100, 120])+273.15
pin_array = np.array([8, 8.5, 9, 9.5])*1e6
# d3 = 49.19553

P_array = np.linspace(8,30,20)*1e6
def PC_point_array(P_array):
    def Cp(T,P):
        return -CP('C','T',T,'P',P,'CO2')
    T_crits = []
    for i in range(len(P_array)):
        Tcrit = optimize.fmin(lambda T:Cp(T,P_array[i]),273.15,xtol=1e-8,disp=0)    
        T_crits.append(Tcrit[0])
    return T_crits
T_crits = PC_point_array(P_array)
P_interp = interp1d(P_array,T_crits)

def run_model(indexes):
    i,j,k,l = indexes
    Tcrit = P_interp(pin_array[l])
    # Small calculation for ballpark estiamte of outlet conditions for
    # initialsation
    hA = 4800
    skip=False
    try:
        T_out_CO2 = max(Tcrit+1,Tamb_array[i]+5)    
        h_out_CO2 = CP('H','T',T_out_CO2,'P',pin_array[l],'CO2')
        Q = (CP('H','P',pin_array[l],'T',Tin_array[k],'CO2') -h_out_CO2)*mdot_array[j]*(mdot_tuberow_design/mdot_total_design)
        E_in_air = CP('H','P',101325,'T',Tamb_array[i],'air')*CP('D','P',101325,'T',Tamb_array[i],'air')*3*12*0.0635
        h_out_air = (E_in_air + Q)/(CP('D','P',101325,'T',Tamb_array[i],'air')*3*12*0.0635)
        T_out_air = CP('T','P',101325,'H',h_out_air,'air')
    except Exception as e:
        print(e)

    def mod(A):
        if hasattr(A, 'TC_in'):
            A.TC_in = Tamb_array[i]
        if hasattr(A, 'mdot_PF_total'):
            A.mdot_PF_total = mdot_array[j]*mdot_tuberow_design/mdot_total_design
        if hasattr(A, 'T_PF_in'):
            A.T_PF_in = Tin_array[k]
        if hasattr(A, 'P_PF_in'):
            A.P_PF_in = pin_array[l]
        if hasattr(A, 'TC_outlet_guess'):
            A.TC_outlet_guess = T_out_air
        if hasattr(A, 'T_PF_out'):
            A.T_PF_out =T_out_CO2 
    # mod,indexes = args
    model_no = i*len(mdot_array)*len(Tin_array)*len(pin_array) + j*len(Tin_array)*len(pin_array) + k*len(pin_array)+l+1
    total_models = len(Tamb_array)*len(mdot_array)*len(Tin_array)*len(pin_array)
    try:
        print("[Running ",model_no," of ",total_models,". Tamb=",Tamb_array[i],", mdot=",mdot_array[j],", Tin=",Tin_array[k],", p_in=",pin_array[l],", T0i_CO2=",T_out_CO2,"', T0i_air=",T_out_air,"]")
        T_out = ACHX_inputs.main(mod=mod,verbosity=0)
        print("Model No.",model_no," solved successfully")
    except Exception as e:
        print("------------------------------------------------------------------------------------------")
        print("Error for Tamb=",Tamb_array[i],", mdot=",mdot_array[j],", Tin=",Tin_array[k],", p_in=",pin_array[l])
        print(e)
        print("------------------------------------------------------------------------------------------")
        T_out = 0
    return(T_out,indexes)
    # return k
    
# ACHXdata = np.zeros((len(Tamb_array),len(mdot_array),len(Tin_array),len(pin_array)))
ACHXdata = np.load("ACHX_case-20_data.npy") #np.zeros((len(Tamb_array),len(mdot_array),len(Tin_array),len(pin_array)))
def log_result(result):
    T_out,indexes = result
    i,j,k,l = indexes
    ACHXdata[i,j,k,l] = T_out
    np.save("ACHX_case-20_data",ACHXdata)
    print("RESULT ARRAY SAVED")

def perf_table():

    pool = mp.Pool(11)

    ACHXpoints = np.asarray([Tamb_array,mdot_array,Tin_array,pin_array])
    np.save("ACHX_case-20_points",ACHXpoints)
    old_data = np.load("ACHX_case-20_data.npy")

    for i in range(len(Tamb_array)):
        for j in range(len(mdot_array)):
            for k in range(len(Tin_array)):
                for l in range(len(pin_array)):
                    indexes = (i,j,k,l) 
                    if old_data[i,j,k,l] in (0.,'e'):
                        pool.apply_async(run_model, args = (indexes,), callback=log_result)
                    # run_model((1,1,1,1))
    pool.close()
    pool.join()

if __name__ == '__main__':
    perf_table()

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

