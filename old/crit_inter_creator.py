import numpy as np
import CoolProp.CoolProp as CPCP
from scipy import optimize
from scipy import interpolate as inter

def crit_temp_object():
    # Returns an interpolation object used to find critical point
    def Cp(T,P):
        # return CP('C','T',T,'P',P,'CO2')
        HEOS = CPCP.AbstractState("HEOS","CO2")
        HEOS.update(CPCP.PT_INPUTS,P,T)
        C = HEOS.cpmass()
        return C

    P_array = [7.4,7.5,7.6,7.8,8,8.5,9,9.5,10]
    T_array = []
    for i in range(len(P_array)):
        Tcrit = optimize.fmin(lambda T:-1*Cp(T,(P_array[i]*1e6)+101325),273.15+40,xtol=1e-8,disp=False)    
        T_array.append(Tcrit[0])
    crit_inter = inter.interp1d(P_array,T_array,fill_value="extrapolate")
    return crit_inter
