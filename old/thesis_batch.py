import ACHX_generic_inputs as ACHX
from CoolProp.CoolProp import PropsSI as CP
import numpy as np
import csv
from itertools import zip_longest
import os
import datetime
import multiprocessing as mp
from scipy import optimize
from scipy.interpolate import interp1d

# filename = "Correlation_comparison_cycle1_28_NDDCT_V2"
filename = "C2_NDDCT_OD_30"

# variable = "Nu_CorrelationH"
variable = "TC_in"
TD = 273.15+30
values = np.linspace(TD,TD+15,16)
# values = np.linspace(273.15+22,273.15+37,16)
# values = ["Yoon-Simple","Yoon","Gnielinski","Pitla","Wang","Krasn.-1969","Liao","Zhang","Dang","Liu","New_correlation"]
n_tests = len(values)

data = np.empty((n_tests,5),dtype="object")
try:
    data = np.load(filename+".npy",allow_pickle=True)
except:
    print("Existing file does not exist. Creating a new file")

def run_model(i):

    def mod(A):
        if hasattr(A, variable):
            setattr(A,variable, values[i])

    print("[Running ",variable,"= ",values[i]," ]")
    try:
        mdot, d, UA, Tout = ACHX.main([0],mod,verbosity=0)
    except Exception as e:
        print("------------------------------------------------------------")
        print("Error for ",variable,"= ",values[i])
        print(e)
        print("------------------------------------------------------------")
    return(mdot,d,UA,Tout,i)

def log_result(result):
    mdot,d,UA,Tout,i = result
    data[i,0] = values[i]
    data[i,1] = mdot
    data[i,2] = d
    data[i,3] = UA
    data[i,4] = Tout
    np.save(filename,data)
    print("RESULT (",i,") SAVED")

def parallel_run():

    pool = mp.Pool(7)

    for i in range(n_tests):
        if not data[i,1]:
            pool.apply_async(run_model, args = (i, ), callback=log_result)

    pool.close()
    pool.join()

if __name__ == "__main__":
    parallel_run()
