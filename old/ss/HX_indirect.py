import numpy as np
from CoolProp.CoolProp import PropsSI as CP
import matplotlib.pyplot as plt

def HX_PP_analysis(T_PF_in,T_PF_out,PP,P_PF=8*10**6,P_CF=1.5*10**5):

	x = 100

	P_PF = 8*10**6
	P_CF = 2*10**5

	fluid_PF = 'CO2'
	fluid_CF = 'H2O'

	T_PF = np.zeros(x+1)
	T_PF[1] = T_PF_in
	H_PF_in = CP('H','T',T_PF_in,'P',P_PF,fluid_PF)
	H_PF_out = CP('H','T',T_PF_out,'P',P_PF,fluid_PF)
	deltaH = H_PF_in - H_PF_out
	PF_dH = deltaH/x
	H_PF = np.zeros(x+1)
	H_CF = np.zeros(x+1)
	PP_it = np.zeros(x+1)
	T_CF = np.zeros(x+1)

	mdot_R = 1

	T_CF_in = T_PF_out - PP # Value to iterate

	while abs(PP_it[x]-5)>0.01:
		PP_it[0:x] = 0
		while abs(min(PP_it) - PP)>0.01:

			H_CF_in = CP('H','T',T_CF_in,'P',P_CF,fluid_CF)
			H_CF_out = H_CF_in + (deltaH/mdot_R)
			i=0
			for i in range(x+1):
				H_PF[i] = H_PF_in - (PF_dH*i)
				T_PF[i] = CP('T','H',H_PF[i],'P',P_PF,fluid_PF)
				H_CF[i] = H_CF_out - (PF_dH*i/mdot_R)
				T_CF[i] = CP('T','H',H_CF[i],'P',P_CF,fluid_CF)
				PP_it[i] = T_PF[i] - T_CF[i]

			T_CF_in = T_CF_in - (PP-min(PP_it))

			print("Current iterative pinch point",min(PP_it))
		print("Coolant inlet temperature",T_CF_in-273.15)
		print("Coolant outlet temperature",T_CF[0]-273.15)

		dT_PF_inlet = T_PF[x-1]-T_PF[x]
		dT_CF_inlet = T_CF[x-1]-T_CF[x]
		dT_PF_dT_CF = dT_PF_inlet/dT_CF_inlet
		mdot_R = mdot_R / dT_PF_dT_CF
		
		print("New CF mass flow rate:",mdot_R)
		print("Inlet pinch point",PP_it[x])

		if abs(dT_PF_dT_CF - 1) < 0.001:
			break

	# l_ND = np.linspace(0,1,x+1) 
	# plt.plot(l_ND,T_PF,l_ND,T_CF)
	# plt.show()
	return (T_CF[0],T_CF[x],mdot_R)



