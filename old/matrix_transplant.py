import numpy as np

old_P = np.load("NDDCT_case-20_points_V4.npy",allow_pickle=True)
old_D = np.load("NDDCT_case-20_data_V4.npy")

add_P = np.load("NDDCT_case-20_points_11MPa.npy",allow_pickle=True)
add_D = np.load("NDDCT_case-20_data_11MPa.npy")


print(old_P)
# print(add_D)
print(np.shape(add_D))

mdot_design = 344.4*0.643
Tamb_array = np.array([30, 32 , 34, 36, 38, 40, 42])+273.15
mdot_array = np.array([0.5,0.6, 0.8, 1, 1.2, 1.4])*344.4*0.643
Tin_array = np.array([70, 80, 100, 120])+273.15
pin_array = np.array([7.8, 8,8.5,9,9.5,10,11])*1e6


CTpoints = np.asarray([Tamb_array,mdot_array,Tin_array,pin_array])
np.save("NDDCT_case-20_points",CTpoints)
CTdata = np.zeros((len(Tamb_array),len(mdot_array),len(Tin_array),len(pin_array)))

for i in range(len(Tamb_array)):
    for j in range(len(mdot_array)):
        for k in range(len(Tin_array)):
            for l in range(len(pin_array)):
                # if l==5:
                    # CTdata[i,j,k,l] = add_D[i,j,k,0]
                # if l==6:
                    # CTdata[i,j,k,l] = old_D[i,j,k,6]
                if l==6:
                    CTdata[i,j,k,l] = old_D[i,j,k,6]
                else:
                    CTdata[i,j,k,l] = old_D[i,j,k,l]

np.save("NDDCT_case-20_data_V4",CTdata)
np.save("NDDCT_case-20_points_V4",CTpoints)

