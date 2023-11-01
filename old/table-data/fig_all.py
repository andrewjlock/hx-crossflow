import numpy as np
from matplotlib import pyplot as plt

# The vector order is  (1) Tamb, (2) mdot, (3) Tin, (4) p_in
# CTdataFileName = "NDDCT_data_varP_full.py"
# exec(open(CTdataFileName).read(), globals(), locals())


# points = np.load("NDDCT_points_flipped.npy",allow_pickle=True)
# data = np.load("NDDCT_data_flipped.npy")-273.15

# points = np.load("NDDCT_case-20_points_V3.npy",allow_pickle=True)
# data = np.load("NDDCT_case-20_data_V3.npy")-273.15


# Tambs = [x-273.15 for x in points[0]]
# mdots = points[1]
# Tins = [x-273.15 for x in points[2]]
# pins = [x/1e6 for x in points[3]]

# plt.figure(figsize=(4,3.5))
# params = {'mathtext.default': 'regular' }
# plt.rcParams.update(params)
# x = Tambs
# y = mdots 
# X, Y = np.meshgrid(x, y)
# Z = data[:,:,1,3].T
# ax1=plt.axes(projection='3d')
# ax1.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
# ax1.set_xlabel(r"$T_{amb}\,(^\circ C)$")
# ax1.set_ylabel(r"$\dot{m}\, (kg/s)$")
# ax1.set_zlabel(r"$T_{out}\, (^\circ C)$")
# # ax1.set_title("Tin = 100 C, Pin = 9 MPa")
# ax1.set_xlim(ax1.get_xlim()[::-1])
# plt.pause(0.1)
# plt.draw()
# plt.tight_layout()
# plt.savefig("NDDCT1",bbox_inches = 'tight',pad_inches = 0,dpi=300)


points = np.load("NDDCT_case-20_points_V4.npy",allow_pickle=True)
data = np.load("NDDCT_case-20_data_V4.npy")-273.15


Tambs = [x-273.15 for x in points[0]]
mdots = points[1]
Tins = [x-273.15 for x in points[2]]
pins = [x/1e6 for x in points[3]]

plt.figure(figsize=(4,3.5))
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)
x = Tambs
y = mdots 
X, Y = np.meshgrid(x, y)
Z = data[:,:,4,-2].T
ax1=plt.axes(projection='3d')
ax1.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
# ax1.plot_surface(X,Y,Z1,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
ax1.set_xlabel(r"$T_{amb}\,(^\circ C)$")
ax1.set_ylabel(r"$\dot{m}\, (kg/s)$")
ax1.set_zlabel(r"$T_{out}\, (^\circ C)$")
# ax1.set_title("Tin = 100 C, Pin = 9 MPa")
ax1.set_xlim(ax1.get_xlim()[::-1])
plt.pause(0.1)
plt.draw()
plt.tight_layout()
plt.savefig("NDDCT1",bbox_inches = 'tight',pad_inches = 0,dpi=300)


plt.figure(figsize=(4,3.5))
x = pins #Tins
y = Tins #pins
X, Y = np.meshgrid(x, y)
Z = data[0,3,:,:]
ax2=plt.axes(projection='3d')
ax2.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
ax2.set_ylabel(r"$T_{in}\,(^\circ C)$")
ax2.set_xlabel(r"$P_{in}\,(MPa)$")
ax2.set_zlabel(r"$ T_{out}\,(^\circ C)$")
# ax2.set_title("Tamb = 30C, mdot = 200 kg/s")
ax2.set_xlim(ax2.get_xlim()[::-1])
plt.tight_layout()
# plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.savefig("NDDCT3_",bbox_inches = 'tight',pad_inches = 0,dpi=300)

plt.figure(figsize=(4,3.5))
x = mdots
y = pins
X, Y = np.meshgrid(x, y)
Z = data[0,:,1,:].T
ax3=plt.axes(projection='3d')
ax3.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
ax3.set_xlabel(r"$\dot{m}\,(kg/s)$")
ax3.set_ylabel(r"$P_{in}\,(MPa)$")
ax3.set_zlabel(r"$T_{out}\,(^\circ C)$")
# ax3.set_title("Tamb = 30C, Tin = 100 C")
ax3.set_xlim(ax3.get_xlim()[::-1])
plt.tight_layout()
plt.savefig("NDDCT2",bbox_inches = 'tight',pad_inches = 0,dpi=300)

# print(points[0][6])
# print(points[2][3])

plt.figure(figsize=(4,3.5))
x = Tambs
y = Tins
X, Y = np.meshgrid(x, y)
Z = data[:,3,:,-1].T
ax4=plt.axes(projection='3d')
ax4.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
ax4.set_xlabel(r"$T_{amb}\,(^\circ C)$")
ax4.set_ylabel(r"$T_{in}\,(^\circ C)$")
ax4.set_zlabel(r"$T_{out}\,(^\circ C)$")
# ax4.set_title("mdot = 200 kg/s, Pin = 9 MPa")
ax4.set_xlim(ax4.get_xlim()[::-1])
plt.tight_layout()
plt.savefig("NDDCT4",bbox_inches = 'tight',pad_inches = 0,dpi=300)


plt.pause(0.1)
plt.draw()
input("Close")
plt.close()



