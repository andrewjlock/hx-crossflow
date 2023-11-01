import csv 
import numpy as np
from matplotlib import pyplot as plt

filenames = ["Cycle2.csv","Cycle1.csv"]
colors = ['r','k']

hs = []
fig,ax = plt.subplots(figsize=(5,3.5))
for filename,color in zip(filenames,colors):

        data = []

        with open(filename, mode='r') as infile:
                reader = csv.reader(infile,delimiter=",")
                for i,row in enumerate(reader):
                        if i == 0:
                                headers = row
                        else:
                                data.append(row)
                                
        data = np.asarray(data).T
        data1 = []
        data2 = []

        for i,d in enumerate(data):
                data1.append([float(x) for x in d if len(x) !=0])

        mydict  = {}
        for header, d in zip(headers, data1):
                mydict[header] = [x for x in d if x!=0]

        

        var = "Bu_j"

        x = mydict['x']
        y = mydict[var]

        y = [x if x > 1e-2 else np.nan for x in y]
        # y = y[0:-1]

        xs = []
        ys = []

        for i in range(4):
                xs.append(x[int(len(x)/4*i):int(len(x)/4*(i+1))])
                ys.append(y[int(len(y)/4*i):int(len(y)/4*(i+1))])

        for x in xs:
                x = [0]+[(x1+x2)/2 for x1,x2 in zip(x[0:-1],x[1:])]

        h1, = ax.plot(xs[0],ys[0],color=color,marker='o',markersize=3,markerfacecolor=None,linewidth=1,label="Top pass")
        h2, = ax.plot(xs[1],ys[1],'--',color=color,marker='s',markersize=3,markerfacecolor=None,linewidth=1,label="Bottom pass")
        ax.plot(xs[2],ys[2][::-1],color=color,markersize=3,marker='o',markerfacecolor=None,linewidth=1)
        ax.plot(xs[3],ys[3][::-1],'--',color=color,marker='s',markersize=3,markerfacecolor=None,linewidth=1)
        hs.append(h1)
        hs.append(h2)

ax.set_xlabel(r"Flow path position [m]")
# ax.set_ylabel(r"$h_\mathrm{CO_2}$")
# ax.set_ylabel(r"$Ri$")
# ax.set_ylabel(r"$Bu_\mathrm{p}$")
ax.set_ylabel(r"$Bu_\mathrm{J3}$")
# ax.set_ylim(1*10**(-1.5),1)
ax.set_yscale('log')
# plt.legend((hs[2],hs[3],hs[0],hs[1]),("Cycle 1: Top pass","Cycle 1: Bottom pass","Cycle 2 Top pass","Cycle 2: Bottom pass"),fontsize=8)
plt.tight_layout()
plt.savefig("Buj_profile",dpi=300)
plt.show()

