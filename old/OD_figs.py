import numpy as np
from matplotlib import pyplot as plt
from CoolProp.CoolProp import PropsSI as CP
from tabulate import tabulate

files = ["C1_NDDCT_OD_23.npy","C1_NDDCT_OD_28.npy","C1_forced_OD_23.npy","C1_forced_OD_28.npy","C2_NDDCT_OD_30.npy","C2_NDDCT_OD_35.npy","C2_forced_OD_30.npy","C2_forced_OD_35.npy"]
# cooler_types = ["NDDCT","NDDCT","NDDCT","NDDCT"]

# labels = ["Cycle 1

# mdot_totals = [154.3,221.2,154.3,221.2]
cycle_nos = [1,2,1,2,3,4,3,4]
# fig_labels = ["C1_forced_approach","C2_forced_approach","C1_NDDCT_approach","C2_NDDCT_approach"]
# fig_labels = ["Cycle 1 forced convection","Cycle 2 forced convection","Cycle 1 NDDCT","Cycle 2 NDDCT"]
fig_labels = files
T1s = np.linspace(23,23+15,16)
T2s = np.linspace(28,28+15,16)
T3s = np.linspace(30,30+15,16)
T4s = np.linspace(35,35+15,16)
Tambs = [T1s,T2s,T1s,T2s,T3s,T4s,T3s,T4s]
Td = [23,28,23,28,30,35,30,35]

# Input data for correlation comparison figure
# prefix = "Correlation_comparison_"
# suffixes = ["cycle1_28_NDDCT_V2.npy","cycle1_28_forced_V2.npy","cycle2_35_NDDCT_V2.npy","cycle2_35_forced_V2.npy"]
# files = [prefix + f for f in suffixes]
# cooler_types = ["NDDCT","forced","NDDCT","forced"]
# cycle_nos = [1,1,2,2]
# mdot_totals = [154.3,154.3,221.2,221.2]
# Tamb_ds = [25,28,35,35]
# fig_labels = ["Yoon-Simple","Yoon","Gnielinski","Pitla","Wang","Krasn.-1969","Liao","Zhang","Dang","Liu","New_correlation"]

labels = []

def main():

    results = []
    for i in range(len(files)):
        try:
            results.append(result(files[i],cycle_nos[i],fig_labels[i],Tambs[i]))
        except Exception as e:
            print(files[i],e)

    plot(results)
    compare(results)


class result:
    def __init__(self,file,cycle_no,label,Tambs):
        markers = {1:"s",2:"o",3:"v",4:"^"}
        # lines = {1:"-",2:"-",3:"--",4:"--"}
        lines = {"NDDCT":"-","forced":"--"}
        pressures = {1:8e6,2:8e6,3:9e6,4:9e6}
        TCIs = {1:33,2:33,3:40,4:40}
        Tamb_d = {1:23,2:28,3:30,4:35}
        Tins  = {1:65.96,2:65.96,3:81.4,4:81.4}
        markerfills = {"NDDCT":"k","forced":"w"}
        if "NDDCT" in file:
            self.cooler_type = "NDDCT"
        elif "forced" in file:
            self.cooler_type = "forced"
        else:
            print("ERROR: no cooler type identified")
        self.p = pressures[cycle_no]
        self.file = file
        self.label = label
        self.Tamb_d = Tamb_d[cycle_no]
        self.A_tube = 12 * 0.0635
        self.A_rat = 0.65
        self.marker = markers[cycle_no]
        self.TCI = TCIs[cycle_no]
        # self.line = lines[cycle_no]
        self.line = lines[self.cooler_type]
        self.markerfill = markerfills[self.cooler_type]
        self.Tambs = Tambs
        self.Tin = Tins[cycle_no] + 273.15
        self.extract_results()
        self.process()
    
    def extract_results(self):
        self.data = np.load(self.file,allow_pickle=True)
        self.T_outs = self.data[:,4]
        
        # self.UA = self.data[:,3]
    
    def process(self):
        self.hin = CP('H','T',self.Tin,'P',self.p,'CO2')
        self.houts = []
        for T in self.T_outs:
            self.houts.append(CP('H','T',T,'P',self.p,'CO2'))

        self.amb_dev = [x-self.Tamb_d for x in self.Tambs]
        self.CO2_dev = [x-self.TCI-273.15 for x in self.T_outs]

        self.h_dev = [100*(self.hin-x)/(self.hin-self.houts[0]) for x in self.houts]


def plot(results):

    fig1,ax = plt.subplots(2,figsize=(6,8),sharex="col")
    # fig2,ax2 = plt.subplots(figsize=(6,4))
    ax.flatten()

    # data = [[] for i in range(len(fig_labels))]
    handles1 = []
    handles2 = []
    for r in results:


    # for r in results:
        h, = ax[0].plot(r.amb_dev,[x-273.15 for x in r.T_outs],r.line,marker=r.marker,label=r.label,linewidth=1,color='k',markersize=5,markerfacecolor=r.markerfill)
        ax[1].plot(r.amb_dev,r.h_dev,r.line,marker=r.marker,label=r.label,linewidth=1,color='k',markersize=5,markerfacecolor=r.markerfill)

        if r.cooler_type == "NDDCT":
            handles1.append(h)
        if r.Tamb_d == 23:
            handles2.append(h)



    # data = np.array(data)
    # data_OG = data[:,:]
    # order = data[:,3].argsort()
    # print(order)
    # data = data[data[:,3].argsort()].T
    # fig_labels1 = np.array(fig_labels1)[order]

    # colors = ["black","dimgrey","darkgrey","lightgrey"]
    # width = 0.5
    # for i,(C,color,label1) in enumerate(zip(data,colors,fig_labels2)):
        # # for j,A_f in enumerate(C):
        # ax.bar(np.arange(len(fig_labels))-(width/2)+((i/(len(results)-1))*width),C,width/3,color=color,label=label1)

    # ax.set_xticks(np.arange(len(fig_labels)))
    # ax.set_xticklabels(fig_labels1)
    # plt.xticks(rotation=45, ha='right')


    # Correlation plot 
    # x = np.arrange(len(fig_labels))


        # ,r.line,marker=r.marker,label=r.label,linewidth=1,color='k',markersize=5,markerfacecolor=r.markerfill)


    # params = {'mathtext.default': 'regular' }
    ax[1].set_xlabel(r"Ambient temperature deviation [$^\circ$C]")
    # ax.set_xlabel(r"Ap")
    # ax.set_ylabel(r"Heat exchanger face area [m$^2$]")
    # ax.set_ylabel(r"Overall heat exchanger conductance [W/K]")
    ax[0].set_ylabel(r"CO$_2$  outlet temperautre [$^\circ$C]")

    # ax[0].set_xlabel(r"Ambient temperature deviation [$^\circ$C]")
    ax[1].set_ylabel(r"Design-point heat rejection fraction [%]")

    labels1 = ["Cycle 1: $T_\mathrm{amb,d}$=23 $^\circ$C","Cycle 1: $T_\mathrm{amb,d}$=28 $^\circ$C","Cycle 2: $T_\mathrm{amb,d}$=30 $^\circ$C","Cycle 2: $T_\mathrm{amb,d}$=35 $^\circ$C"]
    labels2 = ["NDDCT","forced"]

    ax[0].legend(handles=handles1,labels=labels1)
    ax[1].legend(handles=handles2,labels=labels2)
    # ax.legend(loc='lower right',framealpha=1)
    plt.tight_layout()
    plt.pause(0.1)
    plt.draw()
    input("Press key to close")
    plt.savefig("OD_in_const",dpi=300)
    plt.close()

def compare(results):
    for r in results:
        print("-------",r.file,"---------")
        data = zip(r.amb_dev,r.h_dev)
        print(tabulate(data,headers=["Amb dev","Q_dev"]))




main()




