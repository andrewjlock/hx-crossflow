import numpy as np
from matplotlib import pyplot as plt

# Input data for HX area fApproachfiles = ["LT_cycle_forced.npy","HT_cycle_forced_v1.npy","LT_cycle_NDDCT.npy","HT_cycle_NDDCT_v1.npy"]
files = ["C1_forced_approach.npy","C2_forced_approach.npy","C1_NDDCT_approach.npy","C2_NDDCT_approach.npy"]
cooler_types = ["forced","forced","NDDCT","NDDCT"]
mdot_totals = [154.3,221.2,154.3,221.2]
cycle_nos = [1,2,1,2]
# fig_labels = ["C1_forced_approach","C2_forced_approach","C1_NDDCT_approach","C2_NDDCT_approach"]
fig_labels = ["Cycle 1 forced convection","Cycle 2 forced convection","Cycle 1 NDDCT","Cycle 2 NDDCT"]
T1s = np.linspace(15,30,16)
T2s = np.linspace(22,37,16)
Tamb_ds = [T1s,T2s,T1s,T2s]
TCI = [33,40,33,40]

# Input data for correlation comparison figure
# prefix = "Correlation_comparison_"
# suffixes = ["cycle1_28_NDDCT_V2.npy","cycle1_28_forced_V2.npy","cycle2_35_NDDCT_V2.npy","cycle2_35_forced_V2.npy"]
# files = [prefix + f for f in suffixes]
# cooler_types = ["NDDCT","forced","NDDCT","forced"]
# cycle_nos = [1,1,2,2]
# mdot_totals = [154.3,154.3,221.2,221.2]
# Tamb_ds = [25,28,35,35]
# fig_labels = ["Yoon-Simple","Yoon","Gnielinski","Pitla","Wang","Krasn.-1969","Liao","Zhang","Dang","Liu","New_correlation"]
fig_labels1 = ["Yoon et al. (2)","Yoon et al. (1)","Gnielinski","Pitla et al.","Wang et al.","Krasnoshchekov et al.","Liao and Zhao","Zhang et al.","Dang and Hihara","Liu et al.","Proposed correlation"]
fig_labels2 = [r"Cycle 1, NDDCT ($T_\mathrm{amb}=28 ^\circ$C)",r"Cycle 1, forced conv. ($T_\mathrm{amb}=28 ^\circ$C)", r"Cycle 2, NDDCT ($T_\mathrm{amb}=35 ^\circ$C)",r"Cycle 2, forced ($T_\mathrm{amb}=35\ ^\circ$C)"]

labels = []

def main():

    results = []
    for i in range(len(files)):
        try:
            results.append(result(files[i],cooler_types[i],cycle_nos[i],mdot_totals[i],fig_labels[i],Tamb_ds[i]))
        except Exception as e:
            print(files[i],e)

    plot(results)
    compare(results)


class result:
    def __init__(self,file,cooler_type,cycle_no,mdot_total,label,Tamb_d):
        markers = {1:"s",2:"o"}
        lines = {1:"-",2:"--"}
        markerfills = {"NDDCT":"k","forced":"w"}
        TCIs = {1:33,2:40}
        self.file = file
        self.cooler_type = cooler_type
        self.label = label
        self.mdot_total = mdot_total
        self.Tamb_d = Tamb_d
        self.A_tube = 12 * 0.0635
        self.A_rat = 0.65
        self.marker = markers[cycle_no]
        self.line = lines[cycle_no]
        self.markerfill = markerfills[cooler_type]
        self.TCI = TCIs[cycle_no]
        self.extract_results()
        self.process()
    
    def extract_results(self):
        self.data = np.load(self.file,allow_pickle=True)
        self.x = self.data[:,0]
        self.UA = self.data[:,3]
    
    def process(self):
        if self.cooler_type == "forced":
            self.A_f = [(self.mdot_total/m)*self.A_tube for m in self.data[:,1]]
        if self.cooler_type == "NDDCT":
            self.A_f = [np.pi * 0.25 * d**2 * self.A_rat for d in self.data[:,2]]
        self.approach = [self.TCI-x for x in self.Tamb_d]
        print(self.approach)

def plot(results):

    # results.sort(key=lambda x: np.max(x.UA), reverse=True)

    # markers = ['s','s','o','o']
    # markerfills = ['w','k','w','k']
    # lines = ['-','-','--','--']

    # fig,ax = plt.subplots(figsize=(7,5))
    fig,ax = plt.subplots(figsize=(6,4))

    # HX area plot

    data = [[] for i in range(len(fig_labels))]
    for r in results:
        for i,label in enumerate(fig_labels):
            if label in r.x:
                data[i].append(r.UA[np.where(r.x == label)[0][0]])
            else:
                data[i].append(0)

    # for r in results:
        ax.plot(r.approach,r.A_f,r.line,marker=r.marker,label=r.label,linewidth=1,color='k',markersize=5,markerfacecolor=r.markerfill)


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
    ax.set_xlabel(r"Approach temperature $(T_\mathrm{CI} - T_\mathrm{amb})$ [$^\circ$C]")
    # ax.set_xlabel(r"Ap")
    # ax.set_ylabel(r"Heat exchanger face area [m$^2$]")
    # ax.set_ylabel(r"Overall heat exchanger conductance [W/K]")
    ax.set_ylabel(r"Heat exchanger frontal area [m$^2$]")
    ax.set_ylim(0,2600)
    plt.legend()
    # ax.legend(loc='lower right',framealpha=1)
    plt.tight_layout()
    plt.pause(0.1)
    plt.draw()
    input("Press key to close")
    plt.savefig("Approach",dpi=300)
    plt.close()

def compare(results):
    NDDCT = []
    forced = []
    for r in results:
        A_max = np.max(r.A_f)
        A_min = np.min(r.A_f)
        print("For case",r.file)
        print("max:",A_max,", A min:",A_min,", A ratio:", A_max/A_min)

        UA_max = np.max(r.UA)
        UA_min = np.min(r.UA)
        print("max:",UA_max,", UA min:",UA_min,", UA ratio:", UA_max/UA_min)
        print("---------------------------------------------")

        if r.cooler_type == "NDDCT":
            NDDCT.append(np.mean(r.A_f))
        else:
            forced.append(np.mean(r.A_f))

    NDDCT_average = np.mean(NDDCT)
    forced_average = np.mean(forced)

    print("NDDCT ave:",NDDCT_average,", forced ave:",forced_average,", ratio:",NDDCT_average/forced_average)


main()




