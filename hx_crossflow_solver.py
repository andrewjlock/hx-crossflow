
""" Element-based cross-flow heat exchanger model for complex fluids

Author: Andrew Lock
Date refactored: October 2023

Note: This is mostly old and poorly witten code. 
One day it will be re-written with best practices. 

Refer to publication for details: 10.1016/j.applthermaleng.2019.114390
"""

import numpy as np
import CoolProp as CPCP
from CoolProp.CoolProp import PropsSI as CP
import scipy as sci
from scipy import optimize
import matplotlib.pyplot as plt
from statistics import mean
from scipy.interpolate import interp1d


global EosPF

# ------------------PROPERTY CLASSES-------------------------------------------


class Model:
    # Create model detail classes
    def __init__(self):
        self.Nu_CorrelationH = []  # set correlation for heat transfer in H channel
        self.alpha_CorrelationC = []  # set correlation for heat transfer in C channel
        self.row_correction = []
        self.f_CorrelationH = []  # set correlation for friction factor in H channel
        self.f_CorrelationC = []  # set correlation for friction factor in C channel
        self.alpha_o = []  # (W/m2)
        self.iteration_no = 0  # Iteration counter
        self.solver_figure = []  # Switch for solver figure
        self.final_figure = []  # Switch for final figures
        self.consider_bends = []  # Switch for considering bends in pressure loss
        self.bend_loss_coefficient = []  # Friction factor for pressure loss in bends
        # Swtich for solver type:
        # (0) for pressure differential input,
        # (1) for total mass flow input
        self.solver_type = []
        # Switch for cooler type
        self.cooler_type = []


class Geometry:
    def __init__(self):
        self.dx_i = []  # number of cells
        self.k_wall = []  # thermal conductivity (W / m)
        self.l_hx_th = []  # (m)
        self.HX_length = []
        self.n_rows = []
        self.n_passes = []
        self.pitch_longitudal = []  # - (m) Streamwise tube pitch
        self.pitch_transverse = []  # - (m) Normal to stream tube pitch (X,Y)
        self.ID = []  # (m) ID of HX pipe
        self.t_wall = []  # (m) Pipe wall thickness
        self.pitch_fin = []  # (m) Pitch spacing of fins
        self.D_fin = []  # (m) Fin diameter
        self.t_fin = []  # (m) Fin thickness
        self.k_fin = []  # (W/m) Fin matieral thermal conductivity

    #  def read_yaml(self, filepath):

    def micro_init(self):
        self.l_hx_th = self.HX_length
        self.n_CPR = int(self.l_hx_th // self.dx_i + 1)  # Cells per row
        self.n_cells = self.n_CPR * self.n_rows
        self.n_CP = int(self.n_rows / self.n_passes)  # Number of adjacent co-flow rows
        if self.n_rows % self.n_passes != 0:
            raise MyError("Incorrect pipework arrangement")
        self.dx = self.l_hx_th / self.n_CPR
        self.A_amb = self.pitch_transverse * self.dx  # Air entry/exit area
        self.A_CS = np.pi * (self.ID**2) / 4  # Internal cross sectional area of pipe
        self.tube_OD = self.ID + (2 * self.t_wall)
        self.A_WCS = np.pi * (1 / 4) * ((self.tube_OD**2) - (self.ID**2))
        self.A_tube_o = float(np.pi * self.tube_OD * self.dx)
        self.A_tube_i = float(np.pi * self.ID * self.dx)
        # Equation 3.3.13 fro Kroger
        self.eta_phi = ((self.D_fin / self.tube_OD) - 1) * (
            1 + (0.35 * (np.log(self.D_fin / self.tube_OD)))
        )
        self.bend_path_length = np.pi * self.pitch_transverse / 2

        if self.D_fin:
            # Area of the fins only
            self.A_f = (self.dx / self.pitch_fin) * (
                (((self.D_fin**2) - (self.tube_OD**2)) * np.pi / 2)
                + self.D_fin * self.t_fin
            )

            # Area of the root between fins
            self.A_r = self.A_tube_o - (
                (self.dx / self.pitch_fin) * self.t_fin * self.tube_OD * np.pi
            )

            # Total exposed area
            self.A_ft = self.A_f + self.A_r

            # Total maximum cross-section blokage
            self.A_FT_airS = (self.tube_OD * self.dx) + (
                (self.dx / self.pitch_fin) * self.t_fin * (self.D_fin - self.tube_OD)
            )

            # Mininum free flow area
            self.A_c = self.A_amb - self.A_FT_airS
            # Fin height
            self.H_f = 0.5 * (self.D_fin - self.tube_OD)
            # Ratio of exposed area to root area
            self.A_rat = (self.A_f + self.A_r) / self.A_tube_o

        # Construct vectors of X position and row number for post-analysis and
        # plotting
        self.x_pf = np.tile(np.linspace(0, self.HX_length, self.n_CPR + 1), self.n_rows)
        self.x_air = np.tile(
            np.linspace(self.dx / 2, self.HX_length - self.dx / 2, self.n_CPR),
            self.n_rows + 1,
        )
        self.x_wall = np.tile(
            np.linspace(self.dx / 2, self.HX_length - self.dx / 2, self.n_CPR),
            self.n_rows,
        )
        self.row_pf = np.array(
            [np.full(self.n_CPR + 1, i) for i in range(self.n_rows)]
        ).flatten()
        self.row_air = np.array(
            [np.full(self.n_CPR, i) for i in range(self.n_rows + 1)]
        ).flatten()
        self.row_wall = np.array(
            [np.full(self.n_CPR, i) for i in range(self.n_rows)]
        ).flatten()


class Fluid:
    def __init__(self):
        self.PF_fluid = []
        self.Amb_fluid = []
        self.T_PF_in = []  # (K)
        self.mdot_PF = []  # (kg/s)
        self.P_PF_in = []  # pressure (Pa)
        self.T_air_in = []  # (K)
        self.vC_in = []  # (m/s)
        self.P_amb_in = []  # pressure (Pa)
        self.P_PF_dp = (
            []
        )  # Process fluid pressure drop for boundary conditin option(Pa)
        self.T_PF_out = (
            []
        )  # Outlet process fluid temperature for boundary condition option

    def micro_init(self, G):
        self.Vdot_C = G.A_amb * self.vC_in  # (m3/s) air volumetric flow rate per cell
        self.mdot_C = self.Vdot_C * CP(
            "D", "P", self.P_amb_in, "T", self.T_air_in, self.Amb_fluid
        )
        self.mdot_amb_max = self.mdot_C / G.A_c
        crit_object = crit_temp_object()
        self.T_pc = crit_object((self.P_PF_in - 101325) / 1e6)


class CT_Geometry:
    def __init__(self):
        self.R_Hd = []  # Aspect ratio H5/d3
        self.R_dD = []  # Diameter ratio d5/d3
        self.R_AA = []  # Area coverage of heat exchangers Aft/A3
        self.R_hD = []  # Ratio of inlet height to diameter H3/d3
        self.R_sd = []  # Ratio of number of tower supports to d3
        self.R_LH = []  # Ratio of height support to tower height L/H3
        self.D_ts = []  # Diameter of tower supports
        self.C_Dts = []  # Drag coefficeint of tower supports
        self.K_ct = []  # Loss coefficient of cooling tower separation
        self.sigma_c = []  # Sigma_c, per Kroger
        self.dA_width = (
            []
        )  # (m2) Frontal width (including series rows) of section of HX analysed in code
        self.solver_type = []  #
        self.d3 = []  # Initial guess for cooling tower diameter
        self.mdot_PF_total = []  # (kg/s) Total PF mass flow rate througout system
        self.T_a1 = []  # (K) Ambient temperature
        self.p_a1 = []  # (Pa) Ambient temperature

    def dim_update(self, d3):
        # All dimensions in meters and labelled per Kroger
        self.d3 = d3
        self.H5 = self.R_Hd * self.d3
        self.d5 = self.R_dD * self.d3
        self.A3 = np.pi * 0.25 * (self.d3**2)
        self.A5 = np.pi * 0.25 * (self.d5**2)
        self.A_fr = self.A3 * self.R_AA
        self.H3 = self.R_hD * self.d3
        self.n_ts = self.R_sd * self.d3
        self.L_ts = self.R_LH * self.d3

    def micro_init(self, HX_L, pitch_transverse, T_air_in, PC_in):
        self.dA_width = pitch_transverse
        self.dA_HX = self.dA_width * HX_L
        self.T_a1 = T_air_in
        self.p_a1 = PC_in  # (Pa) Ambient temperature


# ------------------HEAT TRANSFER & FRICTON CORRELATIONS---------------


def calc_Nu(
    G,
    F,
    Re=0,
    Pr=0,
    P=0,
    Tb=0,
    Tw=0,
    rho_b=0,
    rho_w=0,
    Correlation=0,
    K_c=0,
    mu_b=0,
    x1=0,
    x2=0,
    Dh=0,
):
    """Function to return Nusselt number for internal pipe fluid flow.

    More options can be added as necessary
    """

    if Correlation == 0:
        Nu = 1

    if Correlation in [1, "Yoon-1"]:
        # Yoon et al correlation for Nu based on bulk temperature only
        if Tb > F.T_pc:
            Nu = 0.14 * Re**0.69 * Pr**0.66
        else:
            rho_pc = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, P, F.T_pc], [CPCP.iDmass])
            Nu = 0.013 * Re**1 * Pr**-0.05 * (rho_pc / rho_b) ** 1.6

    if Correlation in [2, "Yoon-2"]:
        # Yoon et al corelation that incorporates wall temperature
        h_b, k_b = GetFluidPropLow(
            EosPF, [CPCP.PT_INPUTS, P, Tb], [CPCP.iHmass, CPCP.iconductivity]
        )
        h_w, Cp_w, mu_w, k_w, Pr_w = GetFluidPropLow(
            EosPF,
            [CPCP.PT_INPUTS, P, Tw],
            e[
                CPCP.iHmass,
                CPCP.iCpmass,
                CPCP.iviscosity,
                CPCP.iconductivity,
                CPCP.iPrandtl,
            ],
        )
        Re_w = Re * (mu_b / mu_w)
        Cp_bar = (h_w - h_b) / (Tw - Tb)
        f = ((0.79 * np.log(Re_w)) - 1.64) ** -2

        Nu_w = ((f / 8) * (Re_w) * Pr_w) / (
            1.07 + (12.7 * ((f / 8) ** 0.5) * ((Pr_w ** (2 / 3)) - 1))
        )
        Nu = (
            1.38
            * Nu_w
            * ((Cp_bar / Cp_w) ** 0.86)
            * ((rho_w / rho_b) ** 0.57)
            * (k_w / k_b)
        )

    if Correlation in [3, "Gnielinski"]:  # Gnielinski
        f = ((0.79 * np.log(Re)) - 1.64) ** -2
        Nu = ((f / 8) * (Re - 1000) * Pr) / (
            1 + (12.7 * ((f / 8) ** 0.5) * ((Pr ** (2 / 3)) - 1))
        )

    if Correlation in [4, "Pitla"]:  # Pitla et al
        Cp_w, mu_w, k_w = GetFluidPropLow(
            EosPF,
            [CPCP.PT_INPUTS, P, Tw],
            [CPCP.iCpmass, CPCP.iviscosity, CPCP.iconductivity],
        )
        k_b = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, P, Tb], [CPCP.iconductivity])
        Pr_w = mu_w * Cp_w / k_w
        Re_w = Re * (mu_b / mu_w) * (rho_w / rho_b)
        f = ((0.79 * np.log(Re)) - 1.64) ** -2
        f_w = ((0.79 * np.log(Re_w)) - 1.64) ** -2
        Nu_b = ((f / 8) * (Re - 1000) * Pr) / (
            1.0 + (12.7 * ((f / 8) ** 0.5) * ((Pr ** (2 / 3)) - 1))
        )
        Nu_w = ((f_w / 8) * (Re_w - 1000) * Pr_w) / (
            1.0 + (12.7 * ((f_w / 8) ** 0.5) * ((Pr_w ** (2 / 3)) - 1))
        )
        Nu = ((Nu_w + Nu_b) / 2) * (k_w / k_b)

    if Correlation == "Mean-Gnielinski":
        Tf = (Tb + Tw) / 2
        Cp_f, mu_f, k_f, rho_f = GetFluidPropLow(
            EosPF,
            [CPCP.PT_INPUTS, P, Tf],
            [CPCP.iCpmass, CPCP.iviscosity, CPCP.iconductivity, CPCP.iDmass],
        )
        k_b = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, P, Tb], [CPCP.iconductivity])
        Pr_f = mu_f * Cp_f / k_f
        Re_f = Re * (mu_b / mu_f) * (rho_f / rho_b)
        f_f = ((0.79 * np.log(Re_f)) - 1.64) ** -2
        Nu_f = ((f_f / 8) * (Re_f - 1000) * Pr_f) / (
            1.0 + (12.7 * ((f_f / 8) ** 0.5) * ((Pr_f ** (2 / 3)) - 1))
        )
        Nu = Nu_f * (k_f / k_b)

    if Correlation == "Mix-Gnielinski":  # Pitla et al
        rho_w, mu_w, k_w = GetFluidPropLow(
            EosPF,
            [CPCP.PT_INPUTS, P, Tw],
            [CPCP.iDmass, CPCP.iviscosity, CPCP.iconductivity],
        )
        k_b = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, P, Tb], [CPCP.iconductivity])
        Re_w = Re * (rho_w / rho_b) * (mu_b / mu_w)
        f_w = ((0.79 * np.log(Re_w)) - 1.64) ** -2
        Nu = ((f_w / 8) * (Re_w - 1000) * Pr) / (
            1.0 + (12.7 * ((f_w / 8) ** 0.5) * ((Pr ** (2 / 3)) - 1))
        )
        Nu = Nu * (k_w / k_b)

    if Correlation in [5, "Wang"]:  # Wang et al (UQ)
        Tf = (Tb + Tw) / 2
        h_b = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, P, Tb], [CPCP.iHmass])
        h_w = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, P, Tw], [CPCP.iHmass])

        if (Tb - Tw) == 0:
            print(h_b, h_w, Tb, Tw)
        Cp_bar = (h_b - h_w) / (Tb - Tw)
        Cp_w = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, P, Tw], [CPCP.iCpmass])
        mu_f = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, P, Tf], [CPCP.iviscosity])
        k_f = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, P, Tf], [CPCP.iconductivity])
        k_b = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, P, Tb], [CPCP.iconductivity])
        Re_f = Re * (mu_b / mu_f)
        f_f = (0.79 * np.log(Re_f) - 1.64) ** -2
        Pr_f = mu_f * Cp_bar / k_f
        Nu_iso = ((f_f / 8) * Re * Pr_f) / (
            1.07 + (12.7 * ((f_f / 8) ** 0.5) * ((Pr_f ** (2 / 3)) - 1))
        )
        Nu = 1.2838 * Nu_iso * ((rho_w / rho_b) ** -0.1458) * (k_f / k_b)

    if Correlation in [6, "Aspen"]:  # From ASPEN, for liquid being cooled
        mu_w = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, P, Tw], [CPCP.iviscosity])
        if F.PF_fluid == "CO2":
            Fp = 1
        else:
            Fp = (mu_b / mu_w) ** 0.25
        Nu_t = 0.02246 * (Re**0.795) * (Pr ** (0.495 - (0.0225 * np.log(Pr)))) * Fp
        if Re > 30000:
            Nu = Nu_t
        else:
            Gz = (np.pi * Dh / (4 * x2)) * Re * Pr
            Nu_e1 = (365 + 13.03 * (Gz ** (4 / 3))) ** (1 / 4)
            Nu_e2 = (365 + 13.03 * (((x2 / x1) * Gz) ** (4 / 3))) ** (1 / 4)
            Nu_e3 = ((x2 * Nu_e1) - (x1 * Nu_e2)) / (x2 - x1)
            Nu_l = Nu_e3
            Nu_t1 = (
                1
                / (
                    (1 / Nu_t**2)
                    + (1 / (Nu_l * ((np.exp((min(10000, Re) - 2200) / 730)) ** 2)))
                )
            ) ** 0.5
            Nu = max(Nu_t1, Nu_l)

    if Correlation == "Krasn.-1969":
        if not hasattr(F, "Kras_interp"):
            ps = [7.845, 8, 8.5, 9, 10, 12]
            ns = [0.3, 0.38, 0.54, 0.61, 0.68, 0.8]
            Bs = [0.68, 0.75, 0.85, 0.91, 0.97, 1]
            # Changed from 'k' to avoid confict with conductivity
            gs = [0.21, 0.18, 0.104, 0.066, 0.040, 0]
            F.Kras_interp = interp1d(ps, [ns, Bs, gs], fill_value="extrapolate")

        h_b, k_b = GetFluidPropLow(
            EosPF, [CPCP.PT_INPUTS, P, Tb], [CPCP.iHmass, CPCP.iconductivity]
        )
        h_w, Cp_w, k_w, mu_w, Pr_w = GetFluidPropLow(
            EosPF,
            [CPCP.PT_INPUTS, P, Tw],
            [
                CPCP.iHmass,
                CPCP.iCpmass,
                CPCP.iconductivity,
                CPCP.iviscosity,
                CPCP.iPrandtl,
            ],
        )
        Re_w = Re * (mu_b / mu_w)
        f_w = (0.79 * np.log(Re_w) - 1.64) ** -2
        Nu_o = ((f_w / 8) * Re_w * Pr_w) / (
            1.07 + (12.7 * ((f_w / 8) ** 0.5) * ((Pr_w ** (2 / 3)) - 1))
        )
        Cp_bar = (h_w - h_b) / (Tw - Tb)
        n, B, g = F.Kras_interp(P / 1e6)
        m = B * ((Cp_bar / Cp_w) ** g)
        Nu = Nu_o * ((rho_w / rho_b) ** n) * ((Cp_bar / Cp_w) ** m) * (k_w / k_b)

    if Correlation == "Liao":
        h_b, k_b, mu_b = GetFluidPropLow(
            EosPF,
            [CPCP.PT_INPUTS, P, Tb],
            [CPCP.iHmass, CPCP.iconductivity, CPCP.iviscosity],
        )
        h_w, cp_w, k_w, mu_w, Pr_w = GetFluidPropLow(
            EosPF,
            [CPCP.PT_INPUTS, P, Tw],
            [
                CPCP.iHmass,
                CPCP.iCpmass,
                CPCP.iconductivity,
                CPCP.iviscosity,
                CPCP.iPrandtl,
            ],
        )
        Re_w = Re * (mu_b / mu_w)
        cp_bar = (h_w - h_b) / (Tw - Tb)
        if not rho_w > rho_b:
            rho_w = rho_b + 0.01
        Gr = rho_b * (rho_w - rho_b) * 9.81 * G.ID**3 / (mu_b**2)
        Nu = (
            0.128
            * Re_w**0.8
            * Pr_w**0.3
            * (Gr / (Re**2)) ** 0.205
            * (rho_b / rho_w) ** 0.437
            * (cp_bar / cp_w) ** 0.411
            * (k_w / k_b)
        )

    if Correlation == "Zhang":
        Tf = (Tb + Tw) / 2
        h_b, k_b, mu_b = GetFluidPropLow(
            EosPF,
            [CPCP.PT_INPUTS, P, Tb],
            [CPCP.iHmass, CPCP.iconductivity, CPCP.iviscosity],
        )
        h_f, cp_f, k_f, mu_f, Pr_f, rho_f = GetFluidPropLow(
            EosPF,
            [CPCP.PT_INPUTS, P, Tf],
            [
                CPCP.iHmass,
                CPCP.iCpmass,
                CPCP.iconductivity,
                CPCP.iviscosity,
                CPCP.iPrandtl,
                CPCP.iDmass,
            ],
        )
        h_w = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, P, Tw], [CPCP.iHmass])
        Re_f = Re * (mu_b / mu_f)
        cp_bar_f = (h_w - h_f) / (Tw - Tf)
        if not rho_w > rho_f:
            rho_w = rho_f + 0.1
        Gr_f = rho_f * (rho_w - rho_f) * 9.81 * G.ID**3 / (mu_f**2)
        x = (x1 + x2) / 2
        Nu = (
            0.138
            * Re_f**0.68
            * Pr_f**0.07
            * (rho_f / rho_w) ** -0.74
            * (cp_bar_f / cp_f) ** -0.31
            * (Gr_f / (Re**2)) ** 0.08
            * (1 + ((G.ID / x) ** (2 / 3)))
            * (k_f / k_b)
        )

    if Correlation == "Dang":
        Tf = (Tb + Tw) / 2
        cp_f, mu_f, k_f, rho_f = GetFluidPropLow(
            EosPF,
            [CPCP.PT_INPUTS, P, Tf],
            [CPCP.iCpmass, CPCP.iviscosity, CPCP.iconductivity, CPCP.iDmass],
        )
        h_w, Cp_w, k_w, mu_w, Pr_w = GetFluidPropLow(
            EosPF,
            [CPCP.PT_INPUTS, P, Tw],
            [
                CPCP.iHmass,
                CPCP.iCpmass,
                CPCP.iconductivity,
                CPCP.iviscosity,
                CPCP.iPrandtl,
            ],
        )
        h_b, k_b, cp_b = GetFluidPropLow(
            EosPF,
            [CPCP.PT_INPUTS, P, Tb],
            [CPCP.iHmass, CPCP.iconductivity, CPCP.iCpmass],
        )

        Re_f = Re * (mu_b / mu_f)
        Re_w = Re * (mu_b / mu_w)
        f_f = ((0.79 * np.log(Re_f)) - 1.64) ** -2
        cp_bar = (h_w - h_b) / (Tw - Tb)
        rat1 = mu_b / k_b
        rat2 = mu_f / k_f
        if cp_b > cp_bar:
            Pr = Pr
        elif rat1 > rat2:
            Pr = cp_bar * mu_b / k_b
        else:
            Pr = cp_bar * mu_f / k_f
        Nu = (
            ((f_f / 8) * (Re - 1000) * Pr)
            / (1.07 + (12.7 * ((f_f / 8) ** 0.5) * ((Pr ** (2 / 3)) - 1)))
            * (k_f / k_b)
        )

    if Correlation == "Liu":
        h_b, k_b, cp_b = GetFluidPropLow(
            EosPF,
            [CPCP.PT_INPUTS, P, Tb],
            [CPCP.iHmass, CPCP.iconductivity, CPCP.iCpmass],
        )
        h_w, cp_w, k_w, mu_w, Pr_w = GetFluidPropLow(
            EosPF,
            [CPCP.PT_INPUTS, P, Tw],
            [
                CPCP.iHmass,
                CPCP.iCpmass,
                CPCP.iconductivity,
                CPCP.iviscosity,
                CPCP.iPrandtl,
            ],
        )
        Re_w = Re * (mu_b / mu_w)
        Nu = (
            0.01
            * Re_w**0.9
            * Pr_w**0.5
            * (rho_w / rho_b) ** 0.906
            * (cp_b / cp_w) ** 0.585
            * (k_w / k_b)
        )

    if Correlation == "New_correlation":
        t3 = G.ID**3 * 9.81 * rho_b**2 / mu_b**2
        if rho_w > rho_b:
            t4 = (rho_w - rho_b) / rho_b
        else:
            t4 = 1
        Nu = 2.6859e-9 * Re**0.6544 * Pr**0.2827 * t3**0.77429 * t4**-0.2839

    return Nu


def calc_alpha_amb(
    G, F, I, Re=0, Pr=0, Tb=0, rho_b=0, Correlation=0, K_c=0, row_correction=0
):
    """Function to return air-side finned tube heat transfer coefficient"""

    mu = I.mu_f(Tb)

    if Correlation == 1:  # Briggs and Young finned tube correlation
        k = I.k_f(Tb)
        Re = (F.mdot_C / G.A_c) * G.tube_OD / mu
        Nu = (
            0.134
            * (Pr**0.33)
            * (Re**0.681)
            * ((2 * (G.pitch_fin - G.t_fin) / (G.D_fin - G.tube_OD)) ** 0.2)
            * (((G.pitch_fin - G.t_fin) / G.t_fin) ** 0.1134)
        )
        alpha = Nu * k / G.tube_OD

    if Correlation == 2:  # Gaugouli finned tube correlation
        k = I.k_f(Tb)
        Re = (F.mdot_C / G.A_c) * G.tube_OD / mu
        Nu = 0.38 * (Re**0.6) * (Pr ** (1 / 3)) * ((G.A_rat) ** -0.15)
        alpha = Nu * k / G.tube_OD

    if Correlation == 3:  # From ASPEN HTFS3-AC
        cp = I.cp_f(Tb)
        U = F.mdot_C / (rho_b * G.A_amb)
        U_f = K_c * U
        Re_f = rho_b * U_f * G.tube_OD / mu
        u_max = F.mdot_C / (rho_b * G.A_c)
        Re_max = u_max * rho_b * G.tube_OD / mu
        j = 1.207 * (Re_f**0.04) * (Re_max**-0.5094) * (G.A_rat**-0.312)
        alpha = j * cp * F.mdot_amb_max * (Pr ** (-2 / 3))

    if Correlation == 4:  # ASPEN HTFS3
        cp = I.cp_f(Tb)
        U = F.mdot_C / (rho_b * G.A_amb)
        U_f = K_c * U
        Re_f = rho_b * U_f * G.tube_OD / mu
        u_max = F.mdot_C / (rho_b * G.A_c)
        Re_max = u_max * rho_b * G.tube_OD / mu
        j = 0.205 * (Re_max**-0.368) * (Re_f**0.04) * (G.A_rat**-0.15)
        alpha = j * cp * F.mdot_amb_max * (Pr ** (-2 / 3))

    if row_correction:
        # Correction for turbulence difference from multiple rows. From Kroger.
        u_max = F.mdot_C / (rho_b * G.A_c)
        alpha = alpha * ((1 + (u_max / (G.n_rows**2))) ** -0.14)

    return alpha


def calc_f(
    Re, P, Tm, Tp, TW, mdot, A, Dh, fluid, Correlation, rho_b, rho_w, mu_b, epsilon=0
):
    """Function to return friction factor for internal pipe flow"""
    if Re < 2500:
        Nu = 64 / Re

    if Correlation == 0:
        f = 0

    if Correlation == 1:
        f = (0.79 * np.log(Re) - 1.64) ** -2  # Petukov

    if Correlation == 2:
        # Haaland formula for friction factor.
        e = 5e-5
        f = 1 / (-1.8 * np.log10((6.9 / Re) + ((e / (3.7 * Dh)) ** 1.11))) ** 2

    if Correlation == 3:  # Blasius fricion factor
        if Re < 2e4:
            f = 0.316 * Re ** (-1 / 4)
        else:
            f = 0.184 * Re ** (-1 / 5)
    return f


def calc_air_dp(G, rho, mu):
    """Function to return pressure differential per row for air through
    finned tube banks, from ASPEN HTRI
    """

    K_tube = (
        4.75
        * G.n_rows
        * G.pitch_longitudal
        * ((mu / rho) ** 0.3)
        * (((G.pitch_transverse / G.tube_OD) - 1) ** -1.86)
        * (G.tube_OD**-1.3)
    )
    phi = (
        np.pi
        * ((G.D_fin**2) - (G.tube_OD**2))
        * (1 / G.pitch_fin)
        * G.n_rows
        / (2 * G.D_fin)
    )
    B = G.A_FT_airS / G.dx
    tau = G.D_fin / (G.D_fin - B)
    K_fins = 0.0265 * phi * (tau**1.7)
    K_ft = K_tube + K_fins
    N_G = G.n_rows - 1
    G_D = (
        ((G.pitch_longitudal**2) + ((0.5 * G.pitch_transverse) ** 2)) ** 0.5
    ) - G.D_fin
    G_T = G.pitch_transverse - G.D_fin
    G_A = 0.5 * (G.D_fin - G.tube_OD)
    GR_eff = (G_D + G_A) / G_T
    theta = np.arctan(0.5 * G.pitch_transverse / G.pitch_longitudal)
    K_gap = N_G * theta * GR_eff
    K_B = K_ft / (
        (
            (G.D_fin / G.pitch_transverse)
            + (((K_ft / K_gap) ** (1 / 1.7)) * (1 - (G.D_fin / G.pitch_transverse)))
        )
        ** 1.7
    )
    return K_B, K_ft


# ------------------UTILITY FUNCTIONS----------------------------------


def crit_temp_object():
    """Returns an interpolation object used to find critical point for a given
    pressure.
    """

    def Cp(T, P):
        # return CP('C','T',T,'P',P,'CO2')
        HEOS = CPCP.AbstractState("HEOS", "CO2")
        HEOS.update(CPCP.PT_INPUTS, P, T)
        C = HEOS.cpmass()
        return C

    P_array = [7.4, 7.5, 7.6, 7.8, 8, 8.5, 9, 9.5, 10]
    T_array = []
    for i in range(len(P_array)):
        Tcrit = optimize.fmin(
            lambda T: -1 * Cp(T, (P_array[i] * 1e6) + 101325),
            273.15 + 40,
            xtol=1e-8,
            disp=False,
        )
        T_array.append(Tcrit[0])
    crit_inter = interp1d(P_array, T_array, fill_value="extrapolate")
    return crit_inter


class inter_fs:
    """This function creates an interpolation matrix of fluid properties for
    the cold (air) side
    """

    def __init__(self, T_l, T_h, P, fluid):
        n = 100
        T = np.linspace(T_l, T_h, n)

        V = CP("V", "P", P, "T", T[0:n], fluid)
        D = CP("D", "P", P, "T", T[0:n], fluid)
        H = CP("H", "P", P, "T", T[0:n], fluid)
        K = CP("CONDUCTIVITY", "P", P, "T", T[0:n], fluid)
        C = CP("C", "P", P, "T", T[0:n], fluid)
        Pr = CP("PRANDTL", "P", P, "T", T[0:n], fluid)

        self.mu_f = interp1d(T, V, bounds_error=False, fill_value="extrapolate")
        self.rho_f = interp1d(T, D, bounds_error=False, fill_value="extrapolate")
        self.h_f = interp1d(T, H, bounds_error=False, fill_value="extrapolate")
        self.k_f = interp1d(T, K, bounds_error=False, fill_value="extrapolate")
        self.cp_f = interp1d(T, C, bounds_error=False, fill_value="extrapolate")
        self.Pr_f = interp1d(T, Pr, bounds_error=False, fill_value="extrapolate")


def get_X0_iter(G, F, M, I, CT):
    # Function to get initial values for iterative solver

    if M.cooler_type == 1:
        X0 = np.zeros((4 * G.n_cells) + (G.n_rows) + 1 + CT.solver_type)
        mdot_init = CT.mdot_PF_total / ((CT.A_fr / CT.dA_HX))
    else:
        X0 = np.zeros((4 * G.n_cells) + (G.n_rows))
        mdot_init = F.mdot_PF

    dT = F.T_PF_in - F.T_PF_out

    for i in range(G.n_passes):
        for j in range(G.n_CP):
            row = j + (i * G.n_CP)
            X0[row * G.n_CPR : (row + 1) * G.n_CPR] = np.linspace(
                F.T_PF_in - ((i / G.n_passes) * dT),
                F.T_PF_in - (((i + 1) / G.n_passes) * dT),
                G.n_CPR,
            )

    for i in range(G.n_rows):
        X0[G.n_cells + (i * G.n_CPR) : G.n_cells + ((i + 1) * G.n_CPR)] = F.T_air_in + (
            (F.T_air_outlet_guess - F.T_air_in) * ((G.n_rows - i) / G.n_rows)
        )
        X0[2 * G.n_cells : 3 * G.n_cells] = F.T_PF_in - (
            (F.T_PF_in - F.T_air_outlet_guess) * (((i + 0.45) / G.n_rows))
        )
        X0[3 * G.n_cells : 4 * G.n_cells] = F.T_PF_in - (
            (F.T_PF_in - F.T_air_outlet_guess) * (((i + 1) / G.n_rows))
        )

    # Mass flow rate
    X0[4 * G.n_cells : (4 * G.n_cells) + G.n_rows] = 1e4 * mdot_init / G.n_CP

    # Error variable for natural draft equation (velocity)
    if M.cooler_type == 1:
        X0[(4 * G.n_cells) + G.n_rows] = F.vC_in * 1000

        # Error variable for tower diameter for matching PF mass flow rate
        if CT.solver_type == 1:
            X0[(4 * G.n_cells) + G.n_rows + 1] = CT.d3 * 10

    return X0


def open_X(X, M, F, G, CT):
    """
    function to unpack the variable vector X into the 6 vectors
    TH, TWH, TWC, T_air, PH, PC
    """

    T_PF_in = F.T_PF_in
    T_air_in = F.T_air_in
    n_cells = G.n_cells
    n_rows = G.n_rows
    n_CPR = G.n_CPR
    n_CP = G.n_CP
    n_passes = G.n_passes
    P_PF_in = F.P_PF_in
    PF_fluid = F.PF_fluid

    TH = np.zeros(n_cells + n_rows)
    T_air = np.zeros(n_cells + n_CPR)
    TW_H = np.zeros(n_cells)
    TW_C = np.zeros(n_cells)
    mdot_vec = np.zeros(n_rows)
    v = []
    d3 = []

    mdot_vec[0:n_rows] = X[4 * n_cells : (4 * n_cells) + n_rows] / 1e4

    for i1 in range(n_CP):
        TH[i1 * (n_CPR + 1)] = T_PF_in
    T_head = []
    dir_vec = []
    for i2 in range(n_passes):
        T_end = []
        for i3 in range(n_CP):
            row_no = (i2 * n_CP) + i3
            start_n = row_no * (n_CPR + 1)
            end_n = row_no * (n_CPR + 1) + n_CPR + 1
            if i2 % 2 == 0:
                dir_vec.append(1)
                TH[start_n + 1 : end_n] = X[row_no * (n_CPR) : ((row_no + 1) * (n_CPR))]
                T_end.append(TH[end_n - 1])
            if i2 % 2 == 1:
                dir_vec.append(-1)
                TH[start_n : end_n - 1] = X[row_no * (n_CPR) : ((row_no + 1) * (n_CPR))]
                T_end.append(TH[start_n])

    T_air[0:n_cells] = X[n_cells : (2 * n_cells)]
    T_air[n_cells : n_cells + n_CPR] = T_air_in
    TW_H[0:n_cells] = X[2 * n_cells : 3 * n_cells]
    TW_C[0:n_cells] = X[3 * n_cells : 4 * n_cells]

    if M.cooler_type == 1:
        v = X[(4 * n_cells) + n_rows] / 1000
        if CT.solver_type == 1:
            d3 = X[(4 * n_cells) + n_rows + 1] / 10
    else:
        v = F.vC_in
        d3 = 0

    return TH, T_air, TW_H, TW_C, dir_vec, mdot_vec, v, d3


# ------------------SOLVER-------------------------------------------


class Callback:
    """Callback function to print status of nonlinear solver"""

    def __init__(self, G, M):
        self.G = G
        self.M = M
        self.i = 0

    def update(self, x, r):
        self.i += 1
        e_max = np.max(np.absolute(r))
        e_av = np.average(np.absolute(r))
        mdot_vec = x[4 * self.G.n_cells : (4 * self.G.n_cells) + self.G.n_rows] / 10000

        print("Iteration no", self.i)
        print("Maximum residual: ", e_max, "Average residual: ", e_av)
        print("Tube mass flow rates:", mdot_vec)
        print("---------------------------------------------")


def solve(hx_inputs, mod=None, X0=None, fig_switch=0, verbosity=0):
    M = Model()
    F = Fluid()
    G = Geometry()
    CT = CT_Geometry()

    hx_inputs(G, F, M, CT)

    if (
        mod
    ):  # A function that operates on input classes - for iterative optimisation only
        # TODO: Can I find a more elegant way of doing this?
        mod(F)
        mod(M)
        mod(G)
        if M.cooler_type == 1:
            mod(CT)

    # Create fluid property objects
    global EosPF
    EosPF = CPCP.AbstractState("HEOS", F.PF_fluid)

    # Initialise objects that contain geometry and fluid property details.
    # TODO: This is really ugly. Can I improve it?
    G.micro_init()
    F.micro_init(G)

    if M.cooler_type == 1:
        CT.micro_init(G.HX_length, G.pitch_transverse, F.T_air_in, F.P_amb_in)
        CT.dim_update(CT.d3)

    # Construct iterpolation object for fluid properties
    I = inter_fs(F.T_air_in, F.T_PF_in, F.P_amb_in, F.Amb_fluid)

    # Creates initial solution values from inputs or interpolation function
    if X0 is None:
        X0 = get_X0_iter(G, F, M, I, CT)

    if verbosity > 0:
        print("Number of cells:", G.n_cells)

    # TODO: Can I remove this? It's ugly
    #  n_cells = G.n_cells
    #  n_rows = G.n_rows

    callback = Callback(G, M)
    sol = sci.optimize.newton_krylov(
        lambda T: equations(T, G, F, M, I, 0, CT),
        X0,
        method="lgmres",
        f_tol=1e-6,
        callback=callback.update,
    )

    # TODO: Process results much more elegantly
    (
        error,
        TH_final,
        T_air_final,
        TW_H_final,
        TW_C_final,
        Q1,
        Q2,
        Q3,
        Q4,
        Q5,
        Q6,
        Q7,
        Q8,
        Q9,
        alpha_i_final,
        PH,
        dp_amb,
        mdot_vec,
        P_headers,
        v,
        alpha_o_final,
        dT,
        d3,
        Bu_c,
        Bu_j,
        Bu_p,
        UA,
    ) = equations(sol, G, F, M, I, 1, CT)

    # Compute total properties at inlet/outlet
    P_out = P_headers[-1]
    E_out = []
    if G.n_passes % 2 == 0:
        for i in range(G.n_CP):
            T_row_out = TH_final[len(TH_final) - ((i + 1) * (G.n_CPR + 1))]
            E_out.append(
                GetFluidPropLow(
                    EosPF, [CPCP.PT_INPUTS, P_out, T_row_out], [CPCP.iHmass]
                )
                * mdot_vec[-1 - i]
            )
    else:
        for i in range(G.n_CP):
            T_row_out = TH_final[-((i) * (G.n_CPR + 1)) - 1]
            E_out.append(
                GetFluidPropLow(
                    EosPF, [CPCP.PT_INPUTS, P_out, T_row_out], [CPCP.iHmass]
                )
                * mdot_vec[-1 - i]
            )
    h_ave = sum(E_out) / sum(mdot_vec[-G.n_CP : G.n_rows])
    E_out_total = sum(E_out)
    T_out = CP("T", "H", h_ave, "P", P_out, F.PF_fluid)
    E_in_total = sum(mdot_vec[0 : G.n_CP]) * GetFluidPropLow(
        EosPF, [CPCP.PT_INPUTS, F.P_PF_in, F.T_PF_in], [CPCP.iHmass]
    )

    PF_dp = F.P_PF_in - P_out

    # Determine the average pressure loss of air over the heat exchanger
    dp_amb_total = sum(dp_amb)
    P_amb_out = F.P_amb_in - dp_amb_total

    # Determine the energy difference of PF between inlete and outlet
    mdot_total = np.sum(mdot_vec[0:i])

    # Check solution via control volume energy balance
    dE_PF = E_in_total - E_out_total

    # Determine the total energy gain of air over the heat exchanger by summing each cell
    i = 0
    q_amb_out = 0
    for i in range(G.n_CPR):
        qi = CP("H", "T", T_air_final[i], "P", P_amb_out, F.Amb_fluid)
        q_amb_out = q_amb_out + qi
    Q_amb = (
        q_amb_out - (G.n_CPR * CP("H", "T", F.T_air_in, "P", F.P_amb_in, F.Amb_fluid))
    ) * F.mdot_C
    T_air_out_ave = mean(T_air_final[0 : G.n_CPR])

    # Determine and print the discrepance of energy balance between PF and air
    deltaQ = Q_amb - dE_PF

    if verbosity > 0:
        print("Process fluid outlet temperature is: ", T_out)
        print("Mass flow rate vector is:", mdot_vec)
        print(
            "Outlet process fluid pressure: ",
            P_out / 1000,
            "kPa, Differential: ",
            (PF_dp) / 1000,
            "kPa",
        )
        print(
            "Outlet air pressure: ",
            P_amb_out / 1000,
            "kPa, Differential: ",
            dp_amb_total,
            "Pa",
        )
        print("Air face velocity", v)
        print("Air maximum outlet temperature is: ", T_air_final[0])
        print("Air average outlet temperature is: ", T_air_out_ave)
        print("Heat rejected per HX column: ", (dE_PF / 1000), " kW")
        print(
            "Solution discrepance: ",
            deltaQ / 1000,
            " kW, or: ",
            (deltaQ / dE_PF) * 100,
            "%",
        )

    # Display warning if energy balance between the two fluids is too large
    if abs(deltaQ / dE_PF) > 0.001:
        print("Solution error. Suggest smaller element size")

    result_dict = {
        "X": sol,
        "T_air_out_ave": T_air_out_ave,
        "dp_air": dp_amb_total,
        "Q_total": dE_PF,
        "dp_pf": PF_dp,
        "T_pf": TH_final,
        "alpha_pf": alpha_i_final,
        "T_w_h": TW_H_final,
        "T_w_c": TW_C_final,
        "q_cells": Q3,
        "p_pf": PH,
        "T_air": T_air_final,
        "alpha_air": alpha_o_final,
        "dT": dT,
        "mdot_vec": mdot_vec,
        "G": G,
        "p_pf": PH,
        "T_pf_out": T_out,
    }

    return result_dict


def equations(X, G, F, M, I, flag, CT):
    """Equations to solve energy balance for each cell in the model, and return
    the energy balance error"""

    # Initialise error vector
    if M.cooler_type == 1:
        error = np.zeros((4 * G.n_cells) + G.n_rows + 1 + CT.solver_type)
    else:
        error = np.zeros((4 * G.n_cells) + G.n_rows)

    if flag == 1:
        # For when the model and solved correactly and we want to store the
        # resutls
        Q1 = []
        Q2 = []
        Q3 = []
        Q4 = []
        Q5 = []
        Q6 = []
        Q7 = []
        Q8 = []
        Q9 = []
        Alpha_i = []
        Alpha_o = []
        dT = []
        Bu_c_array = []
        Bu_j_array = []
        Bu_p_array = []
        UA = []

    PH = np.zeros(G.n_cells + G.n_rows)
    P_headers = [F.P_PF_in]
    T_headers = [F.T_PF_in]
    P_row_ends = []
    T_row_ends = []

    TH, T_air, TW_H, TW_C, DV, mdot_vec, v, d3 = open_X(X, M, F, G, CT)

    F.Vdot_C = G.A_amb * v  # (m3/s) air volumetric flow rate per cell
    F.mdot_C = F.Vdot_C * CP("D", "P", F.P_amb_in, "T", F.T_air_in, F.Amb_fluid)
    F.mdot_amb_max = F.mdot_C / G.A_c

    # Calculate the pressure drop and loss coefficients across each row based
    # on mean air properties for that row. From ASPEN HTSF3

    K_c = []
    dp_amb = []

    inlet_cells = []
    outlet_cells = []
    header_inlet_cells = []
    header_outlet_cells = []

    for x in range(G.n_CP):
        inlet_cells.append(x * G.n_CPR)
        outlet_cells.append(G.n_cells - (x * G.n_CPR) - 1)
        for y in range(G.n_passes - 1):
            header_inlet_cells.append(
                (G.n_CPR * x) + (y * G.n_CPR * G.n_CP) - 1 + G.n_CPR
            )
            header_outlet_cells.append(((y + 1) * G.n_CPR * G.n_CP) + (x * G.n_CPR))

    for k in range(G.n_rows):
        rho_k = 0.25 * (
            I.rho_f(T_air[k * G.n_CPR])
            + I.rho_f(T_air[((k + 1) * G.n_CPR) - 1])
            + I.rho_f(T_air[(k + 1) * G.n_CPR])
            + I.rho_f(T_air[((k + 2) * G.n_CPR) - 1])
        )

        mu_k = 0.25 * (
            I.mu_f(T_air[k * G.n_CPR])
            + I.mu_f(T_air[((k + 1) * G.n_CPR) - 1])
            + I.mu_f(T_air[(k + 1) * G.n_CPR])
            + I.mu_f(T_air[((k + 2) * G.n_CPR) - 1])
        )

        U_k = F.mdot_C / (rho_k * G.A_amb)
        K_B, K_ft = calc_air_dp(G, rho_k, mu_k)
        K_c.append((K_B / K_ft) ** (1 / 1.7))
        dp_amb.append(1.081 * K_B * rho_k * (U_k**1.65) / G.n_rows)
        dp_amb_total = sum(dp_amb)

    # Pressure drop calculations
    for a in range(G.n_passes):
        for b in range(G.n_CP):
            row = b + (a * G.n_CP)

            if DV[row] == 1:
                # print("Row starting node",row * (G.n_CPR+1))
                PH[row * (G.n_CPR + 1)] = P_headers[a]
                TH[row * (G.n_CPR + 1)] = T_headers[a]

                for c in range(1, G.n_CPR + 1):
                    node = (row * (G.n_CPR + 1)) + c
                    cell = (row * (G.n_CPR)) + c - 1
                    # print("node",node)
                    # print("cell",cell)
                    Tb_PF = (TH[node] + TH[node - 1]) / 2
                    rho_PF_b = GetFluidPropLow(
                        EosPF, [CPCP.PT_INPUTS, PH[node - 1], Tb_PF], [CPCP.iDmass]
                    )
                    rho_PF_w = GetFluidPropLow(
                        EosPF, [CPCP.PT_INPUTS, PH[node - 1], TW_H[cell]], [CPCP.iDmass]
                    )
                    mu_PF = GetFluidPropLow(
                        EosPF, [CPCP.PT_INPUTS, PH[node - 1], Tb_PF], [CPCP.iviscosity]
                    )
                    U_PF = abs(mdot_vec[row] / (rho_PF_b * G.A_CS))
                    Re_PF = rho_PF_b * U_PF * G.ID / mu_PF
                    f_PF = calc_f(
                        Re_PF,
                        PH[node - 1],
                        TH[node],
                        TH[node - 1],
                        TW_H[cell],
                        mdot_vec[row],
                        G.A_CS,
                        G.ID,
                        F.PF_fluid,
                        M.f_CorrelationH,
                        rho_PF_b,
                        rho_PF_w,
                        mu_PF,
                    ) * np.sign(mdot_vec[row])
                    # calculate friction head loss (Darcy Weisbach formula)
                    dp = rho_PF_b * f_PF * (G.dx / G.ID) * (U_PF**2) / 2
                    PH[node] = PH[node - 1] - dp
                if M.consider_bends == 1:
                    dp_PF_bend = rho_PF_b * M.bend_loss_coefficient * (U_PF**2) / 2
                    P_row_ends.append(PH[node] - dp_PF_bend)
                else:
                    P_row_ends.append(PH[node])
                T_row_ends.append(TH[node])

            else:
                PH[((row + 1) * (G.n_CPR + 1)) - 1] = P_headers[a]
                TH[((row + 1) * (G.n_CPR + 1)) - 1] = T_headers[a]

                for c in range(G.n_CPR - 1, -1, -1):
                    node = (row * (G.n_CPR + 1)) + c
                    cell = (row * (G.n_CPR)) + c
                    Tb_PF = (TH[node] + TH[node + 1]) / 2
                    rho_PF_b = GetFluidPropLow(
                        EosPF, [CPCP.PT_INPUTS, PH[node + 1], Tb_PF], [CPCP.iDmass]
                    )
                    rho_PF_w = GetFluidPropLow(
                        EosPF, [CPCP.PT_INPUTS, PH[node + 1], TW_H[cell]], [CPCP.iDmass]
                    )
                    mu_PF = GetFluidPropLow(
                        EosPF, [CPCP.PT_INPUTS, PH[node + 1], Tb_PF], [CPCP.iviscosity]
                    )
                    U_PF = abs(mdot_vec[row] / (rho_PF_b * G.A_CS))
                    Re_PF = rho_PF_b * U_PF * G.ID / mu_PF
                    f_PF = calc_f(
                        Re_PF,
                        PH[node + 1],
                        TH[node],
                        TH[node + 1],
                        TW_H[cell],
                        mdot_vec[row],
                        G.A_CS,
                        G.ID,
                        F.PF_fluid,
                        M.f_CorrelationH,
                        rho_PF_b,
                        rho_PF_w,
                        mu_PF,
                    ) * (np.sign(mdot_vec[row]))
                    # calculate friction head loss (Darcy Weisbach formula)
                    dp = rho_PF_b * f_PF * (G.dx / G.ID) * (U_PF**2) / 2
                    PH[node] = PH[node + 1] - dp
                if M.consider_bends == 1:
                    dp_PF_bend = rho_PF_b * M.bend_loss_coefficient * (U_PF**2) / 2
                    P_row_ends.append(PH[node] - dp_PF_bend)
                else:
                    P_row_ends.append(PH[node])
                T_row_ends.append(TH[node])
                # print("Row end note",node)

        P_headers.append(np.average(P_row_ends[a * G.n_CP : (a + 1) * G.n_CP]))
        # Calculation of header temperature from enthalpy balance

        if a < (G.n_passes - 1):
            h_c = np.zeros(G.n_CP)

            for x in range(G.n_CP):
                h_c[x] = GetFluidPropLow(
                    EosPF,
                    [CPCP.PT_INPUTS, P_headers[a + 1], T_row_ends[x + (a * G.n_CP)]],
                    [CPCP.iHmass],
                )
            h_ave = np.dot(h_c, mdot_vec[(a * G.n_CP) : ((a + 1) * G.n_CP)]) / sum(
                mdot_vec[(a * G.n_CP) : ((a + 1) * G.n_CP)]
            )
            T_headers.append(CP("T", "H", h_ave, "P", P_headers[a + 1], F.PF_fluid))

    # ------------------------------------------------------------------------------
    # ERROR ASSIGNMENT
    # -------------------------------------------------------------------------------

    # Header pressure for pressure drop boundary condition
    # [(n_passes-1)*(n_CP-1) equations]
    if M.solver_type == 0:
        for d in range(G.n_passes - 1):
            for e in range(G.n_CP - 1):
                delta_header_P = (
                    P_row_ends[(d * G.n_CP) + e] - P_row_ends[(d * G.n_CP) + e + 1]
                )
                # Error for similar pressure between parallel streams at headers
                error[
                    4 * G.n_cells + (G.n_passes - 1) + ((d * (G.n_CP - 1)) + e)
                ] = delta_header_P

    # Header pressure for mdot or outlet temperature boundary condition
    # [(n_passes)*(n_CP-1) equations]
    if M.solver_type == 1 or 2:
        for d in range(G.n_passes):
            if G.n_CP > 1:
                delta_header_P_1 = (
                    P_row_ends[(d * G.n_CP)] - P_row_ends[(d * G.n_CP) + 1]
                )

            for e in range(G.n_CP - 1):
                delta_P = P_headers[d] - P_row_ends[(d * G.n_CP) + e]
                delta_P1 = P_headers[d] - P_row_ends[(d * G.n_CP) + e + 1]
                pc_delta_header_P = (delta_P - delta_P1) / delta_P1
                error[4 * G.n_cells + (G.n_passes - 1) + ((d * (G.n_CP - 1)) + e)] = (
                    np.sign(pc_delta_header_P) * (abs(pc_delta_header_P)) * 1e3
                )

    ## Boundary condition error - switch to change boundary condition
    if M.solver_type == 0:  # # Pressure differential boundary condition
        for f in range(G.n_CP):
            # Error for outlet pressure matching required dP
            error[
                (4 * G.n_cells)
                + (G.n_passes - 1)
                + ((G.n_passes - 1) * (G.n_CP - 1))
                + f
            ] = (F.P_PF_in - P_row_ends[-1 - f]) - F.P_PF_dp

    if M.solver_type == 1:  # Total mass flow rate boundary condition
        if M.cooler_type == 1:
            error[
                (4 * G.n_cells) + (G.n_passes - 1) + ((G.n_passes) * (G.n_CP - 1))
            ] = (
                (CT.mdot_PF_total / (CT.A_fr / CT.dA_HX)) - sum(mdot_vec[0 : G.n_CP])
            ) * 1e3
        else:
            error[
                (4 * G.n_cells) + (G.n_passes - 1) + ((G.n_passes) * (G.n_CP - 1))
            ] = (F.mdot_PF - sum(mdot_vec[0 : G.n_CP])) * 1e4

    if M.solver_type == 2:  # Temperature outlet boundary condition
        P_out = P_headers[-1]
        E_out = []
        if G.n_passes % 2 == 0:
            for i in range(G.n_CP):
                T_row_out = TH[len(TH) - ((i + 1) * (G.n_CPR + 1))]
                E_out.append(
                    GetFluidPropLow(
                        EosPF, [CPCP.PT_INPUTS, P_out, T_row_out], [CPCP.iHmass]
                    )
                    * mdot_vec[-1 - i]
                )
        else:
            for i in range(G.n_CP):
                T_row_out = TH[-((i) * (G.n_CPR + 1)) - 1]
                E_out.append(
                    GetFluidPropLow(
                        EosPF, [CPCP.PT_INPUTS, P_out, T_row_out], [CPCP.iHmass]
                    )
                    * mdot_vec[-1 - i]
                )

        h_ave = sum(E_out) / sum(mdot_vec[-G.n_CP : G.n_rows])
        E_out_total = sum(E_out)
        T_out = CP("T", "H", h_ave, "P", P_out, F.PF_fluid)
        h_design_out = GetFluidPropLow(
            EosPF, [CPCP.PT_INPUTS, P_out, F.T_PF_out], [CPCP.iHmass]
        )
        error[(4 * G.n_cells) + (G.n_passes - 1) + ((G.n_passes) * (G.n_CP - 1))] = (
            (T_out - F.T_PF_out) * (sum(mdot_vec) / G.n_passes) * 1e3
        )

    # Heat transfer calculations
    for i in range(G.n_cells):
        # Amb node [i] is the outlet to the current cell
        node_amb_i1 = i + G.n_CPR  # Ambient fluid inlet node to current cell
        node_amb_i2 = i + (2 * G.n_CPR)  # Ambient node two prior to current cell
        node_amb_i3 = i - G.n_CPR  # Ambient node following current cell

        row = abs(i) // G.n_CPR

        x = (i % G.n_CPR) * G.dx
        if DV[row] == 1:
            x1 = x
            x2 = x + G.dx
        else:
            x1 = G.HX_length - x - G.dx
            x2 = G.HX_length - x

        if x1 == 0:
            x1 = 0.001
        if x2 == 0:
            x2 = 0.001

        k_PF = GetFluidPropLow(
            EosPF,
            [CPCP.PT_INPUTS, PH[i], (0.5 * (TH[i + row] + TH[i + 1 + row]))],
            [CPCP.iconductivity],
        )
        km_PF = GetFluidPropLow(
            EosPF, [CPCP.PT_INPUTS, PH[i], TH[i + row]], [CPCP.iconductivity]
        )
        kp_PF = GetFluidPropLow(
            EosPF, [CPCP.PT_INPUTS, PH[i], TH[i + 1 + row]], [CPCP.iconductivity]
        )
        k_amb = I.k_f(0.5 * (T_air[i] + T_air[node_amb_i1]))
        km_amb = I.k_f(T_air[node_amb_i1])
        kp_amb = I.k_f(T_air[i])

        Tb_PF = 0.5 * (TH[i + row] + TH[i + 1 + row])  # bulk temperature
        Pb_PF = 0.5 * (PH[i + row] + PH[i + 1 + row])

        Tb_amb = 0.5 * (T_air[node_amb_i1] + T_air[i])  # bulk temperature
        Pr_PF = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, Pb_PF, Tb_PF], [CPCP.iPrandtl])
        Pr_amb = I.Pr_f(Tb_amb)
        rho_PF_b = GetFluidPropLow(EosPF, [CPCP.PT_INPUTS, Pb_PF, Tb_PF], [CPCP.iDmass])
        rho_PF_w = GetFluidPropLow(
            EosPF, [CPCP.PT_INPUTS, Pb_PF, TW_H[i]], [CPCP.iDmass]
        )
        rho_amb = I.rho_f(Tb_amb)
        U_PF = abs(mdot_vec[row] / (rho_PF_b * G.A_CS))
        mu_PF = GetFluidPropLow(
            EosPF, [CPCP.PT_INPUTS, Pb_PF, Tb_PF], [CPCP.iviscosity]
        )
        Re_PF = rho_PF_b * U_PF * G.ID / mu_PF
        Nu_PF = calc_Nu(
            G,
            F,
            Re=Re_PF,
            Pr=Pr_PF,
            P=Pb_PF,
            Tb=Tb_PF,
            Tw=TW_H[i],
            rho_b=rho_PF_b,
            rho_w=rho_PF_w,
            Correlation=M.Nu_CorrelationH,
            mu_b=mu_PF,
            x1=x1,
            x2=x2,
            Dh=G.ID,
        )
        alpha_o = calc_alpha_amb(
            G,
            F,
            I,
            Pr=Pr_amb,
            Tb=Tb_amb,
            rho_b=rho_amb,
            Correlation=M.alpha_CorrelationC,
            K_c=K_c[row],
            row_correction=M.row_correction,
        )
        alpha_i = Nu_PF * k_PF / G.ID
        b = ((2 * alpha_o) / (G.k_fin * G.t_fin)) ** 0.5
        n_f = np.tanh(b * G.tube_OD * G.eta_phi / 2) / (b * G.tube_OD * G.eta_phi / 2)

        # Convective fluid-wall heat trasnfer hot side
        q3 = -(TW_H[i] - ((TH[i + row] + TH[i + 1 + row]) / 2)) * alpha_i * G.A_tube_i

        # Convective fluid-wall heat transfer cold side
        q4 = (
            -(((T_air[node_amb_i1] + T_air[i]) / 2) - TW_C[i])
            * alpha_o
            * ((G.A_f * n_f) + (G.A_r))
        )

        # PF side bulk convection
        if DV[row] > 0:
            q1_conv = (
                GetFluidPropLow(
                    EosPF, [CPCP.PT_INPUTS, PH[i + row], TH[i + row]], [CPCP.iHmass]
                )
                * mdot_vec[row]
                * DV[row]
            )
            q2_conv = (
                GetFluidPropLow(
                    EosPF,
                    [CPCP.PT_INPUTS, PH[i + row + 1], TH[i + row + 1]],
                    [CPCP.iHmass],
                )
                * mdot_vec[row]
                * DV[row]
                * -1
            )

            if i in inlet_cells or header_outlet_cells:
                q1_cond = (
                    -km_PF
                    * G.A_CS
                    / G.dx
                    / 2
                    * (0.5 * (TH[i + row] + TH[i + 1 + row]) - TH[i + row])
                )
            else:
                q1_cond = (
                    -km_PF
                    * G.A_CS
                    / G.dx
                    * (
                        0.5 * (TH[i + row] + TH[i + 1 + row])
                        - 0.5 * (TH[i + row] + TH[i - 1 + row])
                    )
                )

            if i in outlet_cells or header_inlet_cells:
                q2_cond = (
                    -kp_PF
                    * G.A_CS
                    / G.dx
                    / 2
                    * (TH[i + 1 + row] - 0.5 * (TH[i + row] + TH[i + 1 + row]))
                )
            else:
                q2_cond = (
                    -kp_PF
                    * G.A_CS
                    / G.dx
                    * (
                        0.5 * (TH[i + row] + TH[i + 1 + row])
                        - 0.5 * (TH[i + 1 + row] + TH[i + 2 + row])
                    )
                )
        else:
            q2_conv = (
                GetFluidPropLow(
                    EosPF, [CPCP.PT_INPUTS, PH[i + row], TH[i + row]], [CPCP.iHmass]
                )
                * mdot_vec[row]
                * DV[row]
            )
            q1_conv = (
                GetFluidPropLow(
                    EosPF,
                    [CPCP.PT_INPUTS, PH[i + row + 1], TH[i + row + 1]],
                    [CPCP.iHmass],
                )
                * mdot_vec[row]
                * DV[row]
                * -1
            )

            if i in inlet_cells or header_outlet_cells:
                q2_cond = (
                    -km_PF
                    * G.A_CS
                    / G.dx
                    / 2
                    * (0.5 * (TH[i + row] + TH[i + 1 + row]) - TH[i + row])
                )
            else:
                q2_cond = (
                    -km_PF
                    * G.A_CS
                    / G.dx
                    * (
                        0.5 * (TH[i + row] + TH[i + 1 + row])
                        - 0.5 * (TH[i + row] + TH[i - 1 + row])
                    )
                )

            if i in outlet_cells or header_inlet_cells:
                q1_cond = (
                    -kp_PF
                    * G.A_CS
                    / G.dx
                    / 2
                    * (TH[i + 1 + row] - 0.5 * (TH[i + row] + TH[i + 1 + row]))
                )
            else:
                q1_cond = (
                    -kp_PF
                    * G.A_CS
                    / G.dx
                    * (
                        0.5 * (TH[i + 1 + row] + TH[i + 2 + row])
                        - 0.5 * (TH[i + row] + TH[i + 1 + row])
                    )
                )

        q1 = q1_cond + q1_conv
        q2 = q2_cond + q2_conv

        # Amb side bulk convection
        q5_conv = I.h_f(T_air[node_amb_i1]) * F.mdot_C
        q6_conv = I.h_f(T_air[i]) * F.mdot_C

        if i >= (G.n_cells - G.n_CPR):
            q5_cond = (
                -km_amb
                * G.A_amb
                / G.pitch_longitudal
                / 2
                * (0.5 * (T_air[node_amb_i1] + T_air[i]) - T_air[node_amb_i1])
            )
        else:
            q5_cond = (
                -km_amb
                * G.A_amb
                / G.pitch_longitudal
                * (
                    0.5 * (T_air[node_amb_i1] + T_air[i])
                    - 0.5 * (T_air[node_amb_i1] + T_air[node_amb_i2])
                )
            )

        if i < G.n_CPR:
            q6_cond = (
                -km_amb
                * G.A_amb
                / G.pitch_longitudal
                / 2
                * (T_air[i] - 0.5 * (T_air[node_amb_i1] + T_air[i]))
            )
        else:
            q6_cond = (
                -km_amb
                * G.A_amb
                / G.pitch_longitudal
                * (
                    0.5 * (T_air[i] + T_air[node_amb_i3])
                    - 0.5 * (T_air[node_amb_i1] + T_air[i])
                )
            )

        # Wall conduction

        if i in inlet_cells or header_outlet_cells:
            q7 = 0
        else:
            q7 = (
                -(((TW_H[i] + TW_C[i]) / 2) - ((TW_H[i - 1] + TW_C[i - 1]) / 2))
                * G.k_wall
                * G.A_WCS
                / G.dx
            )

        if i in outlet_cells or header_inlet_cells:
            q8 = 0
        else:
            q8 = (
                -(((TW_H[i + 1] + TW_C[i + 1]) / 2) - ((TW_H[i] + TW_C[i]) / 2))
                * G.k_wall
                * G.A_WCS
                / G.dx
            )

        q5 = q5_cond + q5_conv
        q6 = q6_cond + q6_conv

        q9 = (
            -(TW_C[i] - TW_H[i])
            * G.k_wall
            * 2
            * np.pi
            * G.dx
            / np.log((G.ID + (2 * G.t_wall)) / G.ID)
        )

        error[i] = q1 + q2 - q3  # * (100/G.n_cells)
        error[G.n_cells + i] = q4 + q5 - q6  # * (100/G.n_cells)
        error[2 * G.n_cells + i] = q3 - q4 + q7 - q8  # * (100/G.n_cells)
        error[3 * G.n_cells + i] = q9 - (min(q3, q4))  # * (100/G.n_cells)

        if flag == 1:
            Gr = 9.81 * (rho_PF_w - rho_PF_b) * rho_PF_b * (G.ID**3) / (mu_PF**2)

            Bu_c = Gr / (Re_PF**2)
            Bu_c_array.append(Bu_c)

            Bu_j = Bu_c * (rho_PF_b / rho_PF_w) * ((x1 / G.ID) ** 2)
            Bu_j_array.append(Bu_j)

            Gr_p = Gr * Nu_PF
            Bu_p = Gr_p / (
                Re_PF**2.75
                * Pr_PF**0.5
                * (1 + 2.4 * Re_PF ** (-1 / 8) * (Pr_PF ** (2 / 3) - 1))
            )
            Bu_p_array.append(Bu_p)

            Q1.append(q1)
            Q2.append(q2)
            Q3.append(q3)
            Q4.append(q4)
            Q5.append(q5)
            Q6.append(q6)
            Q7.append(q7)
            Q8.append(q8)
            Q9.append(q9)
            Alpha_i.append(alpha_i)
            Alpha_o.append(alpha_o)
            dT.append(Tb_PF - Tb_amb)

            Ui = alpha_i * G.A_tube_i
            Uo = alpha_o * ((G.A_f * n_f) + (G.A_r))
            Uw = G.k_wall * 2 * np.pi * G.dx / np.log((G.ID + (2 * G.t_wall)) / G.ID)
            UA.append((Ui**-1 + Uo**-1 + Uw**-1) ** -1)

    # Error values for mass flow rate

    # Qout_PF_total = (sum(mdot_vec[0:G.n_CP])) * (GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,PH[0],TH[0]],[CPCP.iHmass]) - GetFluidPropLow(EosPF,[CPCP.PT_INPUTS,P_out,F.T_PF_out],[CPCP.iHmass]))
    # error[(4*G.n_cells) + (G.n_passes-1)+ ((G.n_passes)*(G.n_CP-1))] = (Q_acc + Qout_PF_total)
    # print("Q_acc ",Q_acc)
    # print("Qout_PF_total ",Qout_PF_total)

    T_air_out_ave = mean(T_air[0 : G.n_CPR])

    if M.cooler_type == 1:
        if CT.solver_type == 0:
            Verror = Draft_equation(v, CT, F, T_air_out_ave, dp_amb_total) * 1e4
            error[(4 * G.n_cells) + G.n_rows] = Verror

        if CT.solver_type == 1:
            CT.dim_update(d3)

            Verror = Draft_equation(v, CT, F, T_air_out_ave, dp_amb_total) * 1e1

            x2 = 0
            for i in range(G.n_CP):
                x1 = 0
                for j in range(G.n_passes):
                    x1 = x1 + mdot_vec[i + (j * G.n_CP)]
                x2 = x2 + (x1 / G.n_passes)
            mdot_i = x2

            # mdot_i = sum(mdot_vec[0:G.n_CP])

            mdot_PF_i_total = mdot_i * (CT.A_fr / CT.dA_HX)
            mdot_error = (
                (mdot_PF_i_total - CT.mdot_PF_total) / CT.mdot_PF_total
            ) * 1e2  # 1e3

            error[(4 * G.n_cells) + G.n_rows] = Verror
            error[(4 * G.n_cells) + G.n_rows + 1] = mdot_error  # *1e4

    for j in range(G.n_passes - 1):
        # Error value for continuity between passes
        mdot_header_out = [0]
        mdot_header_in = [0]
        for k in range(G.n_CP):
            mdot_header_out = mdot_header_out + mdot_vec[k + ((j + 1) * G.n_CP)]
            mdot_header_in = mdot_header_in + mdot_vec[k + (j * G.n_CP)]
        error[4 * G.n_cells + j] = (mdot_header_in - mdot_header_out) * 1e5  # 1e3 #1e5
    # print("Error of interest",error[4*G.n_cells:4*G.n_cells + j+2])
    if flag == 0:
        return error
    else:
        return (
            error,
            TH,
            T_air,
            TW_H,
            TW_C,
            Q1,
            Q2,
            Q3,
            Q4,
            Q5,
            Q6,
            Q7,
            Q8,
            Q9,
            Alpha_i,
            PH,
            dp_amb,
            mdot_vec,
            P_headers,
            v,
            Alpha_o,
            dT,
            d3,
            Bu_c_array,
            Bu_j_array,
            Bu_p_array,
            UA,
        )


# ----------------------PLOTTING----------------------------
# TODO: Remove this? I think it's redundant


def plot_HX(TH, T_air, TW_H, TW_C, G, F):
    TH_av = []
    i = 0
    for i in range(G.n_cells):
        TH_av.append((TH[i] + TH[i + 1]) / 2)

    T_air_in_unsorted = T_air[G.n_CPR : G.n_cells + G.n_CPR]
    T_air_out = T_air[0 : G.n_cells]
    T_air_in_sorted = np.zeros(G.n_cells)

    i = 0
    for i in range(G.n_rows):
        T_air_in_sorted[i * G.n_CPR : (i + 1) * G.n_CPR] = list(
            reversed(T_air_in_unsorted[i * G.n_CPR : (i + 1) * G.n_CPR])
        )
    fig1 = plt.figure()
    plt.plot(np.linspace(0, 1.0, num=G.n_cells), TH_av, label="Process Fluid")
    plt.plot(np.linspace(0, 1.0, num=G.n_cells), T_air_in_sorted, label="Air in")
    plt.plot(np.linspace(0, 1.0, num=G.n_cells), T_air_out, label="Air out")
    plt.plot(np.linspace(0, 1.0, num=G.n_cells), TW_H, "--", label="Pipe Wall Hot Side")
    plt.plot(
        np.linspace(0, 1.0, num=G.n_cells), TW_C, "--", label="Pipe Wall Cold Side"
    )

    plt.ylabel("Temperature (K)")
    plt.xlabel("Position along Heat Exchanger (normalised)")
    plt.title("Temperature Distributions in Heat Exchanger")
    plt.legend(loc=1)
    fig1.show()


def plot_HXq(Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, N_cell):
    fig2 = plt.figure()
    plt.plot(np.linspace(0, 1.0, num=N_cell), np.array(Q1) / 1.0e3, label="q1")
    plt.plot(np.linspace(0, 1.0, num=N_cell), np.array(Q2) / 1.0e3, label="q2")
    plt.plot(np.linspace(0, 1.0, num=N_cell), np.array(Q3) / 1.0e3, label="q3")
    plt.plot(np.linspace(0, 1.0, num=N_cell), np.array(Q4) / 1.0e3, label="q4")
    plt.plot(np.linspace(0, 1.0, num=N_cell), np.array(Q5) / 1.0e3, label="q5")
    plt.plot(np.linspace(0, 1.0, num=N_cell), np.array(Q6) / 1.0e3, label="q6")
    plt.plot(np.linspace(0, 1.0, num=N_cell), np.array(Q7) / 1.0e3, label="q7")
    plt.plot(np.linspace(0, 1.0, num=N_cell), np.array(Q8) / 1.0e3, label="q8")
    plt.ylabel("Heat Flow (kW)")
    plt.xlabel("Position along Heat Exchanger (normalised)")
    plt.title("Energy Fluxes in Heat Exchanger")
    plt.legend(loc=1)
    fig2.show()


def plot_generator(
    TH_final,
    T_air_final,
    TW_H_final,
    TW_C_final,
    Q1,
    Q2,
    Q3,
    Q4,
    Q5,
    Q6,
    Q7,
    Q8,
    Q9,
    T,
    G,
    F,
    M,
):
    print("Plotting results")
    plot_HX(TH_final, T_air_final, TW_H_final, TW_C_final, G, F)
    plot_HXq(Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, G.n_cells)

    plt.pause(10)  # <-------
    print("\n \n")
    input("<Hit Enter To Close Figures>")
    plt.close()


def GetFluidPropLow(Eos, state, props):
    """
    Low level fluid property call.
    Uses CoolProp low level interface for all supported fluids.
    Computes Shell Thermia D properties using 'HTFOilProps'

    Inputs:
    Eos   - Equation of state object for desired fluid.
    state - 1x3 list that specifies fluid state according to CoolProp
            low level syntax. Takes the form
            [InputPairId,Prop1Value,Prop2Value], for example
            state = [CP.PT_INPUTS,20e6,273.15+150].
    props - List specifying required output properties in CoolProp low
            level syntax. For example, props = [CP.iPrandtl,CP.iDmass]
            will give Prandtl number and density.

    Outputs:
    outputProps - Array containing desired output properties.

    Notes:
    Currently only supports pressure & temperature input pairs.
    """
    try:
        Eos.update(*state)
        outputProps = [Eos.keyed_output(k) for k in props]
        if len(outputProps) == 1:
            outputProps = outputProps[0]
        return outputProps
    except:
        print("state", state)
        print("props:", props)
        state[2] = 250
        Eos.update(*state)
        outputProps = [Eos.keyed_output(k) for k in props]
        if len(outputProps) == 1:
            outputProps = outputProps[0]
        # MyError('Invalid fluid specified.')
        return outputProps


# COOLING TOWER FUNCTIONS ---------------------------------------------------------------------


def T_calc(T_a1, z):
    T = T_a1 - (0.00975 * z)
    return T


def p_calc(p_a1, T_a1, z):
    Tz = T_calc(T_a1, z)
    p = p_a1 * ((Tz / T_a1) ** 3.5)
    return p


def Draft_equation(v, CT, F, T_heo, dp_HX):
    T_a3 = T_calc(CT.T_a1, CT.H3)

    # T_heo, dp_HX, T0, q, PF_dp = HX.main(ACHX_inputs,mod,v,F.mdot_PF,T_a3,T0,HX_L)

    T_a4 = T_calc(T_heo, CT.H3)
    T_a5 = T_calc(T_a4, CT.H5 - CT.H3)
    p_a6 = p_calc(CT.p_a1, CT.T_a1, CT.H5)
    p_a34 = p_calc(CT.p_a1, CT.T_a1, CT.H3)

    rho_a1 = CP("D", "T", CT.T_a1, "P", CT.p_a1, "Air")
    rho_a3 = CP("D", "T", CT.T_a1, "P", p_a34, "Air")
    rho_a4 = CP("D", "T", T_a4, "P", p_a34, "Air")
    rho_a5 = CP("D", "T", T_a5, "P", p_a6, "Air")
    rho_a6 = CP("D", "T", CT.T_a1, "P", p_a6, "Air")
    rho_a34 = 2 / ((1 / rho_a3) + (1 / rho_a4))

    mdot_a = v * CT.A_fr * rho_a3
    Fr_D = ((mdot_a / CT.A5) ** 2) / (rho_a5 * (rho_a6 - rho_a5) * 9.81 * CT.d5)

    K_he = dp_HX * 2 / ((v**2) * (rho_a34**2))
    K_ct = (
        -18.7
        + (8.095 * (1 / CT.R_hD))
        - (1.084 * (1 / CT.R_hD**2))
        + (0.0575 * (1 / CT.R_hD**3))
    ) * (
        K_he ** (0.165 - (0.035 * (1 / CT.R_hD)))
    )  # from Kroger eq 7.3.6 (p70)

    K_to = (-0.28 * (Fr_D**-1)) + (0.04 * (Fr_D**-1.5))  # Tower outlet losses
    K_tshe = (
        CT.C_Dts
        * CT.L_ts
        * CT.D_ts
        * CT.n_ts
        * (CT.A_fr**2)
        * (rho_a34 / rho_a1)
        / ((np.pi * CT.d3 * CT.H3) ** 3)
    )  # Tower support losses
    K_cthe = (
        K_ct * (rho_a34 / rho_a3) * ((CT.A_fr / CT.A3) ** 2)
    )  # Tower losses due to flow detatchment -
    K_ctche = (
        (1 - (2 / CT.sigma_c) + (1 / (CT.sigma_c**2)))
        * (rho_a34 / rho_a3)
        * ((CT.A_fr / CT.A3) ** 2)
    )  # Losses due to contraction through HX bundles
    K_ctehe = (
        (((1 - CT.A_fr) / CT.A3) ** 2) * (rho_a34 / rho_a4) * ((CT.A_fr / CT.A3) ** 2)
    )  # Losses due to expansion through HX bundles
    K_hes = 0.1

    K_sum = K_tshe + K_cthe + K_ctche + K_ctehe + K_hes

    term1 = p_a34 * ((1 - 0.00957 * (CT.H5 - CT.H3) / T_a4) ** 3.5)
    term2 = p_a6
    term3 = dp_HX
    term4 = (
        (1 / (2 * (CT.A_fr**2) * rho_a34))
        * K_sum
        * ((1 - 0.00957 * (CT.H5 - CT.H3) / T_a4) ** 3.5)
    )
    term5 = (1 / (2 * (CT.A5**2) * rho_a5)) * (1 + K_to)

    mdot12 = (term1 - term2 - term3) / (term4 + term5)

    if mdot12 < 0:
        mdot1 = -((-mdot12) ** 0.5)
    else:
        mdot1 = (mdot12) ** 0.5

    v_fr = mdot1 / (rho_a3 * CT.A_fr)

    Verror = (v_fr - v) / v
    # print("K_he: ",K_he)
    # print("K_cthe: ",K_cthe)
    # print("K_to: ",K_to)
    # print("K_tshe: ",K_tshe)
    # print("K_ctche: ",K_ctche)
    # print("K_ctehe: ",K_ctehe)
    # print("K_hes",K_hes)

    return Verror


# END - COOLING TOWER FUNCTIONS ---------------------------------------------------------------------


class MyError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
