from dataclasses import dataclass
import numpy as np
import pandas as pd
from brian2 import *

@dataclass
class Parameters:
    """
    defining common parameter values for different hvc neuron types
    """
    
    def __init__(neuron) -> None:
        """
        Constructor
        """

        neuron.Vl = -70*mV
        neuron.Vk = -90*mV
        neuron.VNa = 50*mV
        neuron.Vh = -30*mV
        neuron.g_l = 2*nS
        neuron.g_Cal = 19*nS
        neuron.g_Nap = 1*nS
        neuron.tau_bar_n = 10*msecond
        neuron.tau_bar_hp = 1_000*msecond
        neuron.tau_e = 20*msecond
        neuron.tau_h = 1*msecond
        neuron.tau_rs = 1_500*msecond
        neuron.tau_r0 = 200*msecond
        neuron.tau_r1 = 87.5*msecond
        neuron.theta_m = -35*mV
        neuron.theta_n = -30*mV
        neuron.theta_s = -20*mV
        neuron.theta_mp = -40*mV
        neuron.theta_hp = -48*mV
        neuron.theta_a = -20*mV
        neuron.theta_e = -60*mV
        neuron.theta_rf = -105*mV
        neuron.theta_rs = -105*mV
        neuron.theta_at = -65*mV
        neuron.theta_b = 0.4*mV
        neuron.theta_rt = -67*mV
        neuron.theta_rrt = 68*mV
        neuron.sigma_m = -5*mV
        neuron.sigma_n = -5*mV
        neuron.sigma_s = -0.05*mV
        neuron.sigma_mp = -6*mV
        neuron.sigma_hp = 6*mV
        neuron.sigma_a = -10*mV
        neuron.sigma_e = 5*mV
        neuron.sigma_rf = 5*mV
        neuron.sigma_rs = 25*mV
        neuron.sigma_at = -7.8*mV
        neuron.sigma_b = -0.1*mV
        neuron.sigma_rt = 2*mV
        neuron.sigma_rrt = 2.2*mV
        neuron.f = 0.1
        neuron.epsilon = 0.0015*umolar*(1/(pamp*msecond))
        neuron.k_Ca = 0.3*(1/msecond)
        neuron.b_Ca = 0.1*umolar
        neuron.k_s = 0.5*umolar
        neuron.R_pump = 0.0006*(mmolar/msecond)
        neuron.K_p = 15*mmolar
        neuron.p_rf = 100


class HVCX_Params(Parameters):

    def __init__(neuron) -> None:
        """
        Constructor for HVC_X neurons
        """
        super().__init__()
        neuron.g_Na = 450*nS
        neuron.g_K = 50*nS
        neuron.g_SK = 6*nS
        neuron.g_KNa = 40*nS
        neuron.g_h = 4*nS
        neuron.g_A = 5*nS
        neuron.g_CaT = 2.7*nS
        neuron.C_m = 100*pfarad
        neuron.k_r = 0.3


class HVCRA_Params(Parameters):

    def __init__(neuron) -> None:
        """
        Constructor for HVC_RA neurons
        """
        super().__init__()
        neuron.g_Na = 300*nS
        neuron.g_K = 400*nS
        neuron.g_SK = 27*nS
        neuron.g_KNa = 500*nS
        neuron.g_h = 1*nS
        neuron.g_A = 150*nS
        neuron.g_CaT = 0.6*nS
        neuron.C_m = 20*pfarad
        neuron.k_r = 0.95


class HVCINT_Params(Parameters):

    def __init__(neuron) -> None:
        """
        Constructor for HVC_INT neurons
        """
        super().__init__()
        neuron.g_Na = 800*nS
        neuron.g_K = 1700*nS
        neuron.g_SK = 1*nS
        neuron.g_KNa = 1*nS
        neuron.g_h = 4*nS
        neuron.g_A = 1*nS
        neuron.g_CaT = 1.1*nS
        neuron.C_m = 75*pfarad
        neuron.k_r = 0.01



    

