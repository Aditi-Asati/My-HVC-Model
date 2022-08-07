from dataclasses import dataclass
from tkinter import NO
import numpy as np
from brian2 import *
import json

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
        neuron.Ca_ex = 2.5*mmolar
        neuron.F = 96_485*(coulomb/mole)
        neuron.R = 8.314*(joule/(mole*kelvin))
        neuron.T = 298*kelvin
        neuron.alpha_Na = 0.0001*mmolar*((msecond*uamp)**(-1))
        neuron.Naeq = 8.0*mmolar

class HVCX_Params(Parameters):

    def __init__(neuron, config_file=None) -> None:
        """
        Constructor for defining parameters specific to HVC_X neurons
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

        if config_file:
            with open(config_file) as f:
                content = json.loads(f.read())
                for k, v in content.items():
                    exec(f"{k} = {v}")

class HVCRA_Params(Parameters):

    def __init__(neuron, config_file=None) -> None:
        """
        Constructor for defining parameters specific to HVC_RA neurons
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

        if config_file:
            with open(config_file) as f:
                content = json.loads(f.read())
                for k, v in content.items():
                    exec(f"{k} = {v}")

class HVCINT_Params(Parameters):

    def __init__(neuron, config_file=None) -> None:
        """
        Constructor for defining parameters specific to HVC_INT neurons
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

        
        if config_file:
            with open(config_file) as f:
                content = json.loads(f.read())
                for k, v in content.items():
                    exec(f"{k} = {v}")



def stimuli(df, mag, stim='pulse', dur=3, st=1, pwidth=0.05, gap=2, base=0, noise=0.2, ramp=1, rampup_t=None, rampdown_t=None, psp_dur=0.04, freq=1/(0.02), synaptize=False, noisy=False):   
    T = np.round(df['t'].max()) #maximum time
    #step current
    if stim == 'step':
        df['step'] = np.ones(np.size(df['t']))*base
        step = (df['t']>st) & (df['t']<st+dur)
        df['step'][step] = mag
    #sine current
    elif stim == 'sin':
        df['sin'] = np.ones(np.size(df['t']))*base
        step = (df['t']>st) & (df['t']<st+dur)
        df['sin'][step] = mag*0.1*np.sin(1e2*df['t'][step]+5)+mag
    #linear increase
    elif stim == 'lin':
        df['lin'] = np.ones(np.size(df['t']))*base
        step = (df['t']>st) & (df['t']<st+dur)
        t_step = df['t'][step]; l_t_step = np.max(t_step)-np.min(t_step);
        df['lin'][step] = (mag/l_t_step)*(t_step)-(mag/l_t_step)*(t_step.iloc[0])
    #pulsatile
    elif stim == 'pulse':
        df['pulse'] = np.ones(np.size(df['t']))*base
        step = (df['t']<0)
        for i in np.arange(st, st+dur, pwidth+gap):
            step = step|((df['t']>=i)&(df['t']<=i+pwidth))
        df['pulse'][step] = mag    
    #bump
    elif stim == 'bump':
        df['bump'] = np.ones(np.size(df['t']))*base
        bump = (df['t']>st) & (df['t']<st+dur)
        df['bump'][bump] = mag
        rampup_t = ramp/2 if (rampup_t is None) else rampup_t; rampdown_t = ramp/2 if (rampdown_t is None) else rampdown_t
        rampup = (df['t']>st) & (df['t']<st+rampup_t); rampdown = (df['t']>st+dur-rampdown_t) & (df['t']<st+dur)
        t_step = df['t'][rampup]; l_t_step = np.max(t_step)-np.min(t_step);
        df['bump'][rampup] = (mag/l_t_step)*(t_step)-(mag/l_t_step)*(t_step.iloc[0])
        t_step = df['t'][rampdown]; l_t_step = np.max(t_step)-np.min(t_step);
        df['bump'][rampdown] = (-mag/l_t_step)*(t_step)+(mag/l_t_step)*(t_step.iloc[-1])
    
    #synaptize the stimulus - makes the stimulus high frequency pulse like (realistic)
    if synaptize==True:
        step = (df['t']<0)
        for i in np.arange(0, T, psp_dur+(1/freq)):
            step = step|((df['t']>=i)&(df['t']<=i+(1/freq)))
        df[stim][step] = 0    
    #make the stimulus noisy                     
    if noisy==True:
        df[stim] = df[stim]+(np.ones(np.size(df['t']))*noise*np.random.uniform(-1,1,np.size(df['t'])))        

    return df[stim] #return the stimulus asked for
    
if __name__ == "__main__":
    pass 
