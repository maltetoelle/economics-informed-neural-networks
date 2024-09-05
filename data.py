from typing import List, Tuple
import numpy as np
import scipy.io

# to ensure same order always
OBS_VAR_NAMES = ['dy', 'dc', 'dinve', 'labobs', 'dw', 'pinfobs', 'robs']

ENDO_NAMES = {
    'labobs': {'tex': r'${lHOURS}$', 'long_name': 'log hours worked'},
    'robs': {'tex': r'${FEDFUNDS}$', 'long_name': 'Federal funds rate'},
    'pinfobs': {'tex': r'${dlP}$', 'long_name': 'Inflation'},
    'dy': {'tex': r'${dlGDP}$', 'long_name': 'Output growth rate'},
    'dc': {'tex': r'${dlCONS}$', 'long_name': 'Consumption growth rate'},
    'dinve': {'tex': r'${dlINV}$', 'long_name': 'Investment growth rate'},
    'dw': {'tex': r'${dlWAG}$', 'long_name': 'Wage growth rate'},
    'ewma': {'tex': r'${\eta^{w,aux}}$', 'long_name': 'Auxiliary wage markup moving average variable'},
    'epinfma': {'tex': r'${\eta^{p,aux}}$', 'long_name': 'Auxiliary price markup moving average variable'},
    'zcapf': {'tex': r'${z^{flex}}$', 'long_name': 'Capital utilization rate flex price economy'},
    'rkf': {'tex': r'${r^{k,flex}}$', 'long_name': 'rental rate of capital flex price economy'},
    'kf': {'tex': r'${k^{s,flex}}$', 'long_name': 'Capital services flex price economy'},
    'pkf': {'tex': r'${q^{flex}}$', 'long_name': 'real value of existing capital stock flex price economy'},
    'cf': {'tex': r'${c^{flex}}$', 'long_name': 'Consumption flex price economy'},
    'invef': {'tex': r'${i^{flex}}$', 'long_name': 'Investment flex price economy'},
    'yf': {'tex': r'${y^{flex}}$', 'long_name': 'Output flex price economy'},
    'labf': {'tex': r'${l^{flex}}$', 'long_name': 'hours worked flex price economy'},
    'wf': {'tex': r'${w^{flex}}$', 'long_name': 'real wage flex price economy'},
    'rrf': {'tex': r'${r^{flex}}$', 'long_name': 'real interest rate flex price economy'},
    'mc': {'tex': r'${\mu_p}$', 'long_name': 'gross price markup'},
    'zcap': {'tex': r'${z}$', 'long_name': 'Capital utilization rate'},
    'rk': {'tex': r'${r^{k}}$', 'long_name': 'rental rate of capital'},
    'k': {'tex': r'${k^{s}}$', 'long_name': 'Capital services'},
    'pk': {'tex': r'${q}$', 'long_name': 'real value of existing capital stock'},
    'c': {'tex': r'${c}$', 'long_name': 'Consumption'},
    'inve': {'tex': r'${i}$', 'long_name': 'Investment'},
    'y': {'tex': r'${y}$', 'long_name': 'Output'},
    'lab': {'tex': r'${l}$', 'long_name': 'hours worked'},
    'pinf': {'tex': r'${\pi}$', 'long_name': 'Inflation'},
    'w': {'tex': r'${w}$', 'long_name': 'real wage'},
    'r': {'tex': r'${r}$', 'long_name': 'nominal interest rate'},
    'a': {'tex': r'${\varepsilon_a}$', 'long_name': 'productivity process'},
    'b': {'tex': r'${c_2*\varepsilon_t^b}$', 'long_name': 'Scaled risk premium shock'},
    'g': {'tex': r'${\varepsilon^g}$', 'long_name': 'Exogenous spending'},
    'qs': {'tex': r'${\varepsilon^i}$', 'long_name': 'Investment-specific technology'},
    'ms': {'tex': r'${\varepsilon^r}$', 'long_name': 'Monetary policy shock process'},
    'spinf': {'tex': r'${\varepsilon^p}$', 'long_name': 'Price markup shock process'},
    'sw': {'tex': r'${\varepsilon^w}$', 'long_name': 'Wage markup shock process'},
    'kpf': {'tex': r'${k^{flex}}$', 'long_name': 'Capital stock flex price economy'},
    'kp': {'tex': r'${k}$', 'long_name': 'Capital stock'}
}

SHOCK_NAMES = {
    'ea': {'tex': r'${\eta^a}$', 'long_name': 'productivity shock'},
    'eb': {'tex': r'${\eta^b}$', 'long_name': 'risk premium shock'},
    'eg': {'tex': r'${\eta^g}$', 'long_name': 'Spending shock'},
    'eqs': {'tex': r'${\eta^i}$', 'long_name': 'Investment-specific technology shock'},
    'em': {'tex': r'${\eta^m}$', 'long_name': 'Monetary policy shock'},
    'epinf': {'tex': r'${\eta^{p}}$', 'long_name': 'Price markup shock'},
    'ew': {'tex': r'${\eta^{w}}$', 'long_name': 'Wage markup shock'}
}

PARAMETER_NAMES = {
    'curvw': {'tex': r'${\varepsilon_w}$', 'long_name': 'Curvature Kimball aggregator wages'},
    'cgy': {'tex': r'${\rho_{ga}}$', 'long_name': 'Feedback technology on exogenous spending'},
    'curvp': {'tex': r'${\varepsilon_p}$', 'long_name': 'Curvature Kimball aggregator prices'},
    'constelab': {'tex': r'${\bar l}$', 'long_name': 'steady state hours'},
    'constepinf': {'tex': r'${\bar \pi}$', 'long_name': 'steady state inflation rate'},
    'constebeta': {'tex': r'${100(\beta^{-1}-1)}$', 'long_name': 'time preference rate in percent'},
    'cmaw': {'tex': r'${\mu_w}$', 'long_name': 'coefficient on MA term wage markup'},
    'cmap': {'tex': r'${\mu_p}$', 'long_name': 'coefficient on MA term price markup'},
    'calfa': {'tex': r'${\alpha}$', 'long_name': 'capital share'},
    'czcap': {'tex': r'${\psi}$', 'long_name': 'capacity utilization cost'},
    'csadjcost': {'tex': r'${\varphi}$', 'long_name': 'investment adjustment cost'},
    'ctou': {'tex': r'${\delta}$', 'long_name': 'depreciation rate'},
    'csigma': {'tex': r'${\sigma_c}$', 'long_name': 'risk aversion'},
    'chabb': {'tex': r'${\lambda}$', 'long_name': 'external habit degree'},
    'ccs': {'tex': r'${d_4}$', 'long_name': 'Unused parameter'},
    'cinvs': {'tex': r'${d_3}$', 'long_name': 'Unused parameter'},
    'cfc': {'tex': r'${\phi_p}$', 'long_name': 'fixed cost share'},
    'cindw': {'tex': r'${\iota_w}$', 'long_name': 'Indexation to past wages'},
    'cprobw': {'tex': r'${\xi_w}$', 'long_name': 'Calvo parameter wages'},
    'cindp': {'tex': r'${\iota_p}$', 'long_name': 'Indexation to past prices'},
    'cprobp': {'tex': r'${\xi_p}$', 'long_name': 'Calvo parameter prices'},
    'csigl': {'tex': r'${\sigma_l}$', 'long_name': 'Frisch elasticity'},
    'clandaw': {'tex': r'${\phi_w}$', 'long_name': 'Gross markup wages'},
    'crdpi': {'tex': r'${r_{\Delta \pi}}$', 'long_name': 'Unused parameter'},
    'crpi': {'tex': r'${r_{\pi}}$', 'long_name': 'Taylor rule inflation feedback'},
    'crdy': {'tex': r'${r_{\Delta y}}$', 'long_name': 'Taylor rule output growth feedback'},
    'cry': {'tex': r'${r_{y}}$', 'long_name': 'Taylor rule output level feedback'},
    'crr': {'tex': r'${\rho}$', 'long_name': 'interest rate persistence'},
    'crhoa': {'tex': r'${\rho_a}$', 'long_name': 'persistence productivity shock'},
    'crhoas': {'tex': r'${d_2}$', 'long_name': 'Unused parameter'},
    'crhob': {'tex': r'${\rho_b}$', 'long_name': 'persistence risk premium shock'},
    'crhog': {'tex': r'${\rho_g}$', 'long_name': 'persistence spending shock'},
    'crhols': {'tex': r'${d_1}$', 'long_name': 'Unused parameter'},
    'crhoqs': {'tex': r'${\rho_i}$', 'long_name': 'persistence investment-specific technology shock'},
    'crhoms': {'tex': r'${\rho_r}$', 'long_name': 'persistence monetary policy shock'},
    'crhopinf': {'tex': r'${\rho_p}$', 'long_name': 'persistence price markup shock'},
    'crhow': {'tex': r'${\rho_w}$', 'long_name': 'persistence wage markup shock'},
    'ctrend': {'tex': r'${\bar \gamma}$', 'long_name': 'net growth rate in percent'},
    'cg': {'tex': r'${\frac{\bar g}{\bar y}}$', 'long_name': 'steady state exogenous spending share'}
}

def read_smets_wouters_observations():
    data = scipy.io.loadmat('Smets_Wouters_2007/usmodel_data.mat')
    data = {k: data[k].reshape(-1,1) for k in OBS_VAR_NAMES}
    data = np.concatenate([data[k] for k in OBS_VAR_NAMES], axis=1)
    return data

def read_smets_wouters_prediction():
    dynare_output = scipy.io.loadmat('Smets_Wouters_2007/Smets_Wouters_2007/Output/Smets_Wouters_2007_results.mat')
    dynare_prediction = dynare_output['oo_']['FilteredVariablesKStepAhead'][0,0]
    return dynare_prediction

def prepare_train_test_data(
        n_trainsteps: int = 200,
        n_lag: int = 4,
        n_sw_forward: int = 4,
        sw_params: List[str] = OBS_VAR_NAMES
    ) -> Tuple[Tuple[np.ndarray]]:
    obs = read_smets_wouters_observations()
    sw_pred = read_smets_wouters_prediction()
    sw_pred = sw_pred[...,:len(obs)]

    obs_with_lag = [obs[i:i+n_lag][None] for i in range(len(obs)-n_lag)]
    obs_with_lag = np.concatenate(obs_with_lag, axis=0)
    obs_with_lag = obs_with_lag.reshape(len(obs_with_lag), -1)
    sw_params_idx = [i for i, n in enumerate(ENDO_NAMES) if n in sw_params]
    sw_pred_reshaped = sw_pred[:n_sw_forward,sw_params_idx,:len(obs_with_lag)].transpose(2,0,1)
    sw_pred_reshaped = sw_pred_reshaped.reshape(len(obs_with_lag), -1)
    
    inputs = np.concatenate([obs_with_lag, sw_pred_reshaped], axis=1)
    targets = obs[n_lag+1:]
    inputs = inputs[:-1]

    train_inputs, test_inputs = inputs[:n_trainsteps], inputs[n_trainsteps:]
    train_targets, test_targets = targets[:n_trainsteps], targets[n_trainsteps:]

    return (train_inputs, test_inputs), (train_targets, test_targets)