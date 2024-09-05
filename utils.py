from typing import Dict
import numpy as np
import pandas as pd
from scipy import stats
from data import OBS_VAR_NAMES

def rsme(y_hat: np.ndarray, y_gt: np.ndarray):
    return np.sqrt(np.mean((y_gt - y_hat)**2, axis=0))

def paired_sample_t_test(sample1: np.ndarray, sample2: np.ndarray):
    t_statistic, p_value = stats.ttest_rel(sample1, sample2)
    return np.round(p_value, 3)

def rsmes_per_method(y_gt: np.ndarray, y_hats: Dict[str, np.ndarray]):
    RSMES = []
    for method, y_hat in y_hats.items():
        # for each forecasting step
        for t in range(y_hat.shape[1]):
            y_hat_t = y_hat[:,t]
            y_gt_t = y_gt[t:len(y_hat_t)+t]
            rsme_t = rsme(y_hat_t,y_gt_t)
            RSMES.append({
                'method': method,
                't': t, 
                **{v: x for v, x in zip(OBS_VAR_NAMES, rsme_t)}
            })
    rsme_df = pd.DataFrame(RSMES)
    return rsme_df