from typing import List
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from statsmodels.tsa.api import VAR as VAR_
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import OBS_VAR_NAMES

torch.manual_seed(2)
np.random.seed(2)

N_OBS = len(OBS_VAR_NAMES) # 7

class AbstractModel:
    def __init__(self, n_lag: int, n_sw_forward: int, sw_params: List[str]):
        self.n_lag = n_lag
        self.n_sw_forward = n_sw_forward
        self.sw_params = sw_params

    def fit(self, inputs: np.ndarray, target: np.ndarray):
        return NotImplementedError

    def estimate(self, inputs: np.ndarray, n_timesteps: int = 1):
        # for now n_timesteps cannot be greater than n_sw_forward
        return NotImplementedError
    
    def add_prediction_to_inputs(self, inputs: np.ndarray, y_hat: np.ndarray):
        # shift observations and add prediction as input to next
        obs_inputs = inputs[:,:N_OBS*self.n_lag].reshape(-1,self.n_lag,N_OBS)
        obs_inputs = np.concatenate([obs_inputs[:,1:], y_hat[:,None]], axis=1)
        obs_inputs = obs_inputs.reshape(len(obs_inputs), -1)
        if len(self.sw_params):
            # shift sw predictions, need to discard last one as no newer one available
            sw_inputs = inputs[:,N_OBS*self.n_lag:].reshape(-1,self.n_sw_forward,len(self.sw_params))
            sw_inputs = np.concatenate([sw_inputs[:-1,1:], sw_inputs[1:,-1:]], axis=1)
            sw_inputs = sw_inputs.reshape(len(sw_inputs),-1)

            inputs = np.concatenate([obs_inputs[:-1], sw_inputs], axis=1)
        else:
            # actually not needed but to be consistent with experiments with SW
            inputs = obs_inputs[:-1]
        return inputs


class LinearNN_np(AbstractModel):

    def fit(self, inputs: np.ndarray, targets: np.ndarray):
        n_params = inputs.shape[1]+1
        x0 = np.random.normal(size=(n_params*N_OBS))

        def fun_nn(x):
            x = x.reshape(n_params,N_OBS)
            y_hat = inputs @ x[:-1] + x[-1]
            # MSE loss
            loss = np.sum((y_hat - targets)**2)
            # weight decay
            loss += 3e-5 * np.sum(x**2)
            return loss
        
        self.res = least_squares(fun_nn, x0)
    
    def estimate(self, inputs: np.ndarray, n_timesteps: int = 1):
        n_params = inputs.shape[1]+1
        res_params = self.res.x.reshape(n_params,N_OBS)
        res_A, res_b = res_params[:-1], res_params[-1]
        y_hat = []
        for t in range(n_timesteps):
            y_hat_t = inputs @ res_A + res_b
            y_hat.append(y_hat_t[:len(inputs)-n_timesteps+t,None])
            inputs = self.add_prediction_to_inputs(inputs=inputs, y_hat=y_hat_t)
            
        y_hat = np.concatenate(y_hat, axis=1)
        return y_hat
    

class NonLinearNN_np(AbstractModel):
    pass


class LinearNN_torch_backprop(AbstractModel):
    pass


class NonLinearNN_torch_backprop(AbstractModel):
    def fit(self, inputs: np.ndarray, targets: np.ndarray):
        inputs, targets = torch.from_numpy(inputs).float(), torch.from_numpy(targets).float()
        self.model = nn.Sequential(
            nn.Linear(inputs.size(1), 10),
            nn.Tanh(),
            nn.Linear(10,targets.size(1))
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        for _ in range(10):
            optimizer.zero_grad()
            output = self.model(inputs)
            loss = F.mse_loss(output, targets)
            loss.backward()
            optimizer.step()
    
    @torch.no_grad()
    def estimate(self, inputs: np.ndarray, n_timesteps: int = 1):
        y_hat = []
        for t in range(n_timesteps):
            inputs = torch.from_numpy(inputs).float()
            y_hat_t = self.model(inputs)
            y_hat.append(y_hat_t[:len(inputs)-n_timesteps+t,None])
            inputs = self.add_prediction_to_inputs(inputs=inputs, y_hat=y_hat_t)
            
        y_hat = np.concatenate(y_hat, axis=1)
        return y_hat


class LinearNN_torch_lbfgs(AbstractModel):
    pass


class NonLinearNN_torch_lbfgs(AbstractModel):
    def fit(self, inputs: np.ndarray, targets: np.ndarray):
        inputs, targets = torch.from_numpy(inputs).float(), torch.from_numpy(targets).float()
        hidden_size = 40
        self.model = nn.Sequential(
            # nn.Dropout1d(p=0.3),
            # nn.Linear(inputs.size(1),targets.size(1))

            # # nn.Dropout1d(p=0.1),
            nn.Linear(inputs.size(1), hidden_size),
            # nn.LayerNorm((hidden_size,)),
            # nn.BatchNorm1d(hidden_size,),
            # nn.Sigmoid(),
            # nn.LeakyReLU(),
            nn.Tanh(),
            nn.Dropout1d(p=0.1),
            nn.Linear(hidden_size,targets.size(1))

            # nn.Linear(inputs.size(1), hidden_size),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_size,targets.size(1))
        )
        self.model.train()
        print(self.model)
        optimizer = torch.optim.LBFGS(
            self.model.parameters(), 
            history_size=10, 
            max_iter=4, 
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            output = self.model(inputs)
            loss = F.mse_loss(output, targets)
            loss.backward()
            return loss

        for _ in range(20):
            optimizer.step(closure)
    
    @torch.no_grad()
    def estimate(self, inputs: np.ndarray, n_timesteps: int = 1):
        self.model.eval()
        y_hat = []
        for t in range(n_timesteps):
            inputs = torch.from_numpy(inputs).float()
            y_hat_t = self.model(inputs)
            y_hat.append(y_hat_t[:len(inputs)-n_timesteps+t,None])
            inputs = self.add_prediction_to_inputs(inputs=inputs, y_hat=y_hat_t)
            
        y_hat = np.concatenate(y_hat, axis=1)
        return y_hat

class VAR(AbstractModel):
    def fit(self, inputs: np.ndarray, targets: np.ndarray):
        inputs_df = pd.DataFrame(inputs, columns=OBS_VAR_NAMES)
        model = VAR_(inputs_df)
        self.fitted_model = model.fit(self.n_lag)
    
    def estimate(self, inputs: np.ndarray, n_timesteps: int = 1):
        inputs_df = pd.DataFrame(inputs, columns=OBS_VAR_NAMES)
        y_hat_test_var = [
            self.fitted_model.forecast(inputs_df.values[i:i+self.n_lag], steps=n_timesteps)[None] 
            for i in range(len(inputs_df)-self.n_lag)
        ]
        y_hat_test_var = np.concatenate(y_hat_test_var, axis=0)
        return y_hat_test_var