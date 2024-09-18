# Bachelorthesis: Economics Informed Neural Networks

Author: Malte Tölle

## Abstract:

Artifical intelligence has shown tremendous results in past years. However, in economics its applications has been limited due to their black-box nature. Policy implications derived from model forecasts must communicated to a wider audience in a causal manner. This is often not feasible with neural networks as their inner workings are often not fully comprehensible due to their non-linearity and high number of parameters. New Keynesian Models on the other hand strive for maxium explainability based on microfoundations sacrificing predictive performance.
In this bachelorthesis a step towards combining NKMs with the often superior, non-linear predictive performance of NNs is taken. The famous Smets-Wouters-Model is combined with an NN to improve predictive performance. Our findings show that predictive performance is on par or better than vector autoregressive models, if optimization procedures based on the Hessian are employed. These algorithms such as L-BFGS enable finding of the global minimum when many local minima exist. Additionally, we use SHAP values to improve explainability of the NNs predictions.

#### Comparison of Forecasting Performance

![Comparison of Forecasting Performance](https://github.com/maltetoelle/economics-informed-neural-networks/blob/main/comparison_forecasting_performance.pdf?raw=true)

## Replication:

```
conda env create -f env.yml
```
Run all cells in __results.iypnb__.


