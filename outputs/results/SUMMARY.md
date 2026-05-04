# Experimental Results Summary
Generated: 2026-04-21 19:37:34
Total runtime: 52.62 hours (3156.9 min)

## Runtime per experiment
| Experiment | Time (min) |
|---|---|
| pipeline_diagram | 0.0 |
| payoff_vega | 0.0 |
| lr_finder | 0.0 |
| bs_ann | 276.9 |
| iv_ann | 644.6 |
| iv_speed | 0.7 |
| heston_ann | 234.6 |
| heston_iv_surface | 0.1 |
| lr_schedules | 1954.3 |
| dataset_size | 45.7 |

## Table 5: BS-ANN

| Dataset | MSE | RMSE | MAE | MAPE | R2 |
|---|---|---|---|---|---|
| train_wide | 7.007e-09 | 8.371e-05 | 6.322e-05 | 2.054e+02 | 0.9999999 |
| test_wide | 7.230e-09 | 8.503e-05 | 6.415e-05 | 2.118e+02 | 0.9999998 |
| test_narrow | 6.387e-09 | 7.992e-05 | 6.109e-05 | 1.541e+02 | 0.9999998 |

Article target: MSE ~8e-9, R2 > 0.99999

## Table 7: IV-ANN

| Input | MSE | MAE | MAPE | R2 |
|---|---|---|---|---|
| Without scaling | 1.526e-05 | 1.608e-03 | 6.533e-03 | 0.9997802 |
| With gradient-squash | 1.315e-07 | 2.775e-04 | 7.826e-04 | 0.9999981 |

Article target (scaled): MSE ~1.5e-8, R2 > 0.9999998

## Table 8: Speed comparison (IV computation)

| Method | Time (s) | Robust |
|---|---|---|
| Newton-Raphson | 6.46 | No |
| Brent | 7.62 | No |
| Secant | 5.35 | No |
| Bisection | 20.59 | No |
| IV-ANN (CPU) | 0.02 | Yes |

## Table 10: Heston-ANN

| Dataset | MSE | RMSE | MAE | MAPE | R2 |
|---|---|---|---|---|---|
| train | 1.884e-08 | 1.373e-04 | 8.623e-05 | 3.398e+01 | 0.9999992 |
| test | 2.345e-08 | 1.531e-04 | 9.099e-05 | 1.140e+01 | 0.9999990 |

Article target: MSE ~1.5e-8, R2 > 0.9999993

## Table 11: Heston-ANN + IV-ANN pipeline

| Case | RMSE | MAE | MAPE | R2 |
|---|---|---|---|---|
| case1 | 5.430e-04 | 4.171e-04 | 1.328e-03 | 0.9399778 |
| case2 | 4.348e-04 | 3.462e-04 | 1.105e-03 | 0.9336404 |
| surface | 4.328e-04 | 3.434e-04 | 1.094e-03 | 0.9565116 |

## Table 3: Dataset size study

| Case | Train size (x baseline) | Train MSE | Test MSE | R2 (%) |
|---|---|---|---|---|
| 0 | x0.125 | 3.088e-03 | 3.156e-03 | 95.4465 |
| 1 | x0.25 | 7.079e-04 | 7.471e-04 | 98.9221 |
| 2 | x0.5 | 9.391e-05 | 1.064e-04 | 99.8465 |
| 3 | x1 | 2.390e-05 | 2.687e-05 | 99.9612 |
| 4 | x2 | 8.203e-06 | 8.799e-06 | 99.9873 |
| 5 | x4 | 2.446e-06 | 2.582e-06 | 99.9963 |
| 6 | x8 | 8.723e-07 | 9.367e-07 | 99.9986 |

