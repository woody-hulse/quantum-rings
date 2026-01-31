## Model improvements

* Basic XGBoost and MLP, no added features *

```
============================================================
COMPARISON SUMMARY
============================================================

Metric                                  MLP      XGBoost     Winner
----------------------------------------------------------------
Training Time (s)                    0.9206       0.3359    XGBoost
Threshold Accuracy                   0.3200       0.1600        MLP
Runtime MSE                          4.8878       3.0600    XGBoost
Runtime MAE                          1.9938       1.3253    XGBoost
Challenge Threshold Score            0.4622       0.3200        MLP
Challenge Runtime Score              0.1397       0.2657    XGBoost
Challenge Combined Score             0.3332       0.2983        MLP

============================================================
WINNER: MLP (combined score: 0.3332 vs 0.2983)
============================================================
```

* Engineered max degree, degree entropy, clustering coefficient, connected components, layer/moment count, gates crossing middle cut *

============================================================
COMPARISON SUMMARY
============================================================

Metric                                  MLP      XGBoost     Winner
----------------------------------------------------------------
Training Time (s)                    1.4904       0.3342    XGBoost
Threshold Accuracy                   0.7692       0.3077        MLP
Runtime MSE                          1.2247       0.6976    XGBoost
Runtime MAE                          0.9169       0.6062    XGBoost
Challenge Threshold Score            0.8000       0.4744        MLP
Challenge Runtime Score              0.3997       0.5454    XGBoost
Challenge Combined Score             0.6399       0.5028        MLP

============================================================
WINNER: MLP (combined score: 0.6399 vs 0.5028)
============================================================