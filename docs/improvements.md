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

# Engineered max degree, degree entropy, clustering coefficient, connected components, layer/moment count, gates crossing middle cut

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


# Improved evaluations (more training runs) and more features

============================================================
XGBOOST MODEL REPORT
5-fold cross-validation, 20 runs per fold
Total evaluations: 100
============================================================

--- Per-Fold Results ---
Fold      Train    Val   Thresh Acc     Combined
--------------------------------------------------
1           112     25       0.5880       0.3050
2           112     25       0.6320       0.4079
3           109     28       0.5643       0.3620
4           110     27       0.4000       0.3219
5           105     32       0.7125       0.2886

--- Overfitting Check (Train vs Val) ---
Metric                      Train Mean     Val Mean          Gap
-------------------------------------------------------------
Threshold Accuracy              1.0000       0.5794      +0.4206
Runtime MSE                     0.0003       1.1220      +1.1217
Runtime MAE                     0.0112       0.7930      +0.7818

--- Validation Metrics (Aggregated) ---
Metric                                 Mean          Std          Min          Max
------------------------------------------------------------------------------
Training Time (s)                    0.3927       0.2510       0.3039       1.4974
Val Threshold Accuracy               0.5794       0.1271       0.2963       0.7500
Val Runtime MSE                      1.1220       0.4469       0.3860       2.2456
Val Runtime MAE                      0.7930       0.1556       0.4951       1.1239
Challenge Threshold Score            0.6656       0.0895       0.4444       0.8571
Challenge Runtime Score              0.4204       0.0728       0.2587       0.5504
Challenge Combined Score             0.3371       0.0532       0.2345       0.4687

------------------------------------------------------------
Final Score: 0.3371 ± 0.0532
============================================================

============================================================
MLP MODEL REPORT
5-fold cross-validation, 20 runs per fold
Total evaluations: 100
============================================================

--- Per-Fold Results ---
Fold      Train    Val   Thresh Acc     Combined
--------------------------------------------------
1           112     25       0.6180       0.2536
2           112     25       0.9160       0.4210
3           109     28       0.6464       0.3539
4           110     27       0.5833       0.3254
5           105     32       0.7891       0.2809

--- Overfitting Check (Train vs Val) ---
Metric                      Train Mean     Val Mean          Gap
-------------------------------------------------------------
Threshold Accuracy              0.9378       0.7106      +0.2273
Runtime MSE                     2.3135       3.1266      +0.8131
Runtime MAE                     1.0461       1.2623      +0.2162

--- Validation Metrics (Aggregated) ---
Metric                                 Mean          Std          Min          Max
------------------------------------------------------------------------------
Training Time (s)                    1.1205       0.5046       0.2149       2.8265
Val Threshold Accuracy               0.7106       0.1984       0.0000       0.9600
Val Runtime MSE                      3.1266       3.9959       0.1822      19.0163
Val Runtime MAE                      1.2623       0.9029       0.3345       4.0444
Challenge Threshold Score            0.7435       0.1837       0.1537       0.9600
Challenge Runtime Score              0.3621       0.1877       0.0000       0.7159
Challenge Combined Score             0.3270       0.1731       0.0000       0.6285

------------------------------------------------------------
Final Score: 0.3270 ± 0.1731
============================================================

============================================================
CATBOOST MODEL REPORT
5-fold cross-validation, 32 runs per fold
Total evaluations: 160
============================================================

--- Per-Fold Results ---
Fold      Train    Val   Thresh Acc     Combined
--------------------------------------------------
1           112     25       0.6938       0.4126
2           112     25       0.5450       0.4052
3           109     28       0.7031       0.4795
4           110     27       0.4236       0.3393
5           105     32       0.5840       0.2865

--- Overfitting Check (Train vs Val) ---
Metric                      Train Mean     Val Mean          Gap
-------------------------------------------------------------
Threshold Accuracy              0.9879       0.5899      +0.3980
Runtime MSE                     0.0210       0.9979      +0.9770
Runtime MAE                     0.0769       0.7196      +0.6427

--- Validation Metrics (Aggregated) ---
Metric                                 Mean          Std          Min          Max
------------------------------------------------------------------------------
Training Time (s)                    0.1205       0.2109       0.0649       1.3432
Val Threshold Accuracy               0.5899       0.1480       0.2500       0.8571
Val Runtime MSE                      0.9979       0.4665       0.3282       2.2710
Val Runtime MAE                      0.7196       0.1399       0.4298       1.0573
Challenge Threshold Score            0.6908       0.1191       0.4400       0.9286
Challenge Runtime Score              0.4452       0.0908       0.3012       0.6679
Challenge Combined Score             0.3846       0.0824       0.2045       0.5933

------------------------------------------------------------
Final Score: 0.3846 ± 0.0824
============================================================
