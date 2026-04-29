| run | channels | train_patches | val_patches | test_patches | best_model | best_eval_rmse | best_eval_r2 | artifact_fixed | notable_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs 04_24 | 4 | 266 | 69 | 26 | U-Net | 25.487 | 0.693 | no | legacy 4-channel baseline; raw best metrics but invalid border coverage |
| runs 04_27 | 4 | 286 | 72 | 32 | U-Net + Consistency | 26.312 | 0.672 | yes | post-fix rerun; artifact removed but quality regressed |
| 20260427_134448 | 4 | 876 | 174 | 78 | U-Net | 26.171 | 0.676 | yes | dense patches, augmentation, calibration |
| 20260427_143154 | 5 | 876 | 174 | 78 | U-Net | 25.762 | 0.686 | yes | added NLCD land cover channel |
| 20260427_152647 | 5 | 876 | 174 | 78 | U-Net | 25.762 | 0.686 | yes | latest full comparison including Prithvi 100M and 300M |
