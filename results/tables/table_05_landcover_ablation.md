| model | channels_before | channels_after | eval_rmse_before | eval_rmse_after | delta_eval_rmse | eval_r2_before | eval_r2_after | delta_eval_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RF | nlcd, ndvi, lst, tmean | nlcd, landcover, ndvi, lst, tmean | 30.477 | 29.670 | -0.807 | 0.560 | 0.583 | 0.023 |
| U-Net | nlcd, ndvi, lst, tmean | nlcd, landcover, ndvi, lst, tmean | 26.171 | 25.762 | -0.409 | 0.676 | 0.686 | 0.010 |
| U-Net + Consistency | nlcd, ndvi, lst, tmean | nlcd, landcover, ndvi, lst, tmean | 26.718 | 25.818 | -0.900 | 0.662 | 0.684 | 0.022 |
| Prithvi 100M | nlcd, ndvi, lst, tmean | nlcd, landcover, ndvi, lst, tmean | 36.786 | 37.234 | 0.449 | 0.359 | 0.343 | -0.016 |
| Prithvi 100M + Consistency | nlcd, ndvi, lst, tmean | nlcd, landcover, ndvi, lst, tmean | 37.109 | 36.671 | -0.437 | 0.348 | 0.363 | 0.015 |
