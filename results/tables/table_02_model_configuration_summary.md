| model | family | features | pretraining | consistency_weight | consistency_years | train_stride | eval_stride | train_patches | augmentation | tta_inference | calibration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RF | Random Forest | nlcd, landcover, ndvi, lst, tmean | none | 0.000 | n/a | n/a | n/a | sample_fraction=0.05 | n/a | no | no |
| U-Net | U-Net | nlcd, landcover, ndvi, lst, tmean | none | 0.000 | 2015, 2021 | 64 | 64 | 876 | True | yes | yes |
| U-Net + Consistency | U-Net | nlcd, landcover, ndvi, lst, tmean | none | 0.200 | 2015-2022 | 64 | 64 | 876 | True | yes | yes |
| Prithvi 100M | Prithvi Geo-ViT | nlcd, landcover, ndvi, lst, tmean | Prithvi 100M | 0.000 | 2015-2022 | 64 | 64 | 876 | True | yes | yes |
| Prithvi 100M + Consistency | Prithvi Geo-ViT | nlcd, landcover, ndvi, lst, tmean | Prithvi 100M | 0.200 | 2015-2022 | 64 | 64 | 876 | True | yes | yes |
| Prithvi 300M | Prithvi Geo-ViT | nlcd, landcover, ndvi, lst, tmean | Prithvi 300M | 0.000 | 2015-2022 | 64 | 64 | 876 | True | yes | yes |
| Prithvi 300M + Consistency | Prithvi Geo-ViT | nlcd, landcover, ndvi, lst, tmean | Prithvi 300M | 0.200 | 2015-2022 | 64 | 64 | 876 | True | yes | yes |
