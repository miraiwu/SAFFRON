# SAFFRON
This is the SAFFRON



## 1.Download Pre-trained Models

checkpoint/saffron_GZ_GFY.pt https://zenodo.org/records/16924473


## 2.Prepare Data Directory
Organize your data as follows:
```bash
SAFFRON/
 ├── checkpoint/
 │    └── saffron_GZ_GFY.pt
 ├── data/
 │    └── test/
 │         ├── 20_cor_ssfse.nii.gz
 │         ├── 20_sag_ssfse.nii.gz
 │         ├── 20_tra_ssfse.nii.gz
 │         └── seg/
 │              ├── 20_cor_ssfse.nii.gz (same file name)
 │              ├── 20_sag_ssfse.nii.gz
 │              └── 20_tra_ssfse.nii.gz
 ├── saffron_clinical_recon.py

## 3. brain reconstruction

run python saffron_clinical_recon.py
