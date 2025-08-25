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
 │         ├── 20**_cor.nii.gz
 │         ├── 20**_sag.nii.gz
 │         ├── 20**_tra.nii.gz
 │         └── seg/
 │              ├── 20**_cor.nii.gz (same file name)
 │              ├── 20**_sag.nii.gz
 │              └── 20**_tra.nii.gz
 ├── saffron_clinical_recon.py
```

## 3. brain reconstruction

run python saffron_clinical_recon.py
