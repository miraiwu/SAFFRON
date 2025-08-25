# SAFFRON
This is the SAFFRON



## 1.Download Pre-trained Models
Download the checkpoint from Zenodo (https://zenodo.org/records/16924473)
 and place it under the checkpoint/ folder:

```bash
checkpoint/saffron_GZ_GFY.pt

```

## 2.Prepare Environment

It is recommended to use **conda**. Example:

```bash
conda create -n saffron python=3.10
conda activate saffron
```

Install the package itself:
```bash
pip install -e .
```

## 3.Prepare Data Directory
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

## 4. brain reconstruction

run python saffron_clinical_recon.py
