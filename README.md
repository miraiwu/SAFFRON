# SAFFRON



## 1.Download Pre-trained Models
Download the checkpoint from Zenodo (https://zenodo.org/records/16924473)
 and place it under the checkpoint/ folder:

```bash
checkpoint/saffron_GZ_GFY.pt

```

## 2.Create Environment

It is recommended to use **conda**. Example:

```bash
conda create -n saffron python=3.10
conda activate saffron
```
Build CUDA 12.4 (recommended)
```bash
python -m pip install -U pip
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
```

Install required dependencies:
```bash
pip install -r requirements.txt
```
Install the package itself:
```bash
pip install -e . --no-deps --no-build-isolation --config-settings editable_mode=compat
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
