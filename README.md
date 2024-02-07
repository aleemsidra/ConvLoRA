## Overview
- **CONVLORA AND ADABN BASED DOMAIN ADAPTATION VIA SELF-TRAINING**
The code repository for paper "ConvLORA and ADABN based DOMAIN ADAPTATION via SELF-TRAINING" accepted at [IEEE ISBI 2024](https://biomedicalimaging.org/2024/) in PyTorch.


<p align="center"><img width="80%" src="/home/sidra/Documents/ConvLoRA/imgs/uda_arch.png" /></p>

## Results

<p align="center"><img width="80%" src="/home/sidra/Documents/ConvLoRA/imgs/results.png" /></p>

## Dataset
## [CC359 ](https://www.ccdataset.com/home)
Calgary-Campinas (CC359) dataset IS a multi-vendor (GE, Philips, Siemens), multi-field strength (1 5, 3) magnetic resonance (MR) T1-weighted volumetric brain imaging dataset. It has six different domains and contains 359 3D brain MR image volumes, primarily focused on the task of skull stripping. 


## Arguments

Following arguments are required to run the code. The details are in <main.py>

Task Related Arguments

<dataset:> Option for the dataset, default to CC359
<site:> Site in CC359 dataset
<step:> Specifies stage of adaptation pipeline (base_model, refine, adapt)
<seed:> Seed value for reproducibility


## Training scripts 
Training base model

```
python main.py --config ./config/baseline.json --data "cc359" --site 2 --step "base_model" --seed 1234 --wandb_mode "online" --suffix <"user defined">  

```

Training ESH model

```
python main.py --config ./config/feature_seg.json --data "cc359" --site 2 --step "feature_segmentor"  --seed 1234  --wandb_mode "online"  --suffix <"user defined"> 
```

Adaptation


```
python main.py --config ./config/refinment.json --data "cc359" --site 3  --step "adapt"  --seed 1234  --wandb_mode "online"  --suffix <"user defined">

```

## Contact
Feel free to raise an issue or contact me at sidra.aleem2@mail.dcu.ie for queries and discussions.
