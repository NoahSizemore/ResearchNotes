# EXP-01: Creating and Implementing Deep Ensemble
**Date:** 2026-05-04
**Author:** Noah Sizemore
**Status:** In Progress

## Goal

The goal for the following is to implement the necessary meterics for deep ensemble based on the previous experiments in [Color_CIFAR](). I will be using the determined best hyperparameters from those experiements in this one as well. The testing will be done on five seeds: 42, 43, 44, 45, 46, and 47. 

## Setup

- **Model:** Four layer deep U-Net implementing batch normalization
- **Dataset:** CIFAR-10
- **Method:** Deterministic and ensemble
- **Key hyperparameters:** LR = 2e-4; Batch size = 128; epochs = 9; seeds = 42, 43, 44, 45, 46, and 47
- **Loss:** MAE and MSE
- **GPU:** A30
- **Training time:** [approximate wall-clock time]
- **Commit:** `[hash]`
- **Config file:** `configs/[filename].yaml`
- **W&B run:** [link if applicable]

## Results

[Paste tables, key numbers, or reference figures]

| Metric | Value |
|--------|-------|
| PSNR   |       |
| SSIM   |       |
| ECE    |       |
| AUSE   |       |

## Key Figures

![Description](figures/expNNN_figure_name.png)

## Observations

What did you learn? What was surprising? What didn't work as expected?

## Next Steps

What should be tried next based on these results?
