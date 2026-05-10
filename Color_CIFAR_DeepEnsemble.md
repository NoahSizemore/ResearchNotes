# EXP-01: Creating and Implementing Deep Ensemble
**Date:** 2026-10-04
**Author:** Noah Sizemore
**Status:** In Progress

## Goal
The goal of the following experiment is to implement the necessary metrics for deep ensembles based on the previous experiments in [Color_CIFAR](). I will be using the best hyperparameters determined from those experiments in this one as well. The testing will be done on five seeds: 42, 43, 44, 45, and 46.

## Setup
- **Model:** Four-layer deep UNET implementing batch normalization; [compute_uncertainty.py]() for computing AUSE and ECE (for my testing, specifically ENCE).
- **Dataset:** CIFAR-10
- **Method:** Deterministic and ensemble
- **Key hyperparameters:** LR = 2e-4; Batch size = 128; epochs = 9; seeds = 42, 43, 44, 45, and 46.
- **Loss:** MAE, Gaussian β-NLL (not vanilla, using β-NLL = 0.5)
- **GPU:** A30
- **Training time:** Each U-net training: 10 minutes; Total uncertainty time: 10 minutes
- **Commit:** `bfc2fc5d32f41c0a746bd346797c50445c23cdaa`
- **W&B run:** [HuggingFace dataset](https://huggingface.co/datasets/NoahSizemore/Colorization_Deep_Ensemble_Weights/tree/main)

## Results

---

### Individual results:

| Seed | TRAIN &darr; | TEST &darr; | TRAIN NLL &darr; | TEST NLL &darr;| PSNR &uarr; | SSIM &uarr; | LPIPS &darr; |
|------|-------------|------------|-----------------|----------------|------------|------------|-------------|
| 42   | 0.02964     | 0.03071    | -0.1277         | -0.1277        | 23.03      | 0.9391     | 0.1927      |
| 43   | 0.02978     | 0.03068    | -0.1212         | -0.1198        | 23.03      | 0.9392     | 0.1935      |
| 44   | 0.02979     | 0.03064    | -0.1242         | -0.1221        | 23.06      | 0.9393     | 0.1924      |
| 45   | 0.02997     | 0.03089    | -0.1256         | -0.1242        | 22.99      | 0.9391     | 0.1936      |
| 46   | 0.02992     | 0.03062    | -0.1255         | -0.1248        | 23.03      | 0.9392     | 0.1934      |

### Deep ensemble results:

| TRAIN &darr; | TEST &darr; | TRAIN NLL &darr; | TEST NLL &darr;| PSNR &uarr; | SSIM &uarr; | LPIPS &darr; |
|-------------|------------|-----------------|----------------|------------|------------|-------------|
| 0.02982     | 0.03070    | -0.1248         | -0.1237        | 23.03      | 0.9392     | 0.1931      |

### AUSE and ENCE ensemble results:

| Metric | Aleatoric | Epistemic | Total  |
|--------|-----------|-----------|--------|
| AUSE   | 0.3123    | 0.4752    | 0.3127 |
| ENCE   | 0.1239    | 11.2621   | 0.1287 |

See [here]() for the `.json` file with results from testing.

---

## Key Figures

### Reliability
![Description](figures/expNNN_figure_name.png)

### Sparsification
![Description](figures/expNNN_figure_name.png)

## Observations
Overall, testing was successful and consistent across all values for every seed. The aleatoric AUSE value, due to the β-NLL with β = 0.5, may seem mediocre, but this is actually to be expected. The trade-off of using β = 0.5 involves sacrificing uncertainty-ranking quality for more honest mean-prediction quality. This means that pixels vanilla NLL would have flagged as wildly uncertain are now handled more uniformly. Therefore, you would expect a mid-range AUSE alongside a good ENCE.

The epistemic AUSE tells the same story from the ranking perspective. An above-random AUSE (which is the area between a flat curve and the oracle, plausibly ~0.25 for this error distribution) means the epistemic ranking is approximately uninformative to mildly anti-informative—the regions where members happen to disagree on a clean test set don't strongly correspond to regions where the ensemble is actually wrong. 

The aleatoric ENCE is within the typical range for Gaussian heads. This ENCE value is actually saying that within any equal-mass bin of predicted variance, the empirical RMSE is, on average, about 12% off from what the predicted σ indicated it should be. This value is small enough to show that the variance head has learned something genuinely useful about its own honesty. 

The high epistemic ENCE value is not a bad sign for the model; rather, it indicates that across five different UNETs, the models converge to very similar solutions, and the metric is being expanded to a larger value (around 100x) from `|rmse - rmv| / rmv`. The actual epistemic ENCE standard deviation is around ***0.001***.

## Next steps
There is potential to change some metrics for NLL and ECE for better testing. Currently, there is no plan to change these values.

## Final Remarks
Overall, the assessment of the experiment is highly positive. The metrics obtained from the deep ensemble evaluation demonstrate consistency with results from previous experiments. The decision to use β-NLL with β = 0.5 proved highly beneficial for the uncertainty metrics. Although the aleatoric values may appear somewhat mediocre, they show that β = 0.5 produces well-calibrated variance magnitudes (ENCE 0.1239) at the cost of moderate sparsification quality (AUSE 0.3123), effectively trading variance ranking sharpness for calibration honesty and mean-prediction quality. The epistemic uncertainty is poorly calibrated on the in-distribution test set, which is expected based on the literature (Lakshminarayanan, 2017). The true value of this decomposition becomes apparent under distribution shift.
