# EXP-003: CIFAR-10 Gaussian UNET 

**Date:** 2026-05-13
**Author:** Noah Sizemore
**Status:** Complete

## Goal
The goal of all of the following is to implement a Guassian UNET and find the best hyperparameters for the model.

## W&Bs
- **W&B run (FINAL):** [HuggingFace](https://huggingface.co/datasets/NoahSizemore/Gaussian_UNET_searchparam/tree/main)

---

# EXP-003A: CIFAR-10 Gaussian UNET Implementing β-NLL

**Date:** 2026-05-11
**Author:** Noah Sizemore
**Status:** Complete

## Goal
The goal for this experiment is to adapt the previous experiment to use Gaussian UNET. In this test, the Gaussian UNET implements β-NLL (β = 0.5) training objective based on results from [other literature.](https://iclr.cc/virtual/2022/poster/6755) Results from this experiment are expanding upon the [deterministic UNET](https://github.com/FOURM-LAB/CrispSynth/blob/main/notes/experiments/exp001_deterministic_unet_colorization.md) (in the sense that the UNET is being adapted) and [deep ensemble](https://github.com/FOURM-LAB/CrispSynth/blob/main/notes/experiments/exp002_deep_ensemble_colorization.md) (the experiment will report on uncertainty metrics AUSE, ENCE, and, now, NLL) experiments.

## Setup
- **Model:** Four-layer deep UNET implementing batch normalization; [compute_uncertainty.py](https://github.com/NoahSizemore/ResearchProjects/blob/main/compute_uncertainty.py) for computing AUSE and ECE (for my testing, specifically ENCE).
- **Dataset:** CIFAR-10
- **Method:** Gaussian UNET with β-NLL
- **Key hyperparameters:** LR = 2e-4; Batch size = 128; epochs = 9; seeds = 42
- **Loss:** MAE, PSNR, SSIM, LPIPS
- **GPU:** A30
- **Training time:** UNET training: 3 minutes; Total time to determine uncertainty: 1 minute
- **Commit:** `ab18c6893efcd8dd914fd6e7e2ec2c15f37750`

## Results

| Seed | TRAIN ↓ | TEST ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Alea. AUSE ↓ | Epis. AUSE ↓ | Tot. AUSE ↓ | Alea. ENCE ↓ | Epis. ENCE ↓ | Tot. ENCE ↓ | L1 Mu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | -0.1212 | -0.1198 | 22.8486 | 0.9358 | 0.2000 | 0.3250 | 0.8354 | 0.3250 | 0.0751 | 4704976.0445 | 0.0751 | 0.0320 |

See [here](https://github.com/FOURM-LAB/CrispSynth/blob/main/notes/experiments/images/gaussian_cifar/gnll_ause_results.json) for the `.json` file with results from uncertainty testing.

## Key Figures

### Reliability:

![Aleatoric reliability figure](https://github.com/FOURM-LAB/CrispSynth/blob/main/notes/experiments/images/gaussian_cifar/gnll_reliability_aleatoric.png)
![Epistemic reliability figure](https://github.com/FOURM-LAB/CrispSynth/blob/main/notes/experiments/images/gaussian_cifar/gnll_reliability_epistemic.png)
![Total reliability figure](https://github.com/FOURM-LAB/CrispSynth/blob/main/notes/experiments/images/gaussian_cifar/gnll_reliability_total.png)

### Sparsification:

![Aleatoric sparsification figure](https://github.com/FOURM-LAB/CrispSynth/blob/main/notes/experiments/images/gaussian_cifar/gnll_sparsification_aleatoric.png)
![Epistemic sparsification figure](https://github.com/FOURM-LAB/CrispSynth/blob/main/notes/experiments/images/gaussian_cifar/gnll_sparsification_epistemic.png)
![Total sparsification figure](https://github.com/FOURM-LAB/CrispSynth/blob/main/notes/experiments/images/gaussian_cifar/gnll_sparsification_total.png)

## Observations

The results are generally positive, demonstrating that the Gaussian U-Net with $\beta = 0.5$ is performing as intended. The training and testing loss gap is 0.0014, indicating no overall overfitting. The PSNR, SSIM, and LPIPS are relatively similar to those of the deterministic U-Net model. This is neither positive nor negative; however, it is encouraging that the model behaves consistently with previous experiments.

Both the AUSE and ENCE aleatoric values yield meaningful results. The AUSE aleatoric score is better than the deep ensemble value, although not by a steep margin. An AUSE of 0.3250 indicates that the variance head's spatial ranking of uncertainty is moderately informative for triage, but not strongly so. Pixels labeled "high uncertainty" are, on average, more error-prone than random pixels; however, the correlation is weak enough that removing the top-uncertainty pixels only gradually reduces the surviving error. The ENCE aleatoric score, in comparison to the AUSE aleatoric score, is a much stronger metric. An ENCE of 0.0751 means the model is making sharper variance predictions that happen to be well-calibrated, which the ENCE metric rewards. This is the highlight of the testing, with the ENCE aleatoric score representing a highly desirable value.

Both the AUSE and ENCE epistemic values may initially appear concerning; however, there is more nuance to these values than is immediately apparent.

For AUSE, an all-zeros uncertainty array means the `argsort` tie-breaking determines the ranking arbitrarily (essentially randomly). An AUSE of 0.8354 is what a "random ranking of pixel uncertainty, integrated against the oracle" produces on this error distribution. It is roughly twice the random-uncertainty baseline because of the specific way zero-variance ties are sorted: NumPy's `argsort` is stable and returns indices in their original order. Consequently, the predicted-uncertainty curve removes pixels in an image-then-channel-then-spatial-position order. On natural images, this correlates positively with error (early pixels in row-major order tend to be sky or background pixels with low error, so removing them sequentially causes the surviving error to grow). The result is a curve that climbs above 1.0 in the mid-range, producing a huge gap with the oracle. Therefore, this number is not a meaningful measure of uncertainty quality; rather, it is an artifact of how zero variance interacts with the sparsification computation.

For ENCE, every bin has a predicted variance of essentially zero (numerical noise from floating-point ties might yield bin centers around `10^{-12}`), so `rmv_bin ≈ 1e-6` for every bin. The empirical RMSE per bin is approximately $0.1$. The normalized gap is roughly $0.1 / 10^{-6} = 10^5$ per bin, averaged across $10$ bins. This results in the massive magnitude of 4.7 million seen in the results. This correlates with the epistemic value observed in the deep ensemble testing, where the ENCE epistemic value was 11.26. Overall, similar to the deep ensemble, ENCE evaluated on epistemic variance is not a meaningful metric for any ensemble on in-distribution data.

The total variance value is to be expected, as matching the aleatoric value is the anticipated result. Since $\text{total} = \text{aleatoric} + \text{epistemic}$, and the epistemic variance is exactly zero, the total is exactly equal to the aleatoric variance.

L1 Mu is the MAE loss which is a normal value which is to be expected from the model.

## Next Steps

The new model introduces the possibility of discovering better hyperparameters, meaning additional testing may be required to achieve optimal results. Future testing could include the following adjustments:
* Lowering the learning rate (LR) while maintaining the current batch size.
* Training for more epochs.
* Experimenting with different $\beta$-NLL values.
* Testing group normalization.
* Potentially adjust L1 Mu to the standard formula for more accurate results $0.5 * (log(2π) + logvar + (target - mu)² / var)$

---

# EXP-003B: CIFAR-10 Gaussian UNET Hyperparameter Search: LR

**Date:** 2026-05-12
**Author:** Noah Sizemore
**Status:** Complete

## Goal
The goal of this experiment is to expand upon the previous experiment prior EXP-003A, and search for potentially better performing hyperparameters.

## Setup pt.1
- **Model:** Four-layer deep UNET implementing batch normalization; [compute_uncertainty.py](https://github.com/NoahSizemore/ResearchProjects/blob/main/compute_uncertainty.py) for computing AUSE and ECE (for my testing, specifically ENCE).
- **Dataset:** CIFAR-10
- **Method:** Gaussian UNET with β-NLL
- **Key hyperparameters:** LR = 1e-4; Batch size = 128; epochs = 9; seeds = 42
- **Loss:** MAE, PSNR, SSIM, LPIPS
- **GPU:** A30
- **Training time:** UNET training: 8 minutes; Total time to determine uncertainty: 1 minute
- **Commit:** `ab18c6893efcd8dd914fd6e7e2ec2c15f37750`

## Results pt.1

### Adjusting LR from 2e-4 to 1e-4:

| Seed | TRAIN ↓ | TEST ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Alea. AUSE ↓ | Epis. AUSE ↓ | Tot. AUSE ↓ | Alea. ENCE ↓ | Epis. ENCE ↓ | Tot. ENCE ↓ | L1 Mu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | -0.1911 | -0.1896 | 22.6139 | 0.9309 | 0.2132 | 0.4825 | 0.8403 | 0.4825 | 0.4270 | 4845239.8926 | 0.4270 | 0.0329 |

## Observations pt.1
The new learning rate did not improve the models performance across any metric. All of the following were, even minimally worse than EXP-003A. Moving forward, not adjusting the LR ro 1-e4 was not productive.

## Next Steps pt.1

The new model introduces the possibility of discovering better hyperparameters, meaning additional testing may be required to achieve optimal results. Future testing could include the following adjustments:
* Training for more epochs.
* Experimenting with different $\beta$-NLL values.
* Testing group normalization.
* Potentially adjust L1 Mu to the standard formula for more accurate results $0.5 * (log(2π) + logvar + (target - mu)² / var)$

---

# EXP-003C: CIFAR-10 Gaussian UNET Hyperparameter Search: More Epochs

**Date:** 2026-05-12
**Author:** Noah Sizemore
**Status:** Complete

## Setup pt.2
- **Model:** Four-layer deep UNET implementing batch normalization; [compute_uncertainty.py](https://github.com/NoahSizemore/ResearchProjects/blob/main/compute_uncertainty.py) for computing AUSE and ECE (for my testing, specifically ENCE).
- **Dataset:** CIFAR-10
- **Method:** Gaussian UNET with β-NLL
- **Key hyperparameters:** LR = 2e-4; Batch size = 128; epochs = 20; seeds = 42
- **Loss:** MAE, PSNR, SSIM, LPIPS
- **GPU:** A30
- **Training time:** UNET training: 3 minutes; Total time to determine uncertainty: 1 minute
- **Commit:** `ab18c6893efcd8dd914fd6e7e2ec2c15f37750`

## Results pt.2

### Training for 20 epochs:

| EPOCH | TRAIN ↓ | TEST ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Alea. AUSE ↓ | Epis. AUSE ↓ | Tot. AUSE ↓ | Alea. ENCE ↓ | Epis. ENCE ↓ | Tot. ENCE ↓ | L1 Mu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 09 | -0.1911 | -0.1896 | 22.6139 | 0.9309 | 0.2132 | 0.4825 | 0.8403 | 0.4825 | 0.4270 | 4845239.8926 | 0.4270 | 0.0329 |
| 18 | -0.1113 | -0.1045 | 22.9930 | 0.9374 | 0.1909 | 0.3330 | 0.8394 | 0.3330 | 0.1193 | 4635813.0499 | 0.1193 | 0.0314 |
| 20 | -0.1104 | -0.1026 | 22.9505 | 0.9371 | 0.1916 | 0.3493 | 0.8479 | 0.3493 | 0.1573 | 4661291.5396 | 0.1573 | 0.0315 |

## Observations pt.2
Testing over more epochs proved to be valuable. Epoch 18 proved to be better overall compared to previous testing. Epochs above 18 showed minor overfitting (e.g., the training/validation gap was above 5%). Moving forward, the new hyperparameters for the model will include training for 18 epochs rather than 9.

Interstingly, the models preformance increased when double the previous epoch. New question being asked: Would training for 27, 36, or other epochs in increments of nine improve the model?

## Next Steps pt.2

The new model introduces the possibility of discovering better hyperparameters, meaning additional testing may be required to achieve optimal results. Future testing could include the following adjustments:
* Experimenting with epochs 27 and 36
* Experimenting with different $\beta$-NLL values.
* Testing group normalization.
* Potentially adjust L1 Mu to the standard formula for more accurate results $0.5 * (log(2π) + logvar + (target - mu)² / var)$

---

# EXP-003D: CIFAR-10 Gaussian UNET Hyperparameter Search: Even More Epochs

**Date:** 2026-05-13
**Author:** Noah Sizemore
**Status:** Complete

## Setup pt.3
- **Model:** Four-layer deep UNET implementing batch normalization; [compute_uncertainty.py](https://github.com/NoahSizemore/ResearchProjects/blob/main/compute_uncertainty.py) for computing AUSE and ECE (for my testing, specifically ENCE).
- **Dataset:** CIFAR-10
- **Method:** Gaussian UNET with β-NLL
- **Key hyperparameters:** LR = 2e-4; Batch size = 128; epochs = 36; seeds = 42
- **Loss:** MAE, PSNR, SSIM, LPIPS
- **GPU:** A30
- **Training time:** UNET training: 13 minutes; Total time to determine uncertainty: 1 minute
- **Commit:** `ab18c6893efcd8dd914fd6e7e2ec2c15f37750`

## Results pt.3

### Training for 36 epochs:

| EPOCH | TRAIN ↓ | TEST ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Alea. AUSE ↓ | Epis. AUSE ↓ | Tot. AUSE ↓ | Alea. ENCE ↓ | Epis. ENCE ↓ | Tot. ENCE ↓ | L1 Mu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | -0.1911 | -0.1896 | 22.6139 | 0.9309 | 0.2132 | 0.4825 | 0.8403 | 0.4825 | 0.4270 | 4845239.8926 | 0.4270 | 0.0329 |
| 18 | -0.1113 | -0.1045 | 22.9930 | 0.9374 | 0.1909 | 0.3330 | 0.8394 | 0.3330 | 0.1193 | 4635813.0499 | 0.1193 | 0.0314 |
| 20 | -0.1104 | -0.1026 | 22.9505 | 0.9371 | 0.1916 | 0.3493 | 0.8479 | 0.3493 | 0.1573 | 4661291.5396 | 0.1573 | 0.0315 |
| 36 | -0.1010 | -0.0787 | 22.7387 | 0.9362 | 0.1934 | Null | Null | Null | Null | Null | Null | 0.0325 |

## Observations pt.3
After running the model for 36 epochs, results point to poor performance. The training/validation gap well exceeds 5%, meaning the model is overfitting. From testing, the model had signs of overfitting at epoch 21.

## Next Steps pt.3

The new model introduces the possibility of discovering better hyperparameters, meaning additional testing may be required to achieve optimal results. Future testing could include the following adjustments:
* Experimenting with different $\beta$-NLL values.
* Testing group normalization.
* Potentially adjust L1 Mu to the standard formula for more accurate results $0.5 * (log(2π) + logvar + (target - mu)² / var)$

---

# EXP-003E: CIFAR-10 Gaussian UNET Hyperparameter Search: Group Normalization

**Date:** 2026-05-13
**Author:** Noah Sizemore
**Status:** Complete

## Setup pt.4
- **Model:** Four-layer deep UNET implementing group normalization (groups 16); [compute_uncertainty.py](https://github.com/NoahSizemore/ResearchProjects/blob/main/compute_uncertainty.py) for computing AUSE and ECE (for my testing, specifically ENCE).
- **Dataset:** CIFAR-10
- **Method:** Gaussian UNET with β-NLL
- **Key hyperparameters:** LR = 2e-4; Batch size = 128; epochs = 36; seeds = 42; Group normalization (groups 16)
- **Loss:** MAE, PSNR, SSIM, LPIPS
- **GPU:** A30
- **Training time:** UNET training: 9 minutes; Total time to determine uncertainty: 1 minute
- **Commit:** `ab18c6893efcd8dd914fd6e7e2ec2c15f37750`

## Results pt.4

### Using UNET with group normalization (groups 16):

| TYPE | EPOCH | TRAIN ↓ | TEST ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Alea. AUSE ↓ | Epis. AUSE ↓ | Tot. AUSE ↓ | Alea. ENCE ↓ | Epis. ENCE ↓ | Tot. ENCE ↓ | L1 Mu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Batch | 9 | -0.1212 | -0.1198 | 22.8486 | 0.9358 | 0.2000 | 0.3250 | 0.8354 | 0.3250 | 0.0751 | 4704976.0445 | 0.0751 | 0.0320 |
| Batch | 18 | -0.1113 | -0.1045 | 22.9930 | 0.9374 | 0.1909 | 0.3330 | 0.8394 | 0.3330 | 0.1193 | 4635813.0499 | 0.1193 | 0.0314 |
| Group | 18 | -0.1158 | -0.1146 | 23.0105 | 0.9379 | 0.1918 | 0.3251 | 0.8291 | 0.3251 | 0.0748 | 4612828.0460 | 0.0748 | 0.0314 |
| Group | 21 | -0.1144 | -0.1151 | 23.0570 | 0.9388 | 0.1888 | 0.3262 | 0.8287 | 0.3262 | 0.0825 | 4596993.5879 | 0.0825 | 0.0312 |

See [here](https://github.com/FOURM-LAB/CrispSynth/blob/main/notes/experiments/images/gaussian_cifar_improvements/gnll_ause_results.json) for the `.json` file with results from uncertainty testing.

## Key Figures

### Reliability:

![Aleatoric reliability figure](https://github.com/FOURM-LAB/CrispSynth/blob/main/notes/experiments/images/gaussian_cifar_improvements/gnll_reliability_aleatoric.png)
![Epistemic reliability figure](https://github.com/FOURM-LAB/CrispSynth/blob/main/notes/experiments/images/gaussian_cifar_improvements/gnll_reliability_epistemic.png)
![Total reliability figure](https://github.com/FOURM-LAB/CrispSynth/blob/main/notes/experiments/images/gaussian_cifar_improvements/gnll_reliability_total.png)

### Sparsification:

![Aleatoric sparsification figure](https://github.com/FOURM-LAB/CrispSynth/blob/main/notes/experiments/images/gaussian_cifar_improvements/gnll_sparsification_aleatoric.png)
![Epistemic sparsification figure](https://github.com/FOURM-LAB/CrispSynth/blob/main/notes/experiments/images/gaussian_cifar_improvements/gnll_sparsification_epistemic.png)
![Total sparsification figure](https://github.com/FOURM-LAB/CrispSynth/blob/main/notes/experiments/images/gaussian_cifar_improvements/gnll_sparsification_total.png)

## Observations pt.4
The testing from this experiment proved to be very impactful. Group normalization (groups of 16) has improved the models preformance across most metrics. In spite of findings from the last experiment, the model preformed better past epoch 18 using group normalization. Some note worthy numbers, the training/validation gap for epoch 21 is ***-0.61%***, which is much better than other experiments have had. Additionaly, PSNR, LPIPS, and L1 Mu all had improved values compared to all other testing. In terms of uncertainty, the model performed slightly worse than epoch 18, however changes are very minoot and overall less impactful than the other improvements mentioned. In spite of all of the following, testing to confirm findings using different seeds may be needed to validate actual improvement as described.

## Next Steps pt.4

The new model introduces the possibility of discovering better hyperparameters, meaning additional testing may be required to achieve optimal results. Future testing could include the following adjustments:
* Experimenting with different $\beta$-NLL values.
* Potentially adjust L1 Mu to the standard formula for more accurate results $0.5 * (log(2π) + logvar + (target - mu)² / var)$
