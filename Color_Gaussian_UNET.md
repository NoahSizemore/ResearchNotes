# EXP-003A: CIFAR-10 Gaussian UNET Implementing β-NLL

**Date:** 2026-05-11
**Author:** Noah Sizemore
**Status:** In Progress

## Goal
The goal for this experiment is to adapt previous experiment to use Gaussian UNET. In this test, the Gaussian UNET implements β-NLL (β = 0.5) training objective based on results from [other literature.](https://iclr.cc/virtual/2022/poster/6755) Results from this experiement are expanding upon the [deterministic UNET](https://github.com/FOURM-LAB/CrispSynth/blob/main/notes/experiments/exp001_deterministic_unet_colorization.md) (in the sense that the UNET is being adapted) and [deep ensemble](https://github.com/FOURM-LAB/CrispSynth/blob/main/notes/experiments/exp002_deep_ensemble_colorization.md) (the experiment will report on uncertainty metrics AUSE, ENCE, and, now, NLL) experiments.

## Setup
- **Model:** Four-layer deep UNET implementing batch normalization; [compute_uncertainty.py](https://github.com/NoahSizemore/ResearchProjects/blob/main/compute_uncertainty.py) for computing AUSE and ECE (for my testing, specifically ENCE).
- **Dataset:** CIFAR-10
- **Method:** Gaussian UNET with β-NLL
- **Key hyperparameters:** LR = 2e-4; Batch size = 128; epochs = 9; seeds = 42
- **Loss:** MAE, PSNR, SSIM, LPIPS
- **GPU:** A30
- **Training time:** UNET training: 3 minutes; Total time to determine uncertainty: 1 minute
- **Commit:** `ab18c68993efcd8dd914fd6e67e2ec2c15f37750`
- **W&B run:** [HuggingFace](https://huggingface.co/datasets/NoahSizemore/Gaussian_UNET/tree/main)

## Results

| Seed | TRAIN &darr; | TEST &darr; | PSNR &uarr; | SSIM &uarr; | LPIPS &darr; | Alea. AUSE &darr; | Epis. AUSE &darr; | Tot. AUSE &darr; | Alea. ENCE &darr; | Epis. ENCE &darr; | Tot. ENCE &darr; | L1 Mu       |
|------|--------------|-------------|-------------|-------------|--------------|-------------------|-------------------|------------------|-------------------|-------------------|------------------|-------------|
| 42   | -0.1212      | -0.1198     | 22.8486     | 0.9358      | 0.2000       | 0.3250            | 0.8354            | 0.3250           | 0.0751            | 4704976.0445      | 0.0751           | 0.0320      |

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

For ENCE, every bin has a predicted variance of essentially zero (numerical noise from floating-point ties might yield bin centers around `10^{-12}`), so so `rmv_bin ≈ 1e-6` for every bin. The empirical RMSE per bin is approximately $0.1$. The normalized gap is roughly $0.1 / 10^{-6} = 10^5$ per bin, averaged across $10$ bins. This results in the massive magnitude of 4.7 million seen in the results. This correlates with the epistemic value observed in the deep ensemble testing, where the ENCE epistemic value was 11.26. Overall, similar to the deep ensemble, ENCE evaluated on epistemic variance is not a meaningful metric for any ensemble on in-distribution data.

The total variance value is to be expected, as matching the aleatoric value is the anticipated result. Since $\text{total} = \text{aleatoric} + \text{epistemic}$, and the epistemic variance is exactly zero, the total is exactly equal to the aleatoric variance.

L1 Mu is the MAE loss which is a normal value which is to be expected from the model.

## Next Steps

The new model introduces the possibility of discovering better hyperparameters, meaning additional testing may be required to achieve optimal results. Future testing could include the following adjustments:
*   Lowering the learning rate (LR) while maintaining the current batch size.
*   Training for more epochs.
*   Experimenting with different $\beta$-NLL values.
*   Testing group normalization.
*   Potentially adjust L1 Mu to the standard formula for more accurate results $0.5 * (log(2π) + logvar + (target - mu)² / var)$


# EXP-003B: CIFAR-10 Gaussian UNET Hyperparameter Search

**Date:** 2026-05-12
**Author:** Noah Sizemore
**Status:** Complete

## Goal 
The goal of this experiment is to expand upon the previous experiment prior EXP-003A, and search for potentially better preforming hyperparameters.

## Setup pt.1
- **Model:** Four-layer deep UNET implementing batch normalization; [compute_uncertainty.py](https://github.com/NoahSizemore/ResearchProjects/blob/main/compute_uncertainty.py) for computing AUSE and ECE (for my testing, specifically ENCE).
- **Dataset:** CIFAR-10
- **Method:** Gaussian UNET with β-NLL
- **Key hyperparameters:** LR = 1e-4; Batch size = 128; epochs = 9; seeds = 42
- **Loss:** MAE, PSNR, SSIM, LPIPS
- **GPU:** A30
- **Training time:** UNET training: 3 minutes; Total time to determine uncertainty: 1 minute
- **Commit:** `ab18c68993efcd8dd914fd6e67e2ec2c15f37750`

## Results pt.1

### Adjusting LR from 2e-4 to 1e-4

| Seed | TRAIN &darr; | TEST &darr; | PSNR &uarr; | SSIM &uarr; | LPIPS &darr; | Alea. AUSE &darr; | Epis. AUSE &darr; | Tot. AUSE &darr; | Alea. ENCE &darr; | Epis. ENCE &darr; | Tot. ENCE &darr; | L1 Mu       |
|------|--------------|-------------|-------------|-------------|--------------|-------------------|-------------------|------------------|-------------------|-------------------|------------------|-------------|
| 42   | -0.1911      | -0.1896     | 22.6139     | 0.9309      | 0.2132       | 0.4825            | 0.8403            | 0.4825           | 0.4270            | 4845239.8926      | 0.4270           | 0.0329      |


## Observations pt.1
The new learning rate did not improve the models performance across any metric. All of the following were, even minimally worse than EXP-003A. Moving forward, not adjusting the LR ro 1-e4 was not productive.

## Next Steps pt.1

he new model introduces the possibility of discovering better hyperparameters, meaning additional testing may be required to achieve optimal results. Future testing could include the following adjustments:
*   Training for more epochs.
*   Experimenting with different $\beta$-NLL values.
*   Testing group normalization.
*   Potentially adjust L1 Mu to the standard formula for more accurate results $0.5 * (log(2π) + logvar + (target - mu)² / var)$

## Setup pt.2
- **Model:** Four-layer deep UNET implementing batch normalization; [compute_uncertainty.py](https://github.com/NoahSizemore/ResearchProjects/blob/main/compute_uncertainty.py) for computing AUSE and ECE (for my testing, specifically ENCE).
- **Dataset:** CIFAR-10
- **Method:** Gaussian UNET with β-NLL
- **Key hyperparameters:** LR = 2e-4; Batch size = 128; epochs = 20; seeds = 42
- **Loss:** MAE, PSNR, SSIM, LPIPS
- **GPU:** A30
- **Training time:** UNET training: 3 minutes; Total time to determine uncertainty: 1 minute
- **Commit:** `ab18c68993efcd8dd914fd6e67e2ec2c15f37750`

## Results pt.2

### Training for 20 epochs

| EPOCH| TRAIN &darr; | TEST &darr; | PSNR &uarr; | SSIM &uarr; | LPIPS &darr; | Alea. AUSE &darr; | Epis. AUSE &darr; | Tot. AUSE &darr; | Alea. ENCE &darr; | Epis. ENCE &darr; | Tot. ENCE &darr; | L1 Mu       |
|------|--------------|-------------|-------------|-------------|--------------|-------------------|-------------------|------------------|-------------------|-------------------|------------------|-------------|
| 09   | -0.1911      | -0.1896     | 22.6139     | 0.9309      | 0.2132       | 0.4825            | 0.8403            | 0.4825           | 0.4270            | 4845239.8926      | 0.4270           | 0.0329      |
| 18   | -0.1113      | -0.1045     | 22.9930     | 0.9374      | 0.1909       | 0.3330            | 0.8394            | 0.3330           | 0.1193            | 4635813.0499      | 0.1193           | 0.0314      |
| 20   | -0.1104      | -0.1026     | 22.9505     | 0.9371      | 0.1916       | 0.3493            | 0.8479            | 0.3493           | 0.1573            | 4661291.5396      | 0.1573           | 0.0315      |


## Observations pt.2
Testing over more epochs proved to be valuable. Epoch 18 proved to be better overall compared to previous testing. Epochs above 18 showed minor overfitting (e.g., the training/validation gap was above 5%). Moving forward, the new hyperparameters for the model will include training for 18 epochs rather than 9.

Interstingly, the models preformance increased when double the previous epoch. New question being asked: Would training for 27, 36, or other epochs in increments of nine improve the model?

 
## Next Steps pt.2

he new model introduces the possibility of discovering better hyperparameters, meaning additional testing may be required to achieve optimal results. Future testing could include the following adjustments:
*   Experimenting with epochs 27 and 36
*   Experimenting with different $\beta$-NLL values.
*   Testing group normalization.
*   Potentially adjust L1 Mu to the standard formula for more accurate results $0.5 * (log(2π) + logvar + (target - mu)² / var)$
