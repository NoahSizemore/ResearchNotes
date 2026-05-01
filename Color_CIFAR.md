# EXP-01: Testing Learning Rates

**Date:** 2026-04-24
**Author:** Noah Sizemore
**Status:** Complete

## Goal

The goal of this table is to determine which hyperparameters should be changed to achieve the best possible scores. This is based on adjustments made to the model by adding cosine annealing LR scheduler.

## Setup

- **Model:** U-Net, 4 layers, BatchNorm
- **Dataset:** CIFAR-10
- **Method:** deterministic
- **Key hyperparameters:**  LR = 1e-2, 1e-3, 1e-4, 1e-5; batch size = 128, 256, 512, 1024; seed = 42; epoch = 6
- **Loss:** MAE, PSNR, SSIM, LPIPS
- **GPU:** A30
- **Training time:** 5 minutes per 5 epoch runs (1 epoch a minute)

## Results

The intial model preformance before the scheduler. This model uses LR = 1e-4, batch size = 256, seed = 42:

| Epoch #  | Training | Testing  | PSNR     |
| -------- | -------- | -------- | -------- |
| 2        | 0.0329   | 0.0326   | 26.1427  |
| 10       | 0.0303   | 0.0318   | 26.3487  |
| 100      | 0.0115   | 0.0319   | 27.4832  |

Testing new learning rates for the model. This model uses batch size = 256, seed = 42:

| Hyperparam| Training | Testing  | 
| -------- | -------- | -------- | 
| LR = 1e-2| 0.0320   | 0.0325   | 
| **LR = 1e-3**| **0.0313** | **0.0314** | 
| LR = 1e-4| 0.0312   | 0.0315   | 
| LR = 1e-5| 0.0330   | 0.0332   | 

Using LR of 1e-3 for changes is batch size:

| Hyperparam| Training | Testing  | 
| -------- | -------- | -------- | 
| Batch = 128| 0.0313   | 0.0317   | 
| Batch = 256| 0.0313   | 0.0314   | 
| **Batch = 512**| **0.0316** | **0.0316** |
| Batch = 1024| 0.0320   | 0.0321   |

Combining both best options to determine best overall results from training:

| Epoch #  | Training | Testing  |
| -------- | -------- | -------- | 
| 2        | 0.0330   | 0.0329   | 
| 4        | 0.0321   | 0.0325   | 
| 6        | 0.0316   | 0.0321   | 
| 8        | 0.0311   | 0.0312   |
| 10       | 0.0308   | 0.0310   |
| **11** | **0.0307** | **0.0310** | 

Adding PSNR, SSIM, and LPIPS values:

*Note: SSIM and LPIPS are on a scale of 0 to 1.*
| Epoch #  | &darr; Training |  &darr; Testing  | &uarr; PSNR     | &uarr; SSIM     | &darr; LPIPS    |
| -------- | -------- | -------- | -------- | -------- | -------- |
| 2        | 0.0333   | 0.0342   | 22.4915  | 0.9336   | 0.2186   |
| 5        | 0.0319   | 0.0318   | 22.7617  | 0.9352   | 0.2054   |
| 10       | 0.0308   | 0.0311   | 22.9189  | 0.9380   | 0.1987   |
| 11       | 0.0306   | 0.0309   | 22.9538  | 0.9382   | 0.1977   |
| 24       | 0.0271   | **0.0307** | **23.0250** | 0.9399   | 0.1926   |
| 31       | 0.0236   | 0.0312   | 22.9468  | 0.9398   | **0.1914** |
| 50       | **0.0198** | 0.0321   | 22.7741  | 0.9388   | 0.1940   |

## Observations

After testing several differnt parameters across multiple epochs (with the majority of testing for 6 epochs), these were the findings: Signs of overfitting began at epoch 20. By epoch 22, the training loss had dropped by 0.10, making the loss 0.0302, while the testing loss grew to 0.0320. The overall best epoch after testing was 11 with the following parameters: LR = 1e-3; batch size = 512; seed = 42; epoch = 11. 

Some observed changes: Epoch 24 shows signs of potentially being the best option after having tested the model on previous meterics; however, the model is still overfitting. The best LPIPS score occurred during epoch 31, and, as expected, the best training value was recorded during the final epoch.

## Next Steps

Test other learning rates and batch sizes and confirm values across different seeds.

---

# EXP-02: Testing New Hyperparameters Across Different Seeds

**Date:** 2026-04-29
**Author:** Noah Sizemore
**Status:** Complete

## Goal

The goal of this table is to determine which hyperparameters should be changed to achieve the best possible scores. This is based on adjustments made to the model in EXP-01. I am hoping to achieve the best hyperparameters for the model, and confirm they are the best through different seed testing, to use for further experiments.

## Setup

- **Model:** U-Net, 4 layers, BatchNorm
- **Dataset:** CIFAR-10
- **Method:** deterministic
- **Key hyperparameters:**  LR = 1e-3, 2e-4; batch size = 128, 256, 512; seed = 42, 60; epoch = 6
- **Loss:** MAE, PSNR, SSIM, LPIPS
- **GPU:** A30
- **Training time:** 5 minutes per 5 epoch runs (1 epoch a minute)

## Results

Trying a different learning rate with a smaller batch size: LR = 2e-4, Batch size = 128, seed = 42:

| Epoch #  | &darr; Training |  &darr; Testing  | &uarr; PSNR     | &uarr; SSIM     | &darr; LPIPS    |
| -------- | -------- | -------- | -------- | -------- | -------- |
| 2        | 0.0324   | 0.0324   | 22.5763  | 0.9344   | 0.2126   |
| 4        | 0.0317   | 0.0317   | 22.7054  | 0.9362   | 0.2036   |
| 6        | 0.0311   | 0.0313   | 22.8783  | 0.9371   | 0.1979   |
| **9** | **0.0301** | **0.0308** | **23.0207** | **0.9392** | **0.1940** |
| 11       | 0.0292   | 0.0307   | 23.0571  | 0.9392   | 0.1908   |

New table comparing parameters from EXP-01 to new testing from this experiment:

| Hyperparameters | &darr; Training |  &darr; Testing  | &uarr; PSNR     | &uarr; SSIM     | &darr; LPIPS    |
| -------- | -------- | -------- | -------- | -------- | -------- |
| LR = 2e-4, BS = 128, seed = 42| 0.0311   | 0.0313   | 22.8783  | 0.9371   | 0.1979   |
| LR = 1e-3, BS = 512, seed = 42| 0.0317   | 0.0322   | 22.6270  | 0.9358   | 0.2085   |
| LR = 1e-3, BS = 256, seed = 42| 0.0314   | 0.0314   | 22.8367  | 0.9374   | 0.2002   |
| **LR = 2e-4, BS = 128, seed = 60**| **0.0308**   | **0.0310**   | **22.9205**  | **0.9386**   | **0.1958**   |
| LR = 1e-3, BS = 512, seed = 60| 0.0317   | 0.0321   | 22.6479  | 0.9372   | 0.2063   |
| LR = 1e-3, BS = 256, seed = 60| 0.0314   | 0.0315   | 22.7782  | 0.9379   | 0.2029   |

## Observations

The new hyperparameters performed better, and did so earlier, than the previous parameters, showing greater improvements. Epochs 10 and 11 were the first to show signs of overfitting. This is an overall positive outcome for the testing phase as the model preformed this way acorss multiple seeds.

## Next Steps

Things I potentially want to add in the future to improve the score: a perceptual loss term, switching BatchNorm to GroupNorm (or InstanceNorm).

---

# EXP-03: Perceptual Loss and Group Normalization

**Date:** 2026-04-30
**Author:** Noah Sizemore
**Status:** Complete

## Goal

The goal of this table is to determine which hyperparameters should be changed to achieve the best possible scores. This is based on adjustments made to the model in EXP-01 and EXP-02. I am hoping to achieve the best overall model structure to finalize findings before documentation via group normalization testing and implementing a perceptual loss.

## Setup

- **Model:** U-Net, 4 layers, BatchNorm
- **Dataset:** CIFAR-10
- **Method:** deterministic
- **Key hyperparameters:**  LR = 2e-4; batch size = 128; seed = 42
- **Loss:** MAE, PSNR, SSIM, LPIPS
- **GPU:** A30
- **Training time:** Will specify for each table.

## Results pt. 1

Testing U-Net using GroupNorm in place of BatchNorm. This test uses LR = 2e-4, BS = 128, seed = 42, epoch = 11; time = 5 minutes per 5 epoch runs (1 epoch a minute):
  
| Num of groups | &darr; Training |  &darr; Testing  | &uarr; PSNR     | &uarr; SSIM     | &darr; LPIPS    |
| -------- | -------- | -------- | -------- | -------- | -------- |
| BatchNorm| 0.0292   | 0.0307   | 23.0571  | 0.9392   | 0.1908   |
| 8        | 0.0309   | 0.0311   | 22.9018  | 0.9381   | 0.1994   |
| 16       | 0.0308   | 0.0310   | 22.9320  | 0.9383   | 0.1982   |
| 32       | 0.0306   | 0.0310   | 22.9528  | 0.9382   | 0.1982   |

Group normalization showing signs of potentially improvements, testing groups 16 and 32 for 24 epochs. This test uses LR = 2e-4, BS = 128, seed = 42, epoch = 24; time = 5 minutes per 5 epoch runs (1 epoch a minute):

| Num of groups | &darr; Training |  &darr; Testing  | &uarr; PSNR     | &uarr; SSIM     | &darr; LPIPS    |
| -------- | -------- | -------- | -------- | -------- | -------- |
| 16 @ 11       | 0.0308   | 0.0310   | 22.9320  | 0.9383   | 0.1982   |
| 16 @ 15       | 0.0304   | 0.0310   | 22.9367  | 0.9388   | 0.1959   |
| **16 @ 17** | **0.0300** | **0.0308** | **22.9822** | **0.9388** | **0.1937** |
| 16 @ 20       | 0.0293   | 0.0308   | 22.9445  | 0.9393   | 0.1981   |
| 16 @ 24       | 0.0288   | 0.0309   | 22.9807  | 0.9387   | 0.1938   |
| 32 @ 11       | 0.0306   | 0.0310   | 22.9528  | 0.9382   | 0.1982   |
| 32 @ 15       | 0.0301   | 0.0310   | 22.9653  | 0.9388   | 0.1968   |
| 32 @ 17       | 0.0296   | 0.0309   | 22.9853  | 0.9384   | 0.1939   |
| 32 @ 20       | 0.0288   | 0.0309   | 22.9793  | 0.9388   | 0.1963   |
| 32 @ 24       | 0.0282   | 0.0310   | 22.9815  | 0.9398   | 0.1946   |

## Observations pt. 1

This testing has shown that the switch to group normalization with 16 groups performed similarly to batch normalization after further training. This presents a two-sided situation. On one hand, the longer training suggests the model is learning more, meaning group normalization learns deeper representations than batch noramlization. On the other hand, batch normalization achieved slightly better results much more quickly (eight epochs faster), meaning the model uses less time and computational energy to produce similar outcomes.

Based on these findings, I will move forward with for batch noramlization testing.

## Next Steps

Testing perceptual loss on the batch normalization model.

## Results pt. 2

Adding perceptual loss

Changes: Changed the sigmoid function in U-Net to Tanh for better output matching for LPIPS, added perceptual loss to the training, and adapted the ab values to use Tanh instead of sigmoid where needed. This test uses LR = 2e-4, BS = 128, seed = 42, epoch = 80; time = 2.5 hours (around 2 minutes per epoch):

| Epoch #  | &darr; Training |  &darr; Testing  | &uarr; PSNR     | &uarr; SSIM     | &darr; LPIPS    |
| -------- | -------- | -------- | -------- | -------- | -------- |
| 2        | 0.0858   | 0.0653   | 22.5883  | 0.9346   | 0.1945   |
| 10       | 0.0782   | 0.0628   | 22.9586  | 0.9377   | 0.1821   |
| 20       | 0.0662   | 0.0621   | 22.9943  | 0.9372   | 0.1793   |
| 80       | 0.0301   | 0.0653   | 22.7600  | 0.9352   | 0.1893   |

## Observations pt. 2

Adding perceptual loss did not improve the model's performance. The pixel-level metrics did not improve much relatively, not to mention the increased loss in both training and testing. Overall, the model performed better before introducing perceptual loss.

---

## Final

**After all testing, the best hyperparameters for the model were LR = 2e-4, BS = 128, cosine annealing learning rate scheduler, no perceptual loss, using batch normalization, with the model performing best at epochs 8-10.**

Best results:

These are the results that had the best scores without showing signs of overfitting to the model. The range verys between seeds and the images used on the model, however epochs 8 and 9 with the hyperparameters mentioned prior produce result like the following:

| Num of groups | &darr; Training |  &darr; Testing  | &uarr; PSNR     | &uarr; SSIM     | &darr; LPIPS    |
| -------- | -------- | -------- | -------- | -------- | -------- |
| **8** | **0.0302** | **0.0309** | **23.0104** | **0.9389** | **0.1946** |
| **9** | **0.0301** | **0.0308** | **23.0207** | **0.9392** | **0.1940** |
