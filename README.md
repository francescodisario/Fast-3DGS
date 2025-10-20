# [COGS’25 – 2nd Place, ICCV 2025 Workshop] Post-Hoc Compression Algorithm 🚀

*Runner-up (2nd place) at the COGS’25 workshop, ICCV 2025 🥈*

Our compression method is **post-hoc** and takes inspiration from **gradient-based pruning** in deep neural networks, specifically *informative gradient–based pruning* 🧠✂️.

You can download compressed ckpts [here](https://drive.google.com/file/d/1pntjwW5F2RlT1vAnEZ2bAsm9v-1prRhr/view?usp=sharing).

## Overview 📌
Starting from a **pretrained Gaussian Splatting model** (checkpoint provided), the algorithm alternates between two phases:

1. **Gradient accumulation** over the training set.  
2. **Pruning + fine-tuning** to recover quality after each pruning step.

At each pruning step, we remove the Gaussians with the **smallest gradient magnitudes**.  
The **gradient of a Gaussian** is defined as the **sum of the norms of the gradients of all its parameters**.

We iterate this process—pruning a small fraction each time—until a **stopping criterion** is met ✅.

During fine-tuning, we apply **quantization** ⚙️:
- Spherical harmonics coefficients → **8-bit unsigned integers**  
- All other parameters → **16-bit float**


Finally, we **compress the parameters with LZMA** 💾.


## How to Run 🚀 ##
Download and install the [Vanilla 3DGS repo](https://github.com/graphdeco-inria/gaussian-splatting.git) along with the pre-trained [ckpts](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip).
Run **train.py**. Be sure to point to the right pre-trained dir. 
There is also an example of compatibility script to run it with Flash-GS (file **flashGS_example.py**).


## Computational Details 🧮
- **Gradient accumulation** requires a single pass over the training set and typically takes only a few seconds.  
- **Fine-tuning** runs **1,000 iterations per pruning step**, and **5,000 iterations** at the final stage.  
- In total, the algorithm performs about **30,000 iterations**.  
- Each pruning step removes **5% of the remaining Gaussians**.


### Results

Results obtained with vanilla-3dgs implementation

| Scene      | PSNR  | SSIM  | LPIPS | NUM GAUSSIANS | FPS | SIZE (MB) |
|------------|-------|-------|-------|---------------|-----|-----------|
| bicycle    | 25.22 | 0.755 | 0.248 | 579283        | 171 | 21.66     |
| bonsai     | 32.14 | 0.939 | 0.215 | 296058        | 205 | 10.56     |
| counter    | 28.98 | 0.902 | 0.220 | 203121        | 175 | 8.04      |
| flowers    | 21.55 | 0.596 | 0.368 | 491931        | 234 | 20.21     |
| garden     | 27.37 | 0.861 | 0.125 | 1130273       | 143 | 44.51     |
| kitchen    | 30.93 | 0.919 | 0.150 | 263772        | 185 | 10.39     |
| room       | 31.42 | 0.913 | 0.244 | 166795        | 195 | 5.62      |
| stump      | 26.66 | 0.767 | 0.260 | 327342        | 262 | 13.05     |
| treehill   | 22.27 | 0.590 | 0.457 | 94202         | 304 | 3.44      |
| **AVG**    | **27.39** | **0.805** | **0.254** | **394753** | **208** | **15.28** |
| drjohnson  | 28.99 | 0.897 | 0.276 | 165140        | 325 | 6.17      |
| playroom   | 30.22 | 0.910 | 0.252 | 381647        | 251 | 13.93     |
| **AVG**    | **29.61** | **0.903** | **0.264** | **273394** | **288** | **10.05** |
| train      | 21.79 | 0.764 | 0.296 | 87527         | 415 | 3.38      |
| truck      | 25.04 | 0.872 | 0.165 | 361868        | 243 | 12.64     |
| **AVG**    | **23.42** | **0.818** | **0.230** | **224698** | **329** | **8.01** |


## Acknowledgments

This work was built upon two 3DGS compression techniques.  
If you liked the project, please cite:

```bibtex
@inproceedings{ali2025elmgs,
  title     = {Elmgs: Enhancing memory and computation scalability through compression for 3D Gaussian splatting},
  author    = {Ali, Muhammad Salman and Bae, Sung-Ho and Tartaglione, Enzo},
  booktitle = {2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages     = {2591--2600},
  year      = {2025},
  organization = {IEEE}
}

@article{di2025gode,
  title   = {Gode: Gaussians on demand for progressive level of detail and scalable compression},
  author  = {Di Sario, Francesco and Renzulli, Riccardo and Grangetto, Marco and Sugimoto, Akihiro and Tartaglione, Enzo},
  journal = {arXiv preprint arXiv:2501.13558},
  year    = {2025}
}
