# Bridging the Gap: Consistent Image Outpainting via Training-Free Noise Optimization

[![Conference](https://img.shields.io/badge/ACM%20MM-2025-blue)](https://2025.acmmm.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[Paper]** (Link will be provided later) | **[Code]** (Coming in August 2025)

This repository contains the official implementation for the paper "Bridging the Gap: Consistent Image Outpainting via Training-Free Noise Optimization", accepted by **ACM Multimedia 2025**.

`AlignNoise` is a novel, training-free, two-stage noise optimization algorithm for image outpainting. It enhances pre-trained diffusion models to generate high-quality, semantically consistent content beyond the original image boundaries, effectively resolving common issues like semantic misalignment and the generation of irregular high-frequency patterns without any fine-tuning.

---

## 🚀 Key Features

* **Training-Free:** Works as a plug-and-play module for any pre-trained diffusion model, avoiding expensive training or fine-tuning.
* **Enhanced Semantic Alignment:** Utilizes a novel attention-based noise optimization strategy to ensure the generated content is semantically consistent with the source image.
* **Artifact Reduction:** Employs a frequency-based noise refinement technique to suppress abnormal patterns and high-frequency artifacts, leading to cleaner results.
* **State-of-the-Art Performance:** Achieves significant improvements on standard metrics.




## 🗓️ Status

* ✅ Paper accepted by **ACM Multimedia 2025**.
* ✅ Code is made publicly available by **August 2025**.


## Acknowledgments
---
The code is built upon [diffusers](https://github.com/huggingface/diffusers), [initno](https://github.com/xiefan-guo/initno/blob/main/run_sd_initno.py) and [FreeInit](https://github.com/TianxingWu/FreeInit) we thank all the contributors for open-sourcing.