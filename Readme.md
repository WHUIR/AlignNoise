# Bridging the Gap: Consistent Image Outpainting via Training-Free Noise Optimization

[![Conference](https://img.shields.io/badge/ACM%20MM-2025-blue)](https://2025.acmmm.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[Paper]** (Link will be provided later) | **[Code]** 

This repository contains the official implementation for the paper "Bridging the Gap: Consistent Image Outpainting via Training-Free Noise Optimization", accepted by **ACM Multimedia 2025**.



## Status!
* ✅ [2025.7.12] Our paper is accepted by **ACM Multimedia 2025**!
* ✅ [2024.8.07] The source code has been released.


## Dependency

```bash
conda create -n alignnoise python=3.9
conda activate alignnoise
pip install -r requirements.txt
```

## AlignNoise

To perform image outpainting with AlignNoise, simply run the main script:
```bash
python outpainting_alignnoise.py
```


You can configure the source image, mask, and other parameters within the script.



## Key Features

* **Training-Free:** Works as a plug-and-play module for any pre-trained diffusion model, avoiding expensive training or fine-tuning.
* **Enhanced Semantic Alignment:** Utilizes a novel attention-based noise optimization strategy to ensure the generated content is semantically consistent with the source image.
* **Artifact Reduction:** Employs a frequency-based noise refinement technique to suppress abnormal patterns and high-frequency artifacts, leading to cleaner results.
* **State-of-the-Art Performance:** Achieves significant improvements on standard metrics.




## Acknowledgments
---
The code is built upon [diffusers](https://github.com/huggingface/diffusers), [initno](https://github.com/xiefan-guo/initno/blob/main/run_sd_initno.py) and [FreeInit](https://github.com/TianxingWu/FreeInit) we thank all the contributors for open-sourcing.
