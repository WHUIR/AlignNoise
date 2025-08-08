# Bridging the Gap: Consistent Image Outpainting via Training-Free Noise Optimization

[![Conference](https://img.shields.io/badge/ACM%20MM-2025-blue)](https://2025.acmmm.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[Paper]** (Link will be provided later) | **[Code]** 

This repository contains the official implementation for the paper "Bridging the Gap: Consistent Image Outpainting via Training-Free Noise Optimization", accepted by **ACM Multimedia 2025**.

Abstract:
Image outpainting has drawn increasing demands from many real-world applications. The core capacity called for this task is to generate image content beyond the boundaries that are semantically aligned with the source image. Compared to other image generation tasks, image outpainting remains very challenging since we need to identify the scene of the source image and generate new yet consistent boundaries with few local context. However, one common propensity for the outpainting techniques is to generate irregular high-frequency patterns. Furthermore, the dominating data-driven learning paradigm utilized by the existing state-of-the-art methods would require sophisticated model design, significant computation cost and introduce potential bias as well.

To address these challenges, we propose a training-free two-stage noise optimization algorithm for image outpainting (named AlignNoise). Specifically, to avoid expensive post fine-tuning, we utilize the pre-trained diffusion model as generator for image outpainting. Then, we analyze the attention patterns within the outpainted region. It is observed that when these attention scores disproportionately focus more on the outpainted region while neglecting the source area during the initial sampling steps, the inconsistent alignment between the outpainted and source regions is likely to arise. Hence, we devise a noise optimization strategy in AlignNoise to adjust attention between the two regions such that the scene of source image can be well captured, leading to better semantic alignment. Moreover, we propose to refine the initial noise by leveraging the frequency features of the denoised outpainted region, effectively reducing abnormal pattern generation without significantly altering the image’s composition or distribution. In essence, the resultant AlignNoise works as a plug-and-play for the underlying or any generator. Extensive experiments demonstrate the superiority of AlignNoise in resolving semantic alignment without any fine-tuning for image outpainting. Specifically, AlignNoise achieves average improvements of up to 7.64 and 5.78 in terms of FID and IS metric respectively.


## Status
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
