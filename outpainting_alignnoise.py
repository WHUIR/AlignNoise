# *************************************************************************
# Copyright (2023) ML Group @ RUC
# 
# Copyright (2023) SDE-Drag Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *************************************************************************

import argparse
import os
import random
import json
import math
import numpy as np
import torch
from PIL import Image
import datetime
from pipeline.pipeline_sd_alignnoise import StableDiffusionInpaintPipeline

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._add_arguments()
        self.opt = self.parser.parse_args()
        
    def _add_arguments(self):
        self.parser.add_argument("--seed", type=int, default=2226, help='random seed')
        self.parser.add_argument("--steps", type=int, default=50, help="sampling steps")

class ImageProcessor:
    def __init__(self, config):
        self.config = config
        self._setup_pipeline()
        self.mask_image = Image.open("./masks/mask_1.png").resize((512, 512))
        
    def _setup_pipeline(self):
        self._set_seed()
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
            safety_checker=None
        )
        self.pipeline.enable_model_cpu_offload()
        
    def _set_seed(self):
        torch.manual_seed(self.config.opt.seed)
        random.seed(self.config.opt.seed)
        np.random.seed(self.config.opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
            
    def process_images(self):
        filter_params = type('FilterParams', (), {
            'method': 'butterworth',
            'n': 4,
            'd_s': 0.25
        })()
        self.pipeline.init_filter(
            width=512,
            height=512, 
            filter_params=filter_params
        )

        images_dir = 'images' 
        output_dir = 'output_alignnoise' 
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(images_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(images_dir, filename)
            self._set_seed()
            init_image = Image.open(image_path).convert('RGB').resize((512, 512))

            image, _ = self.pipeline(
                prompt=" ",  
                image=init_image,
                mask_image=self.mask_image,
                return_dict=False, 
                num_inference_steps=self.config.opt.steps,
                num_iters=2,
                guidance_scale=0
            )
            output_path = os.path.join(output_dir, filename)
            image[0].save(output_path)
            
            print(f"{image_path} -> {output_path}")


def main():
    config = Config()
    processor = ImageProcessor(config)
    processor.process_images()

if __name__ == "__main__":
    main()
