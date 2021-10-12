# Generate a folder of multiple images based on a text prompt.
# This might be useful if you want to try different random number generator seeds
import vqgan_clip.generate
from vqgan_clip.engine import VQGAN_CLIP_Config
import os

config = VQGAN_CLIP_Config()
config.output_image_size = [128,128]
text_prompts = 'A pastoral landscape painting by Rembrandt'
vqgan_clip.generate.multiple_images(eng_config = config,
        text_prompts = text_prompts,
        iterations = 50,
        save_every = 51,
        num_images_to_generate=3)

