# Generate a single image based on a text prompt
from vqgan_clip import generate
from vqgan_clip.engine import VQGAN_CLIP_Config
import os

config = VQGAN_CLIP_Config()
config.output_image_size = [448,448]
text_prompts = 'A pastoral landscape painting by Rembrandt'
generate.single_image(eng_config = config,
        text_prompts = text_prompts,
        iterations = 100,
        save_every = 50,
        output_filename = 'output' + os.sep + text_prompts)
