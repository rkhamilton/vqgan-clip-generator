# Generate a single image based on a text prompt
import vqgan_clip.generate
from vqgan_clip.engine import VQGAN_CLIP_Config
import os

config = VQGAN_CLIP_Config()
config.output_image_size = [448,448]
vqgan_clip.generate.single_image(eng_config = config,
        image_prompts = 'input_image.jpg',
        iterations = 500,
        save_every = 10,
        output_filename = 'output' + os.sep + 'input image test')
