# Use VQGAN+CLIP to change the frames of an existing video using prompts. This is the naive original implementation where each frame is independent of the ones before.

import vqgan_clip.generate
from vqgan_clip.engine import VQGAN_CLIP_Config
import os

config = VQGAN_CLIP_Config()
config.output_image_size = [448,448]
config.init_weight = 0.2
input_video_path = 'my_video.MOV'
text_prompts = 'A hairy ape'
vqgan_clip.generate.restyle_video_naive(input_video_path,
        extraction_framerate = 5,
        eng_config=config,
        text_prompts = text_prompts,
        iterations = 15,
        output_filename = 'output' + os.sep + text_prompts,
        output_framerate=30,
        copy_audio=True)