# Generate a video with movement. Every frame that is generated has a shift or zoom applied to it.
# This gives the appearance of motion in the result. These videos do not stabilize.

# This is one of the most interesting application of VQGAN+CLIP here.
import vqgan_clip.generate
from vqgan_clip.engine import VQGAN_CLIP_Config
import os

config = VQGAN_CLIP_Config()
config.output_image_size = [448,448]
vqgan_clip.generate.zoom_video(eng_config = config,
        text_prompts = 'An abandoned shopping mall haunted by wolves|The war of the worlds',
        iterations = 2000,
        save_every = 5,
        output_filename = 'output' + os.sep + 'zooming',
        change_prompt_every = 300,
        output_framerate=30, 
        assumed_input_framerate=10, 
        zoom_scale=1.02, 
        shift_x=1, 
        shift_y=1)
