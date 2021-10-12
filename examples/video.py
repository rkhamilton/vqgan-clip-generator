# Generate a vide based on a text prompt. Note that the image will stabilize after a hundred or so iteration with the same prompt,
# so this is most useful if you are changing prompts over time. In the exmaple below the prompt cycles between two every 300 iterations.
import vqgan_clip.generate
from vqgan_clip.engine import VQGAN_CLIP_Config
import os

config = VQGAN_CLIP_Config()
config.output_image_size = [128,128]
text_prompts = 'A pastoral landscape painting by Rembrandt^A black dog with red eyes in a cave'
vqgan_clip.generate.video(eng_config = config,
        text_prompts = text_prompts,
        iterations = 1000,
        save_every = 10,
        output_filename = 'output' + os.sep + 'output',
        change_prompt_every = 300)



