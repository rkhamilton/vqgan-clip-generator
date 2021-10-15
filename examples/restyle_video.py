# This is an example of using restyle_video to apply VQGAN styling to an existing video.

import vqgan_clip.generate
from vqgan_clip.engine import VQGAN_CLIP_Config
import os

config = VQGAN_CLIP_Config()
config.output_image_size = [448,448]
config.init_weight = 0.2
blend_weight_new_frame=0.2 # How much of the previous generated frame to blend with the new orginal frame for init_image of a new generated frame. 0=all original frame, 1=all previous generated frame.
input_video_path = '20211004_132008000_iOS.MOV'
vqgan_clip.generate.restyle_video(input_video_path,
        extraction_framerate = 30,
        eng_config=config,
        text_prompts = 'Covered in spiders | Surreal:0.5',
        iterations = 30,
        save_every=None,
        output_filename = 'output' + os.sep + 'outputfile',
        output_framerate=60,
        copy_audio=False,
        current_source_frame_prompt_weight=0.1,
        generated_frame_init_weight=0.2)
