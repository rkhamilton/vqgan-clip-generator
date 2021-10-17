# Generate a video with movement. Every frame that is generated has a shift or zoom applied to it.
# This gives the appearance of motion in the result. These videos do not stabilize.

# This is one of the most interesting application of VQGAN+CLIP here.
from vqgan_clip import generate, video_tools
from vqgan_clip.engine import VQGAN_CLIP_Config
import os

#Let's generate a single image to initialize the video.
config = VQGAN_CLIP_Config()
config.output_image_size = [448,448]
text_prompts = 'An abandoned shopping mall haunted by wolves^The war of the worlds'
init_image = os.path.join('output','init_image')
generate.single_image(eng_config = config,
        text_prompts = text_prompts,
        iterations = 100,
        save_every = None,
        output_filename = init_image)

# Now generate a zoom video starting from that initial frame.
config.init_image = init_image+'.png'
generate.zoom_video_frames(eng_config = config,
        text_prompts = text_prompts,
        iterations = 1000,
        save_every = 5,
        change_prompt_every = 300,
        zoom_scale=1.02, 
        shift_x=1, 
        shift_y=1)

video_tools.encode_video(output_file=os.path.join('output','zoom_video.mp4'),
        metadata=text_prompts,
        output_framerate=60,
        assumed_input_framerate=30)