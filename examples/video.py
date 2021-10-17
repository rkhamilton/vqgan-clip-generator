# Generate a vide based on a text prompt. Note that the image will stabilize after a hundred or so iteration with the same prompt,
# so this is most useful if you are changing prompts over time. In the exmaple below the prompt cycles between two every 300 iterations.
from vqgan_clip import generate, video_tools
from vqgan_clip.engine import VQGAN_CLIP_Config
import os

#Let's generate a single image to initialize the video.
config = VQGAN_CLIP_Config()
config.output_image_size = [448,448]
text_prompts = 'A pastoral landscape painting by Rembrandt^A black dog with red eyes in a cave'
init_image = os.path.join('output','init_image')
generate.single_image(eng_config = config,
        text_prompts = text_prompts,
        iterations = 100,
        save_every = None,
        output_filename = init_image)

# Now generate a zoom video starting from that initial frame.
config.init_image = init_image+'.png'
generate.video_frames(eng_config = config,
        text_prompts = text_prompts,
        iterations = 1000,
        save_every = 10,
        change_prompt_every = 300)

# Use a wrapper for FFMPEG to encode the video.
video_tools.encode_video(output_file=os.path.join('output','zoom_video.mp4'),
        metadata=text_prompts,
        output_framerate=60,
        assumed_input_framerate=30)
