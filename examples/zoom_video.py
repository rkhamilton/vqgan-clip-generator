# Generate a video with movement. Every frame that is generated has a shift or zoom applied to it.
# This gives the appearance of motion in the result. These videos do not stabilize.
# This example uses the ESRGAN upscaler before encoding the video.

from vqgan_clip import generate, video_tools, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
import os

# Let's generate a single image to initialize the video.
config = VQGAN_CLIP_Config()
config.output_image_size = [587,330]
text_prompts = 'A field of broken machines^Harvesting wheat'

upscale_images = True
face_enhance=False
final_video_filename = os.path.join('output','zoom_video.mp4')


init_image = os.path.join('output','init_image')
generate.single_image(eng_config = config,
        text_prompts = text_prompts,
        iterations = 100,
        save_every = None,
        output_filename = init_image)

# Now generate a zoom video starting from that initial frame.
generated_video_frames_path='video_frames'
generate.zoom_video_frames(eng_config = config,
        text_prompts = text_prompts,
        init_image = init_image+'.png',
        iterations = 1000,
        save_every = 5,
        video_frames_path = generated_video_frames_path,
        change_prompt_every = 300,
        zoom_scale=1.02, 
        shift_x=1, 
        shift_y=1)

# Upscale the video frames
if upscale_images:
        upscaled_video_frames_path='upscaled_video_frames'
        esrgan.inference_realesrgan(input=generated_video_frames_path,
                output_images_path=upscaled_video_frames_path,
                face_enhance=face_enhance,
                purge_existing_files=True,
                netscale=4,
                outscale=4)
        video_frames_to_encode = upscaled_video_frames_path
else:
        video_frames_to_encode = generated_video_frames_path

# Encode the video
video_tools.encode_video(output_file=final_video_filename,
        path_to_stills=video_frames_to_encode,
        metadata=text_prompts,
        output_framerate=60,
        assumed_input_framerate=30)
