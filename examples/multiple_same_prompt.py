# Generate a folder of multiple images based on a text prompt.
# This might be useful if you want to try different random number generator seeds
from vqgan_clip import generate, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
import os

config = VQGAN_CLIP_Config()
config.output_image_size = [448,448]
text_prompts = 'A pastoral landscape painting by Rembrandt'
upscale_images = True
output_images_path='./video_frames'
face_enhance = False

generate.multiple_images(eng_config = config,
        text_prompts = text_prompts,
        iterations = 50,
        save_every = 51,
        num_images_to_generate=3,
        output_images_path=output_images_path)

# Upscale the video frames
if upscale_images:
        upscaled_video_frames_path='upscaled_video_frames'
        esrgan.inference_realesrgan(input=output_images_path,
                output_images_path=upscaled_video_frames_path,
                face_enhance=face_enhance,
                purge_existing_files=True,
                netscale=4,
                outscale=4)
