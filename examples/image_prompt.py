# Generate a single image based on a text prompt
from vqgan_clip import generate, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
import os

config = VQGAN_CLIP_Config()
config.output_image_size = [448,448]
upscale_image = True
output_filename = os.path.join('output','image_prompt')

generate.single_image(eng_config = config,
        image_prompts = 'input_image.jpg',
        iterations = 500,
        save_every = 10,
        output_filename = output_filename)

# Upscale the video frames
if upscale_image:
        esrgan.inference_realesrgan(input=output_filename+'.png',
                output_images_path='output',
                face_enhance=True,
                netscale=4,
                outscale=4)