# Generate a single image based on a text prompt
from vqgan_clip import generate, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
import os

config = VQGAN_CLIP_Config()
config.output_image_size = [448,448]
upscale_image = True
text_prompts = 'A pastoral landscape painting by Rembrandt'

output_filename = os.path.join('example_media',text_prompts+'.png')
metadata_comment = generate.image(eng_config = config,
        text_prompts = text_prompts,
        iterations = 100,
        save_every = 50,
        output_filename = output_filename)

# Upscale the image
if upscale_image:
        esrgan.inference_realesrgan(input=output_filename,
                output_images_path='example_media',
                face_enhance=False,
                netscale=4,
                outscale=4)
print(f'generation parameters: {metadata_comment}')