# Generate a single image based on a text prompt
# Note that any input images or video are not provided for example scripts, you will have to provide your own.

from vqgan_clip import generate, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
import os
from vqgan_clip import _functional as VF

config = VQGAN_CLIP_Config()
config.output_image_size = [684, 384]
# Set True if you installed the Real-ESRGAN package for upscaling.
upscale_image = True
text_prompts = 'A pastoral landscape painting by Rembrandt'

output_filename = f'example media{os.sep}example image.jpg'
metadata_comment = generate.image(eng_config=config,
                                  text_prompts=text_prompts,
                                  iterations=400,
                                  output_filename=output_filename)

# Upscale the image
if upscale_image:
    esrgan.inference_realesrgan(input=output_filename,
                                output_images_path='example media',
                                face_enhance=False,
                                netscale=4,
                                outscale=4)
    VF.copy_image_metadata(output_filename, os.path.splitext(output_filename)[0]+'_upscaled.jpg')
print(f'generation parameters: {metadata_comment}')
