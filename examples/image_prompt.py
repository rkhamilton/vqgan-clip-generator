# Generate a single image based on a image prompt
from vqgan_clip import generate, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
import os
from vqgan_clip import _functional as VF

config = VQGAN_CLIP_Config()
config.output_image_size = [448, 448]
upscale_image = True
face_enhance = False
output_filename = f'example media{os.sep}image prompt.png'

metadata_comment = generate.image(eng_config=config,
                                  image_prompts='input image.jpg',
                                  iterations=200,
                                  output_filename=output_filename)

# Upscale the video frames
if upscale_image:
    esrgan.inference_realesrgan(input=output_filename,
                                output_images_path='example media',
                                face_enhance=face_enhance,
                                netscale=4,
                                outscale=4)
    VF.copy_PNG_metadata(output_filename, os.path.splitext(output_filename)[0]+'_upscaled.png')

print(f'generation parameters: {metadata_comment}')
