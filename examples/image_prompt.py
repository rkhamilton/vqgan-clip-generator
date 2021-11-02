# Generate a single image based on a image prompt
# Note that any input images or video are not provided for example scripts, you will have to provide your own.

from vqgan_clip import generate, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
import os
from vqgan_clip import _functional as VF

config = VQGAN_CLIP_Config()
config.output_image_size = [448, 448]
upscale_image = True
face_enhance = False
generated_image_filename = f'example media{os.sep}image prompt.jpg'

metadata_comment = generate.image(eng_config=config,
                                  image_prompts='input image.jpg',
                                  iterations=200,
                                  output_filename=generated_image_filename)

# Upscale the video frames
if upscale_image:
    esrgan.inference_realesrgan(input=generated_image_filename,
                                output_images_path='example media',
                                face_enhance=face_enhance,
                                netscale=4,
                                outscale=4)
    # copy metadata from generated images to upscaled images.
    generated_image_basename = os.path.basename(generated_image_filename)
    output_filename_noext = os.path.splitext(generated_image_basename)[0]
    output_filepath = f'example media{os.sep}{output_filename_noext}_upscaled.jpg'
    VF.copy_image_metadata(generated_image_filename, output_filepath)
print(f'generation parameters: {metadata_comment}')
