# Generate a folder of multiple images based on a text prompt.
# This might be useful if you want to try different random number generator seeds
from vqgan_clip import generate, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
from vqgan_clip import _functional as VF
import os
from tqdm import tqdm

config = VQGAN_CLIP_Config()
config.output_image_size = [256,256]
text_prompts = 'A pastoral landscape painting by Rembrandt'
generated_images_path='./video_frames'
upscaled_video_frames_path='./upscaled_video_frames'
number_images_to_generate = 5
# Set True if you installed the Real-ESRGAN package for upscaling. face_enhance is a feature of Real-ESRGAN.
upscale_images = True
face_enhance = False

for image_number in tqdm(range(1,number_images_to_generate+1),unit='image',desc='random seeds'):
        metadata_comment = generate.image(eng_config = config,
                text_prompts = text_prompts,
                iterations = 50,
                output_filename =  os.path.join(generated_images_path,f'frame_{image_number:012d}.png'),
                leave_progress_bar=False)

# Upscale the image
if upscale_images:
        esrgan.inference_realesrgan(input=generated_images_path,
                output_images_path=upscaled_video_frames_path,
                face_enhance=face_enhance,
                purge_existing_files=True,
                netscale=4,
                outscale=4)
        # copy PNG metadata from generated images to upscaled images
        VF.copy_PNG_metadata(generated_images_path,upscaled_video_frames_path)
print(f'generation parameters: {metadata_comment}')