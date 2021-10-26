# Generate a folder of multiple images based on a text prompt.
# This might be useful if you want to try different random number generator seeds
from vqgan_clip import generate, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
from vqgan_clip import _functional as VF
import os

config = VQGAN_CLIP_Config()
config.output_image_size = [128,128]
text_prompts = 'robot'
generated_images_path='./video_frames'
upscaled_video_frames_path='./video_frames_upscaled'
number_images_to_generate = 10
# Set True if you installed the Real-ESRGAN package for upscaling. face_enhance is a feature of Real-ESRGAN.
upscale_images = True
face_enhance = False

art_types=["8k resolution","pencil sketch","8K 3D","deviantart","CryEngine",
            "Unreal Engine","photo illustration","pixiv","Flickr","Artstation HD",
            "Behance HD","HDR","anime","Ambient occlusion","Global illumination",
            "Chat art","Low poly","Booru","Polycount","Acrylic Art","Hyperrealism",
            "Zbrush Central","Rendered in Cinema4D","Rendered in Maya","Tilt Shift",
            "Mixed Media","Detailed painting","Volumetric lighting",
            "Storybook Illustration","#vfxfriday","Ultrafine detail","matte painting",
            "Watercolor","CGSocity","child's draing","marble sculpture","airbrush art",
            "renaissance painting","Velvia","dye-transfer","stipple","Parrallax",
            "Bryce 3D","Terragen","Bokeh","charcoal drawing","commission for",
            "polished","aftereffects","datamosh","holographic","dutch golden age",
            "digitally enhanced","Art on Instagram","bokeh","psychedelic","wavy",
            "groovy","movie poster","pop art","made of beads and yarn","made of feathers",
            "made of crystals","made of liquid metal","made of glass","made of cardboard",
            "made of vines","made of flowers","made of insects","made of mist",
            "made of paperclips","made of rubber","made of wire","made of trash",
            "made of wrought iron","tattoo","woodcut","American propaganda",
            "Soviet propaganda","Fine Art","Photorealism","drone shot",
            "poster art","Impressionism","Lowbrow","Egyptian art"]

for art_type in art_types:
        metadata_comment = generate.image(eng_config = config,
                text_prompts = text_prompts + ' ' + art_type,
                image_prompts = 'input_image.jpg',
                iterations = 100,
                output_filename =  os.path.join(generated_images_path,f'{art_type}.png'))

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