# Generate a vide based on a text prompt. Note that the image will stabilize after a hundred or so iteration with the same prompt,
# so this is most useful if you are changing prompts over time. In the exmaple below the prompt cycles between two every 300 iterations.
from vqgan_clip import generate, video_tools, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
import os

#Let's generate a single image to initialize the video.
config = VQGAN_CLIP_Config()
config.output_image_size = [587,330]
text_prompts = 'A pastoral landscape painting by Rembrandt^A black dog with red eyes in a cave'
final_video_filename = os.path.join('example_media','video.mp4')
iterations = 500
upscale_images = True
face_enhance = False

init_image = os.path.join('example_media','init_image.png')
generate.single_image(eng_config = config,
        text_prompts = text_prompts,
        iterations = 100,
        save_every = None,
        output_filename = init_image)
        
# Now generate a zoom video starting from that initial frame.
generated_video_frames_path='video_frames'
metadata_comment = generate.video_frames(eng_config = config,
        text_prompts = text_prompts,
        init_image = init_image,
        video_frames_path = generated_video_frames_path,
        iterations = iterations,
        save_every = 10,
        change_prompt_every = 100)

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

# Use a wrapper for FFMPEG to encode the video.
video_tools.encode_video(output_file=final_video_filename,
        path_to_stills=video_frames_to_encode,
        metadata_title=text_prompts,
        metadata_comment=metadata_comment,
        output_framerate=60,
        input_framerate=30)

print(f'generation parameters: {metadata_comment}')