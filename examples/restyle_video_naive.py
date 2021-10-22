# Use VQGAN+CLIP to change the frames of an existing video using prompts. This is the naive original implementation where each frame is independent of the ones before.

from vqgan_clip import generate, video_tools, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
import os

config = VQGAN_CLIP_Config()
config.output_image_size = [256,256]
config.init_weight = 1.0
text_prompts = 'portrait on deviantart'
input_video_path = '20211004_132008000_iOS.MOV'
iterations = 15
final_output_filename = os.path.join('example_media','restyled_video_naive.mp4')
extraction_framerate = 30
output_framerate = 60
copy_audio = True
upscale_images = True
face_enhance=False

original_video_frames = video_tools.extract_video_frames(input_video_path, 
        extraction_framerate = extraction_framerate)

# Restyle the video by applying VQGAN to each frame independently
generated_video_frames_path='video_frames'
metadata_comment = generate.restyle_video_frames_naive(original_video_frames,
        eng_config=config,
        text_prompts = text_prompts,
        iterations = iterations,
        save_every=None,
        generated_video_frames_path = generated_video_frames_path)

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


generated_video_no_audio=os.path.join('example_media','output_no_audio.mp4')
video_tools.encode_video(output_file=generated_video_no_audio,
        path_to_stills=video_frames_to_encode,
        metadata_title=text_prompts,
        metadata_comment=metadata_comment,
        output_framerate=output_framerate,
        input_framerate=extraction_framerate)

if copy_audio:
        # Copy audio from the original file
        video_tools.copy_video_audio(input_video_path, generated_video_no_audio, final_output_filename)
        os.remove(generated_video_no_audio)
else:
        os.rename(generated_video_no_audio,final_output_filename)

print(f'generation parameters: {metadata_comment}')