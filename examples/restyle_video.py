# This is an example of using restyle_video to apply VQGAN styling to an existing video.

from vqgan_clip import generate, video_tools, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
import os

config = VQGAN_CLIP_Config()
config.output_image_size = [256,256]
config.init_image_method = 'original'
config.init_weight = 0.1
text_prompts = 'portrait on deviantart'
input_video_path = '20211004_132008000_iOS.MOV'
final_output_filename = os.path.join('example_media','restyled_video.mp4')
copy_audio = True
extraction_framerate = 30
output_framerate = 30

upscale_images = True
face_enhance=False

# Use a wrapper for FFMPEG to extract stills from the original video.
original_video_frames = video_tools.extract_video_frames(input_video_path, 
        extraction_framerate = extraction_framerate)

# Apply a style to the extracted video frames.
generated_video_frames_path='video_frames'
metadata_comment = generate.restyle_video_frames(original_video_frames,
        eng_config=config,
        text_prompts = text_prompts,
        iterations = 15,
        save_every=None,
        generated_video_frames_path = generated_video_frames_path,
        current_source_frame_prompt_weight=0.1,
        previous_generated_frame_prompt_weight=0.0,
        generated_frame_init_blend=0.05,
        z_smoother=True,
        z_smoother_buffer_len=5,
        z_smoother_alpha=0.9)

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
generated_video_no_audio=os.path.join('example_media','output_no_audio.mp4')
video_tools.encode_video(output_file=generated_video_no_audio,
        path_to_stills=video_frames_to_encode,
        metadata_title=text_prompts,
        metadata_comment=metadata_comment,
        output_framerate=output_framerate,
        input_framerate=extraction_framerate)

# Copy audio from the original file
if copy_audio:
        video_tools.copy_video_audio(input_video_path, generated_video_no_audio, final_output_filename)
        os.remove(generated_video_no_audio)
else:
        os.rename(generated_video_no_audio,final_output_filename)


print(f'generation parameters: {metadata_comment}')