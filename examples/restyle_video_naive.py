# Use VQGAN+CLIP to change the frames of an existing video using prompts. This is the naive original implementation where each frame is independent of the ones before.

from vqgan_clip import generate, video_tools
from vqgan_clip.engine import VQGAN_CLIP_Config
import os

config = VQGAN_CLIP_Config()
config.output_image_size = [256,256]
config.init_weight = 1.0
text_prompts = 'portrait on deviantart'
input_video_path = '20211004_132008000_iOS.MOV'
final_output_filename = os.path.join('output','output.mp4')
copy_audio = True
extraction_framerate = 30
output_framerate = 60

original_video_frames = video_tools.extract_video_frames(input_video_path, 
        extraction_framerate = extraction_framerate)

generate.restyle_video_frames_naive(original_video_frames,
        eng_config=config,
        text_prompts = text_prompts,
        iterations = 15,
        save_every=None)

generated_video_no_audio=os.path.join('output','output_no_audio.mp4')
video_tools.encode_video(output_file=generated_video_no_audio,
        metadata=text_prompts,
        output_framerate=output_framerate,
        assumed_input_framerate=extraction_framerate)

if copy_audio:
        # Copy audio from the original file
        video_tools.copy_video_audio(input_video_path, generated_video_no_audio, final_output_filename)
        os.remove(generated_video_no_audio)
else:
        os.rename(generated_video_no_audio,final_output_filename)