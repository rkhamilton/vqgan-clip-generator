# This is an example of using restyle_video to apply VQGAN styling to an existing video.

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

# Use a wrapper for FFMPEG to extract stills from the original video.
original_video_frames = video_tools.extract_video_frames(input_video_path, 
        extraction_framerate = extraction_framerate)

# Apply a style to the extracted video frames.
generate.restyle_video_frames(original_video_frames,
        eng_config=config,
        text_prompts = text_prompts,
        iterations = 20,
        save_every=None,
        current_source_frame_prompt_weight=0.1,
        previous_generated_frame_prompt_weight=0.0,
        generated_frame_init_blend=0.2)

# Use a wrapper for FFMPEG to encode the video.
generated_video_no_audio=os.path.join('output','output_no_audio.mp4')
video_tools.encode_video(output_file=generated_video_no_audio,
        metadata=text_prompts,
        output_framerate=output_framerate,
        assumed_input_framerate=extraction_framerate)

# Copy audio from the original file
if copy_audio:
        video_tools.copy_video_audio(input_video_path, generated_video_no_audio, final_output_filename)
        os.remove(generated_video_no_audio)
else:
        os.rename(generated_video_no_audio,final_output_filename)