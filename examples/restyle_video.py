# This is an example of using restyle_video to apply VQGAN styling to an existing video.
# Note that any input images or video are not provided for example scripts, you will have to provide your own.
# NOTE: THIS IS AN EXAMPLE OF USING THE DEPRECATED STYLE TRANSFER METHOD which is kept in case you prefer the older look.
# See example/style_transfer.py for the current method.

from vqgan_clip import generate, video_tools, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
import os
from vqgan_clip import _functional as VF

config = VQGAN_CLIP_Config()
config.output_image_size = [256, 256]
config.init_weight = 0.1
text_prompts = 'portrait on deviantart'
input_video_path = '20211004_132008000_iOS.MOV'
output_root_dir = 'example media'
final_output_filename = f'{output_root_dir}{os.sep}restyled video.mp4'
copy_audio = True
video_exraction_framerate = 15
output_framerate = 30
# Set True if you installed the Real-ESRGAN package for upscaling. face_enhance is a feature of Real-ESRGAN.
upscale_images = True
face_enhance = False

# Set some paths
generated_video_frames_path = os.path.join(output_root_dir, 'generated video frames')
upscaled_video_frames_path = os.path.join(output_root_dir, 'upscaled video frames')
extracted_video_frames_path = os.path.join(output_root_dir, 'extracted video frames')

# Use a wrapper for FFMPEG to extract stills from the original video.
original_video_frames = video_tools.extract_video_frames(input_video_path,
                                                         extraction_framerate=video_exraction_framerate,
                                                         extracted_video_frames_path=extracted_video_frames_path)

# Apply a style to the extracted video frames.
metadata_comment = generate.restyle_video_frames(original_video_frames,
                                                 eng_config=config,
                                                 text_prompts=text_prompts,
                                                 iterations=15,
                                                 save_every=None,
                                                 generated_video_frames_path=generated_video_frames_path,
                                                 current_source_frame_prompt_weight=0.1,
                                                 previous_generated_frame_prompt_weight=0.0,
                                                 generated_frame_init_blend=0.05,
                                                 z_smoother=True,
                                                 z_smoother_buffer_len=3,
                                                 z_smoother_alpha=0.9)

# Upscale the video frames
if upscale_images:
    esrgan.inference_realesrgan(input=generated_video_frames_path,
                                output_images_path=upscaled_video_frames_path,
                                face_enhance=face_enhance,
                                purge_existing_files=True,
                                netscale=4,
                                outscale=4)
    # copy metadata from generated images to upscaled images.
    VF.copy_image_metadata(generated_video_frames_path, upscaled_video_frames_path)
    video_frames_to_encode = upscaled_video_frames_path
else:
    video_frames_to_encode = generated_video_frames_path

# Use a wrapper for FFMPEG to encode the video.
generated_video_no_audio = f'example media{os.sep}output no audio.mp4'
video_tools.encode_video(output_file=generated_video_no_audio,
                         path_to_stills=video_frames_to_encode,
                         metadata_title=text_prompts,
                         metadata_comment=metadata_comment,
                         output_framerate=output_framerate,
                         input_framerate=video_exraction_framerate)

# Copy audio from the original file
if copy_audio:
    video_tools.copy_video_audio(
        input_video_path, generated_video_no_audio, final_output_filename)
    os.remove(generated_video_no_audio)
else:
    os.rename(generated_video_no_audio, final_output_filename)


print(f'generation parameters: {metadata_comment}')
