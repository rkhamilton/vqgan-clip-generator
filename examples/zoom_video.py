# Generate a video with movement. Every frame that is generated has a shift or zoom applied to it.
# This gives the appearance of motion in the result. These videos do not stabilize.
# This example uses the ESRGAN upscaler before encoding the video.

from vqgan_clip import generate, video_tools, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
import os
import subprocess
from vqgan_clip import _functional as VF

# Let's generate a single image to initialize the video.
config = VQGAN_CLIP_Config()
config.output_image_size = [256, 144]
text_prompts = 'A pastoral landscape painting by Rembrandt^A black dog with red eyes in a cave^Apple pie'
num_video_frames = 150
video_framerate = 15
output_root_dir = 'example media'
final_video_filename = os.path.join(output_root_dir, 'zoom video.mp4')
# Set True if you installed the Real-ESRGAN package for upscaling
upscale_images = True
face_enhance = False
# Set True if you installed the RIFE package for optical flow interpolation
# IMPORTANT - OF will increase the framerate by 4x (-exp=2 option) or 16x (-exp=4). Keep this in mind as you generate your VQGAN video.
# Suggested video_framerate 15 or 30 with 4x interpolation.
interpolate_with_RIFE = True

# set some paths
generated_video_frames_path = os.path.join(output_root_dir, 'video frames')
upscaled_video_frames_path = os.path.join(output_root_dir, 'upscaled video frames')

# Now generate a zoom video starting from that initial frame.
metadata_comment = generate.video_frames(num_video_frames=num_video_frames,
                                         eng_config=config,
                                         text_prompts=text_prompts,
                                         generated_video_frames_path=generated_video_frames_path,
                                         iterations_per_frame=15,
                                         change_prompts_on_frame=[60, 100],
                                         zoom_scale=1.02,
                                         shift_x=1,
                                         shift_y=1,
                                         z_smoother=True,
                                         z_smoother_buffer_len=3,
                                         z_smoother_alpha=0.9)

# Upscale the video frames
if upscale_images:
    esrgan.inference_realesrgan(input=generated_video_frames_path,
                                output_images_path=upscaled_video_frames_path,
                                face_enhance=face_enhance,
                                purge_existing_files=True,
                                model_filename='RealESRGAN_x4plus_anime_6B.pth',
                                model_url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
                                netscale=4,
                                outscale=4)
    # copy PNG metadata from generated images to upscaled images
    VF.copy_PNG_metadata(generated_video_frames_path,
                         upscaled_video_frames_path)
    video_frames_to_encode = upscaled_video_frames_path
else:
    video_frames_to_encode = generated_video_frames_path

# Use a wrapper for FFMPEG to encode the video.
video_tools.encode_video(output_file=final_video_filename,
                         path_to_stills=video_frames_to_encode,
                         metadata_title=text_prompts,
                         metadata_comment=metadata_comment,
                         input_framerate=video_framerate)

print(f'generation parameters:\n{metadata_comment}')

if interpolate_with_RIFE:
    video_tools.RIFE_interpolation(input=final_video_filename,
                       output=f'{os.path.splitext(final_video_filename)[0]}_RIFE.mp4',
                       interpolation_factor=4,
                       metadata_title=text_prompts,
                       metadata_comment=metadata_comment)
