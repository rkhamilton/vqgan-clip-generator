# This is an example of using style_transfer to apply VQGAN styling to images extracted from a video.
# Key parameters to experiment with are iterations_per_frame, current_source_frame_image_weight, and current_source_frame_prompt_weight

from vqgan_clip import generate, video_tools, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
import os
import subprocess
from vqgan_clip import _functional as VF

config = VQGAN_CLIP_Config()
config.output_image_size = [256, 256]
text_prompts = 'portrait covered in spiders charcoal drawing'
input_video_path = '20211004 132008000_iOS.MOV'
output_root_dir = 'example media'
# Generated video framerate. Images will be extracted from the source video at this framerate, using interpolation if needed.
video_exraction_framerate = 30
final_video_filename = os.path.join(output_root_dir, 'style transfer.mp4')
# Set True if you want to copy audio from the original video to the output
copy_audio = True
# Set True if you installed the Real-ESRGAN package for upscaling. face_enhance is a feature of Real-ESRGAN.
upscale_images = True
face_enhance = False
# Set True if you installed the RIFE package for optical flow interpolation
# IMPORTANT - RIFE will increase the framerate by 4x (-exp=2 option) or 16x (-exp=4). Keep this in mind as you generate your VQGAN video.
# Suggested video_framerate 15 or 30 with the default 4x interpolation.
interpolate_with_RIFE = True

# Set some paths
generated_video_frames_path = os.path.join(output_root_dir, 'generated video frames')
upscaled_video_frames_path = os.path.join(output_root_dir, 'upscaled video frames')

# Use a wrapper for FFMPEG to extract stills from the original video.
original_video_frames = video_tools.extract_video_frames(input_video_path,
                                                         extraction_framerate=video_exraction_framerate,
                                                         extracted_video_frames_path=os.path.join(output_root_dir, 'extracted video frames'))

# Apply a style to the extracted video frames.
metadata_comment = generate.style_transfer(original_video_frames,
                                           eng_config=config,
                                           current_source_frame_image_weight=3.2,
                                           current_source_frame_prompt_weight=0.0,
                                           text_prompts=text_prompts,
                                           iterations_per_frame=60,
                                           change_prompts_on_frame=[],
                                           generated_video_frames_path=generated_video_frames_path,
                                           z_smoother=True,
                                           z_smoother_alpha=0.9,
                                           z_smoother_buffer_len=3)

# Upscale the video frames
if upscale_images:
    esrgan.inference_realesrgan(input=generated_video_frames_path,
                                output_images_path=upscaled_video_frames_path,
                                face_enhance=face_enhance,
                                purge_existing_files=True,
                                # model_filename='RealESRGAN_x4plus_anime_6B.pth',
                                # model_url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
                                netscale=4,
                                outscale=4)
    # copy PNG metadata from generated images to upscaled images
    # This function is unreasonably slow for many files. If you plan to keep the generated images, uncommend the lines below.
    # VF.copy_PNG_metadata(generated_video_frames_path, upscaled_video_frames_path)
    video_frames_to_encode = upscaled_video_frames_path
else:
    video_frames_to_encode = generated_video_frames_path


# Use a wrapper for FFMPEG to encode the video. Try setting video_encode_framerate=video_exraction_framerate/2 for a slow motion look.
generated_video_no_audio = f'{output_root_dir}{os.sep}encoded video.mp4'
video_tools.encode_video(output_file=generated_video_no_audio,
                         path_to_stills=video_frames_to_encode,
                         metadata_title=text_prompts,
                         metadata_comment=metadata_comment,
                         input_framerate=video_exraction_framerate)

if copy_audio:
    # Copy audio from the original file
    video_tools.copy_video_audio(
        input_video_path, generated_video_no_audio, final_video_filename)
    os.remove(generated_video_no_audio)
else:
    os.rename(generated_video_no_audio, final_video_filename)

if interpolate_with_RIFE:
    video_tools.RIFE_interpolation(input=final_video_filename,
                       output=f'{os.path.splitext(final_video_filename)[0]}_RIFE.mp4',
                       interpolation_factor=4,
                       metadata_title=text_prompts,
                       metadata_comment=metadata_comment)


