# This is an example of using style_transfer to apply VQGAN styling to images extracted from a video.
# Key parameters to experiment with are iterations_per_frame, current_source_frame_image_weight, and current_source_frame_prompt_weight

from vqgan_clip import generate, video_tools, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
import os
import subprocess
from vqgan_clip import _functional as VF

config = VQGAN_CLIP_Config()
config.output_image_size = [256, 256]
text_prompts = 'portrait on deviantart'
input_video_path = '20211004_132008000_iOS.MOV'
output_root_dir = 'example_media'
# Generated video framerate. Images will be extracted from the source video at this framerate, using interpolation if needed.
video_exraction_framerate = 30
final_video_filename = os.path.join('\"'+output_root_dir, 'portrait.mp4\"')
# Set True if you want to copy audio from the original video to the output
copy_audio = True
# Set True if you installed the Real-ESRGAN package for upscaling. face_enhance is a feature of Real-ESRGAN.
upscale_images = True
face_enhance = False
# Set True if you installed the RIFE package for optical flow interpolation
# IMPORTANT - OF will increase the framerate by 4x (-exp=2 option) or 16x (-exp=4). Keep this in mind as you generate your VQGAN video.
# Suggested video_framerate 15 or 30 with 4x interpolation.
RIFE_interpolation = True

# Set some paths
generated_video_frames_path = os.path.join(output_root_dir, 'video_frames')
init_image = os.path.join(output_root_dir, 'init_image.png')
upscaled_video_frames_path = os.path.join(
    output_root_dir, 'upscaled_video_frames')

# Use a wrapper for FFMPEG to extract stills from the original video.
original_video_frames = video_tools.extract_video_frames(input_video_path,
                                                         extraction_framerate=video_exraction_framerate)

# Let's generate a single image to initialize the video. Otherwise it takes a few frames for the new video to stabilize on the generated imagery.
generate.image(eng_config=config,
               text_prompts=text_prompts,
               init_image=original_video_frames[0],
               iterations=5,
               output_filename=init_image,
               leave_progress_bar=False)

# Apply a style to the extracted video frames.
metadata_comment = generate.style_transfer(original_video_frames,
                                           eng_config=config,
                                           # effects how closely the output video will be to the source video. Values of 0.1-3.0 are reasonable.
                                           current_source_frame_image_weight=5.0,
                                           current_source_frame_prompt_weight=0.0,
                                           text_prompts=text_prompts,
                                           iterations_per_frame=5,
                                           change_prompts_on_frame=[],
                                           init_image=init_image,
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
    VF.copy_PNG_metadata(generated_video_frames_path,
                         upscaled_video_frames_path)
    video_frames_to_encode = upscaled_video_frames_path
else:
    video_frames_to_encode = generated_video_frames_path


# Use a wrapper for FFMPEG to encode the video. Try setting video_encode_framerate=video_exraction_framerate/2 for a slow motion look.
generated_video_no_audio = os.path.join('\"'+output_root_dir, 'encoded.mp4\"')
video_tools.encode_video(output_file=generated_video_no_audio,
                         path_to_stills=video_frames_to_encode,
                         metadata_title=text_prompts,
                         metadata_comment=metadata_comment,
                         input_framerate=video_exraction_framerate)

if RIFE_interpolation:
    # This section runs RIFE optical flow interpolation and then compresses the resulting (uncompressed) video to h264 format.
    RIFE_interpolation_factor = 4  # Valid choices are 4 or 16
    of_cmnd = f'python arXiv2020-RIFE\\inference_video.py --exp={2 if RIFE_interpolation_factor==4 else 4} --model=arXiv2020-RIFE\\train_log --video={generated_video_no_audio}'
    subprocess.Popen(of_cmnd, shell=True).wait()
    print(f'RIFE optical flow command used was:\n{of_cmnd}')

    # Re-encode the RIFE output to a compressed format
    metadata_option = f'-metadata title=\"{text_prompts}\" -metadata comment=\"{metadata_comment}\" -metadata description=\"Generated with https://github.com/rkhamilton/vqgan-clip-generator\"'
    # RIFE appends a string to the original filename of the form "original_filename_4X_120fps.mp4"
    RIFE_output_filename = os.path.splitext(generated_video_no_audio)[
        0] + f'_{RIFE_interpolation_factor}X_{int(video_exraction_framerate*RIFE_interpolation_factor)}fps.mp4\"'
    FFMPEG_output_filename = os.path.splitext(RIFE_output_filename)[
        0] + '_reencoded.mp4\"'
    ffmpeg_command = f'ffmpeg -y -i {RIFE_output_filename} -vcodec libx264 -crf 23 -pix_fmt yuv420p -hide_banner -loglevel error {metadata_option} {FFMPEG_output_filename}'
    subprocess.Popen(ffmpeg_command, shell=True).wait()
    print(f'FFMPEG command used was:\t{ffmpeg_command}')
    generated_video_no_audio = FFMPEG_output_filename

if copy_audio:
    # Copy audio from the original file
    video_tools.copy_video_audio(
        input_video_path, generated_video_no_audio, final_video_filename)
    os.remove(generated_video_no_audio)
else:
    os.rename(generated_video_no_audio, final_video_filename)
