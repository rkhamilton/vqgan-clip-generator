# Generate a video with movement. Every frame that is generated has a shift or zoom applied to it.
# This gives the appearance of motion in the result. These videos do not stabilize.
# This example uses the ESRGAN upscaler before encoding the video.

from vqgan_clip import generate, video_tools, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
import os, subprocess

#Let's generate a single image to initialize the video.
config = VQGAN_CLIP_Config()
config.output_image_size = [256,144]
text_prompts = 'A pastoral landscape painting by Rembrandt^A black dog with red eyes in a cave^Apple pie'
num_video_frames = 150
video_framerate = 30
final_video_filename = os.path.join('example_media','zoom_video.mp4')
# Set True if you installed the Real-ESRGAN package for upscaling
upscale_images = True
face_enhance = False
# Set True if you installed the RIFE package for optical flow interpolation
# IMPORTANT - OF will increase the framerate by 4x (-exp=2 option) or 16x (-exp=4). Keep this in mind as you generate your VQGAN video.
# Suggested video_framerate 15 or 30 with 4x interpolation.
RIFE_OF_interpolation = True

# set some paths
generated_video_frames_path='video_frames'
init_image = os.path.join('example_media','init_image.png')
upscaled_video_frames_path='upscaled_video_frames'

generate.single_image(eng_config = config,
        text_prompts = text_prompts,
        iterations = 100,
        save_every = None,
        output_filename = init_image)
        
# Now generate a zoom video starting from that initial frame.
metadata_comment = generate.video_frames(num_video_frames=num_video_frames,
        eng_config = config,
        text_prompts = text_prompts,
        init_image = init_image,
        video_frames_path = generated_video_frames_path,
        iterations_per_frame = 30,
        change_prompts_on_frame= [60, 100],
        zoom_scale=1.02, 
        shift_x=1, 
        shift_y=1,
        z_smoother=True,
        z_smoother_buffer_len=5,
        z_smoother_alpha=0.9)

# Upscale the video frames
if upscale_images:
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
        input_framerate=video_framerate)

print(f'generation parameters:\n{metadata_comment}')

if RIFE_OF_interpolation:
        # This section runs RIFE optical flow interpolation and then compresses the resulting (uncompressed) video to h264 format.
        RIFE_interpolation_factor = 4 #Valid choices are 4 or 16
        of_cmnd = f'python arXiv2020-RIFE\\inference_video.py --exp={2 if RIFE_interpolation_factor==4 else 4} --model=arXiv2020-RIFE\\train_log --video={final_video_filename}'
        subprocess.Popen(of_cmnd,shell=True).wait()
        print(f'RIFE optical flow command used was:\n{of_cmnd}')
        metadata_option = f'-metadata title=\"{text_prompts}\" -metadata comment=\"{metadata_comment}\" -metadata description=\"Generated with https://github.com/rkhamilton/vqgan-clip-generator\"'
        # RIFE appends a string to the original filename of the form "original_filename_4X_120fps.mp4"
        RIFE_output_filename = os.path.splitext(final_video_filename)[0] + f'_{RIFE_interpolation_factor}X_{video_framerate*RIFE_interpolation_factor}fps.mp4'
        FFMPEG_output_filename = os.path.splitext(RIFE_output_filename)[0] + '_reencoded.mp4'
        ffmpeg_command = f'ffmpeg -y -i {RIFE_output_filename} -vcodec libx264 -crf 23 -pix_fmt yuv420p -hide_banner -loglevel error {metadata_option} {FFMPEG_output_filename}'
        subprocess.Popen(ffmpeg_command,shell=True).wait()
        print(f'FFMPEG command used was:\t{ffmpeg_command}')
