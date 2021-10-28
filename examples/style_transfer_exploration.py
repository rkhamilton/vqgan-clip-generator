# This is an example of how to change parameters in style transfers to explore the effect of different parameters
# This will not encode a video, but instead save a single frame of video from a configurable number of frames into the output.
# You can generate many such still (e.g. frame 30) with different combinations of settings.
# There are interactions between init_image_methods, current_source_frame_image_weights, and iterations_per_frames
# The configuration in this example is a useful set of initial images to evaluate. Pick your favorite combination and use that
# to generate your style transfer video.

from vqgan_clip import generate, video_tools, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
import os
import glob
from vqgan_clip import _functional as VF
import itertools
import shutil
from tqdm import tqdm

config = VQGAN_CLIP_Config()
config.seed = 1
config.cudnn_determinism = True
config.output_image_size = [256, 256]
text_prompts = 'portrait on deviantart'
input_video_path = '20211004_132008000_iOS.MOV'
# all folders will be created within the output_root_dir
output_root_dir = 'example_media'
# Generated video framerate. Images will be extracted from the source video at this framerate, using interpolation if needed.
video_framerate = 30
# number of frames of video to process before stopping. If using z_smoothing, suggest at least 10 frames.
frames_to_process = 10

# Set some paths
generated_video_frames_path = os.path.join(output_root_dir, 'video_frames')
final_output_images_path = os.path.join(output_root_dir, 'parameter_tests')

# ensure the output folder exists and is empty.
os.makedirs(final_output_images_path, exist_ok=True)
for f in glob.glob(os.path.join(final_output_images_path, '*.png')):
    os.remove(f)

# Use a wrapper for FFMPEG to extract stills from the original video.
original_video_frames = video_tools.extract_video_frames(input_video_path,
                                                         extraction_framerate=video_framerate)

# Truncate to only the desired number of frames.
del original_video_frames[frames_to_process:]

# set the parameters below to lists of values that you would like to explore. All combinations will be tested.
# For parameters you want to stay fixed, use lists with one element, e.g. [0.2]
current_source_frame_image_weights = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
iterations_per_frames = [15]  # [5, 10, 15, 30, 60, 100]
current_source_frame_prompt_weights = [0]  # [0.0, 0.1, 0.5, 0.8, 1.5]
z_smoother_buffer_lens = [3]  # [3,5,7]
z_smoother_alphas = [0.9]  # [0.6, 0.7, 0.8, 0.9]
init_image_methods = ['alternate_img_target', 'alternate_img_target_decay']
total_iterables = len(current_source_frame_image_weights)*len(iterations_per_frames)*len(
    current_source_frame_prompt_weights)*len(z_smoother_buffer_lens)*len(z_smoother_alphas)*len(init_image_methods)
all_iterables = tqdm(itertools.product(init_image_methods, current_source_frame_image_weights, iterations_per_frames, current_source_frame_prompt_weights, z_smoother_buffer_lens, z_smoother_alphas),
                     total=total_iterables, unit='combo', desc='parameter combinations')

for init_image_method, current_source_frame_image_weight, iterations_per_frame, current_source_frame_prompt_weight, z_smoother_buffer_len, z_smoother_alpha in all_iterables:
    # Apply a style to the extracted video frames.
    config.init_image_method = init_image_method
    metadata_comment = generate.style_transfer(original_video_frames,
                                               eng_config=config,
                                               # effects how closely the output video will be to the source video. Values of 0.5-8.0 are reasonable.
                                               current_source_frame_image_weight=current_source_frame_image_weight,
                                               text_prompts=text_prompts,
                                               iterations_per_frame=iterations_per_frame,
                                               generated_video_frames_path=generated_video_frames_path,
                                               current_source_frame_prompt_weight=current_source_frame_prompt_weight,
                                               z_smoother=True,
                                               z_smoother_alpha=z_smoother_alpha,
                                               z_smoother_buffer_len=z_smoother_buffer_len,
                                               leave_progress_bar=False)

    # save the last generated image with a descriptive filename
    final_output_filename = os.path.join(
        final_output_images_path, f'{init_image_method}_image_weight_{current_source_frame_image_weight:1.1f}_prompt_weight_{current_source_frame_prompt_weight:1.1f}_iterations_{iterations_per_frame}_buf_len_{z_smoother_buffer_len}_alpha_{z_smoother_alpha:1.1f}.png')
    generated_files = glob.glob(os.path.join(
        generated_video_frames_path, '*.png'))
    shutil.copy(generated_files[-1], final_output_filename)
