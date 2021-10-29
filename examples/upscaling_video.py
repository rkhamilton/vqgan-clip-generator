# This example demonstrates upscaling an existing video using the tools provided in this package.
# This is not a primary use case for the package, but it may be useful. It also illustrates how you
# may prefer to integrate shell commands to ffmpeg into your video workflows, rather than using
# generate.video_tools methods.
# Note that Real-ESRGAN will run out of VRAM if you try to upscale a large video.

from vqgan_clip import esrgan, video_tools
import os

input_video_path = 'small_video.mp4'
output_root_dir = 'example_media'
final_output_filename = f'{output_root_dir}{os.sep}upscaled video.mp4'
extracted_video_frames_path = f'{output_root_dir}{os.sep}video frames'
upscaled_video_frames_path = f'{output_root_dir}{os.sep}upscaled video frames'
extraction_framerate = 30


# Use a wrapper for FFMPEG to extract stills from the original video.
original_video_frames = video_tools.extract_video_frames(input_video_path,
                                                         extraction_framerate=extraction_framerate,
                                                         extracted_video_frames_path=extracted_video_frames_path)

# This is equivalent to
# os.system(f'ffmpeg -i small_video.mp4 -filter:v fps=30 video_frames\\frame_%12d.jpg')

# Upscale using Real-ESRGAN
esrgan.inference_realesrgan(input=extracted_video_frames_path,
                            output_images_path=upscaled_video_frames_path,
                            face_enhance=False,
                            # Careful! This will delete everything in the output_images_path!
                            purge_existing_files=True,
                            netscale=4,
                            outscale=4)

# Encode the video.
generated_video_no_audio = f'{output_root_dir}{os.sep}output no audio.mp4'
ffmpeg_input_path = f'\"{upscaled_video_frames_path}\\frame_%12d.png\"'
os.system(
    f'ffmpeg -y -f image2 -i {ffmpeg_input_path} -r 30 -vcodec libx264 -crf 23 -pix_fmt yuv420p -strict -2 \"{generated_video_no_audio}\"')

# Copy audio from the original file
video_tools.copy_video_audio(
    input_video_path, generated_video_no_audio, final_output_filename)
os.remove(generated_video_no_audio)

# This is equiavalent to
# os.system(f'ffmpeg -i small_video.mp4 -vn -acodec copy extracted_original_audio.aac')
# os.system(f'ffmpeg -i output_no_audio.mp4 -i extracted_original_audio.aac -c copy -map 0:v:0 -map 1:a:0 upscaled_video.mp4')
