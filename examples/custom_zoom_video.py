# This file is an example of how you might pull out a method from vqgan_clip.generate.py and turn it into a script.
# With the script exposed like this you could experiment with many different elements of image generation. For example
# you could change it so that prompt weights change with each iteration, thereby giving you a smoother style transition.
# You could vary the config.learning_rate, so that the amount of change from frame-to-frame varies.
# T
from vqgan_clip.engine import Engine, VQGAN_CLIP_Config
from vqgan_clip import video_tools
from tqdm import tqdm
import os
from PIL import ImageFile, ImageChops
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import functional as TF
from vqgan_clip import _functional as VF

# EXAMPLE HERE: you can set your parameters up to change with each frame of video.
def parameters_by_frame(frame_num):
        shift_x = 0 if frame_num < 100 else 1
        shift_y = 0 if frame_num < 100 else 1
        zoom_scale = 1.0 if frame_num < 200 else 1.02
        return shift_x, shift_y, zoom_scale

eng_config=VQGAN_CLIP_Config()
eng_config.output_image_size = [587,330]
text_prompts = 'Impressionist painting of a red horse'
image_prompts = []
noise_prompts = []
init_image = None
iterations = 1000
save_every = 5
change_prompt_every = 0
video_frames_path='./video_frames'
"""
* eng_config (VQGAN_CLIP_Config, optional): An instance of VQGAN_CLIP_Config with attributes customized for your use. See the documentation for VQGAN_CLIP_Config().
* text_prompts (str, optional) : Text that will be turned into a prompt via CLIP. Default = []  
* image_prompts (str, optional) : Path to image that will be turned into a prompt via CLIP. Default = []
* noise_prompts (str, optional) : Random number seeds can be used as prompts using the same format as a text prompt. E.g. \'123:0.1|234:0.2|345:0.3\' Stories (^) are supported. Default = []
* init_image (str, optional) : Path to an image file that will be used as the seed to generate output (analyzed for pixels).
* iterations (int, optional) : Number of iterations of train() to perform before stopping. Default = 100 
* save_every (int, optional) : An interim image will be saved to the output location every save_every iterations, and training stats will be displayed. Default = 50  
* change_prompt_every (int, optional) : Serial prompts, sepated by ^, will be cycled through every change_prompt_every iterations. Prompts will loop if more cycles are requested than there are prompts. Default = 0
* video_frames_path (str, optional) : Path where still images should be saved as they are generated before being combined into a video. Defaults to './video_frames'.
* zoom_scale (float) : Every save_every iterations, a video frame is saved. That frame is shifted scaled by a factor of zoom_scale, and used as the initial image to generate the next frame. Default = 1.0
* shift_x (int) : Every save_every iterations, a video frame is saved. That frame is shifted shift_x pixels in the x direction, and used as the initial image to generate the next frame. Default = 0
* shift_y (int) : Every save_every iterations, a video frame is saved. That frame is shifted shift_y pixels in the y direction, and used as the initial image to generate the next frame. Default = 0
"""
if init_image:
    eng_config.init_image = init_image
parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts = VF.parse_all_prompts(text_prompts, image_prompts, noise_prompts)
eng = Engine(eng_config)
eng.initialize_VQGAN_CLIP()
eng.encode_and_append_prompts(0, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)
eng.configure_optimizer()

# if the location for the interim video frames doesn't exist, create it
if not os.path.exists(video_frames_path):
    os.mkdir(video_frames_path)
else:
    VF.delete_files(video_frames_path)

# generate images
current_prompt_number = 0
video_frame_num = 1
output_image_size_x, output_image_size_y = eng.calculate_output_image_size()
try:
    for iteration_num in tqdm(range(1,iterations+1),unit='iteration',desc='zoom video'):
        #perform iterations of train()
        lossAll = eng.train(iteration_num)

        if change_prompt_every and iteration_num % change_prompt_every == 0:
            # change prompts if every change_prompt_every iterations
            current_prompt_number += 1
            eng.clear_all_prompts()
            eng.encode_and_append_prompts(current_prompt_number, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)

        if save_every and iteration_num % save_every == 0:
            # EXAMPLE HERE: you can set your parameters up to change with each frame of video.
            shift_x, shift_y, zoom_scale = parameters_by_frame(video_frame_num)

            # Transform the current video frame
            # Convert z back into a Pil image 
            pil_image = TF.to_pil_image(eng.output_tensor[0].cpu())
                                    
            # Zoom
            if zoom_scale != 1.0:
                new_pil_image = VF.zoom_at(pil_image, output_image_size_x/2, output_image_size_y/2, zoom_scale)
            else:
                new_pil_image = pil_image
            
            # Shift
            if shift_x or shift_y:
                # This one wraps the image
                new_pil_image = ImageChops.offset(new_pil_image, shift_x, shift_y)
            
            # Re-encode and use this as the new initial image for the next iteration
            eng.convert_image_to_init_image(new_pil_image)

            # Re-create optimiser with the new initial image
            eng.configure_optimizer()

            # save a frame of video every .save_every iterations
            # display some statistics about how the GAN training is going whever we save an interim image
            losses_str = ', '.join(f'{loss.item():7.3f}' for loss in lossAll)
            tqdm.write(f'iteration:{iteration_num:6d}\tvideo frame: {video_frame_num:6d}\tloss sum: {sum(lossAll).item():7.3f}\tloss for each prompt:{losses_str}')

            # if making a video, save a frame named for the video step
            filepath_to_save = os.path.join(video_frames_path,f'frame_{video_frame_num:012d}.png')
            eng.save_current_output(filepath_to_save)
            video_frame_num += 1
except KeyboardInterrupt:
    pass

# Encode the images into a video
video_tools.encode_video(output_file=os.path.join('output','custom_zoom_video.mp4'),
        path_to_stills=video_frames_path,
        metadata=text_prompts,
        output_framerate=30,
        input_framerate=30,
        vcodec='libx264',
        crf=23)

