# This module is the interface for creating images and video from text prompts
# This should also serve as examples of how you can use the Engine class to create images and video using your own creativity.
# Feel free to extract the contents of these methods and use them to build your own sequences. 
# Change the image prompt weights over time
# Change the interval at which video frames are exported over time, to create the effect of speeding or slowing video
# Change the engine learning rate to increase or decrease the amount of change for each frame
# Create style transfer videos where each frame uses many image prompts, or many previous frames as image prompts.
# Create a zoom video where the shift_x and shift_x are functions of iteration to create spiraling zooms
# It's art. Go nuts!

from vqgan_clip.engine import Engine, VQGAN_CLIP_Config
from vqgan_clip.z_smoother import Z_Smoother
from tqdm.auto import tqdm
import os
import contextlib
import torch
import warnings
from PIL import ImageFile, Image, ImageChops, PngImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import functional as TF
from vqgan_clip import _functional as VF

def image(output_filename,
        eng_config = VQGAN_CLIP_Config(),
        text_prompts = [],
        image_prompts = [],
        noise_prompts = [],
        init_image = None,
        iterations = 100,
        save_every = None,
        verbose = False,
        leave_progress_bar = True):
    """Generate a single image using VQGAN+CLIP. The configuration of the algorithms is done via a VQGAN_CLIP_Config instance.

    Args:
        * output_filename (str) : location to save the output image. Omit the file extension. 
        * eng_config (VQGAN_CLIP_Config, optional): An instance of VQGAN_CLIP_Config with attributes customized for your use. See the documentation for VQGAN_CLIP_Config().
        * text_prompts (str, optional) : Text that will be turned into a prompt via CLIP. Default = []  
        * image_prompts (str, optional) : Path to image that will be turned into a prompt via CLIP (analyzed for content). Default = []
        * noise_prompts (str, optional) : Random number seeds can be used as prompts using the same format as a text prompt. E.g. \'123:0.1|234:0.2|345:0.3\' Stories (^) are supported. Default = []
        * init_image (str, optional) : Path to an image file that will be used as the seed to generate output (analyzed for pixels).
        * iterations (int, optional) : Number of iterations of train() to perform before stopping. Default = 100 
        * save_every (int, optional) : An interim image will be saved as the final image is being generated. It's saved to the output location every save_every iterations, and training stats will be displayed. Default = None  
        * verbose (boolean, optional) : When true, prints diagnostic data every time a video frame is saved. Defaults to False.
        * leave_progress_bar (boolean, optional) : When False, the tqdm progress bar will disappear when the work is completed. Useful for nested loops.
    """
    # output_filename = _filename_to_jpg(output_filename)
    output_folder_name = os.path.dirname(output_filename)
    if output_folder_name:
        os.makedirs(output_folder_name, exist_ok=True)

    if init_image:
        eng_config.init_image = init_image
        output_size_X, output_size_Y = VF.filesize_matching_aspect_ratio(init_image, eng_config.output_image_size[0], eng_config.output_image_size[1])
        eng_config.output_image_size = [output_size_X, output_size_Y]

    # suppress stdout to keep the progress bar clear
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            eng = Engine(eng_config)
            eng.initialize_VQGAN_CLIP()
    parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts = VF.parse_all_prompts(text_prompts, image_prompts, noise_prompts)
    eng.encode_and_append_prompts(0, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)
    eng.configure_optimizer()
    # metadata to save to jpge file as data chunks
    img_info =  [('text_prompts',text_prompts),
            ('image_prompts',image_prompts),
            ('noise_prompts',noise_prompts),
            ('iterations',iterations),
            ('init_image',init_image),
            ('save_every',save_every),
            ('cut_method',eng_config.cut_method),
            ('seed',eng.conf.seed)]

    # generate the image
    try:
        for iteration_num in tqdm(range(1,iterations+1),unit='iteration',desc='single image',leave=leave_progress_bar):
            #perform iterations of train()
            lossAll = eng.train(iteration_num)
           
            if save_every and iteration_num % save_every == 0:
                if verbose:
                    # display some statistics about how the GAN training is going whever we save an interim image
                    losses_str = ', '.join(f'{loss.item():7.3f}' for loss in lossAll)
                    tqdm.write(f'iteration:{iteration_num:6d}\tloss sum: {sum(lossAll).item():7.3f}\tloss for each prompt:{losses_str}')
                # save an interim copy of the image so you can look at it as it changes if you like
                eng.save_current_output(output_filename,img_info) 

        # Always save the output at the end
        eng.save_current_output(output_filename,img_info) 
    except KeyboardInterrupt:
        pass

    config_info=f'iterations: {iterations}, '\
            f'image_prompts: {image_prompts}, '\
            f'noise_prompts: {noise_prompts}, '\
            f'init_weight_method: {eng_config.init_image_method}, '\
            f'init_weight {eng_config.init_weight:1.2f}, '\
            f'init_image {init_image}, '\
            f'cut_method {eng_config.cut_method}, '\
            f'seed {eng.conf.seed}'
    return config_info

def restyle_video_frames(video_frames,
    eng_config=VQGAN_CLIP_Config(),
    text_prompts = 'Covered in spiders | Surreal:0.5',
    image_prompts = [],
    noise_prompts = [],
    iterations = 15,
    save_every = None,
    generated_video_frames_path='./video_frames',
    current_source_frame_prompt_weight=0.0,
    previous_generated_frame_prompt_weight=0.0,
    generated_frame_init_blend=0.2,
    z_smoother=False,
    z_smoother_buffer_len=3,
    z_smoother_alpha=0.6):
    """DEPRECATED. See generate.style_transfer().  


    Apply a style to an existing video using VQGAN+CLIP using a blended input frame method. The still image 
    frames from the original video are extracted, and used as initial images for VQGAN+CLIP. The resulting 
    folder of stills are then encoded into an HEVC video file. The audio from the original may optionally be 
    transferred. The configuration of the VQGAN+CLIP algorithms is done via a VQGAN_CLIP_Config instance. 
    Unlike restyle_video, in restyle_video_blended each new frame of video is initialized using a blend of the 
    new source frame and the old *generated* frame. This results in an output video that transitions much more
    smoothly between frames. Using the method parameter current_frame_prompt_weight lets you decide how much 
    of the new source frame to use versus the previous generated frame.

    It is suggested to also use a config.init_weight > 0 so that the resulting generated video will look more
    like the original video frames.

    Args:
        Args:
        * video_frames (list of str) : List of paths to the video frames that will be restyled.
        * eng_config (VQGAN_CLIP_Config, optional): An instance of VQGAN_CLIP_Config with attributes customized for your use. See the documentation for VQGAN_CLIP_Config().
        * text_prompts (str, optional) : Text that will be turned into a prompt via CLIP. Default = []  
        * image_prompts (str, optional) : Path to image that will be turned into a prompt via CLIP. Default = []
        * noise_prompts (str, optional) : Random number seeds can be used as prompts using the same format as a text prompt. E.g. \'123:0.1|234:0.2|345:0.3\' Stories (^) are supported. Default = []
        * iterations (int, optional) : Number of iterations of train() to perform for each frame of video. Default = 15 
        * save_every (int, optional) : An interim image will be saved as the final image is being generated. It's saved to the output location every save_every iterations, and training stats will be displayed. Default = 50  
        * generated_video_frames_path (str, optional) : Path where still images should be saved as they are generated before being combined into a video. Defaults to './video_frames'.
        * current_frame_prompt_weight (float) : Using the current frame of source video as an image prompt (as well as init_image), this assigns a weight to that image prompt. Default = 0.0
        * generated_frame_init_blend (float) : How much of the previous generated image to blend in to a new frame's init_image. 0 means no previous generated image, 1 means 100% previous generated image. Default = 0.2
        * z_smoother (boolean, optional) : If true, smooth the latent vectors (z) used for image generation by combining multiple z vectors through an exponentially weighted moving average (EWMA). Defaults to False.
        * z_smoother_buffer_len (int, optional) : How many images' latent vectors should be combined in the smoothing algorithm. Bigger numbers will be smoother, and have more blurred motion. Must be an odd number. Defaults to 3.
        * z_smoother_alpha (float, optional) : When combining multiple latent vectors for smoothing, this sets how important the "keyframe" z is. As frames move further from the keyframe, their weight drops by (1-z_smoother_alpha) each frame. Bigger numbers apply more smoothing. Defaults to 0.6.
"""
    warnings.warn('This function is deprecated and will be removed. Use generate.style_transfer() instead.')
    parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts = VF.parse_all_prompts(text_prompts, image_prompts, noise_prompts)
    # lock in a seed to use for each frame
    if not eng_config.seed:
        # note, retreiving torch.seed() also sets the torch seed
        eng_config.seed = torch.seed()

    # if the location for the generated video frames doesn't exist, create it
    if not os.path.exists(generated_video_frames_path):
        os.mkdir(generated_video_frames_path)
    else:
        VF.delete_files(generated_video_frames_path)

    output_size_X, output_size_Y = VF.filesize_matching_aspect_ratio(video_frames[0], eng_config.output_image_size[0], eng_config.output_image_size[1])
    eng_config.output_image_size = [output_size_X, output_size_Y]

    # suppress stdout to keep the progress bar clear
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            eng = Engine(eng_config)
            eng.initialize_VQGAN_CLIP()

    smoothed_z = Z_Smoother(buffer_len=z_smoother_buffer_len, alpha=z_smoother_alpha)
    # generate images
    video_frame_num = 1
    try:
        last_video_frame_generated = video_frames[0]
        video_frames_loop = tqdm(video_frames,unit='image',desc='style transfer')
        for video_frame in video_frames_loop:
            filename_to_save = os.path.basename(os.path.splitext(video_frame)[0]) + '.png'
            filepath_to_save = os.path.join(generated_video_frames_path,filename_to_save)

            # INIT IMAGE
            # By default, the init_image is the new frame of source video.
            pil_image_new_frame = Image.open(video_frame).convert('RGB').resize([output_size_X,output_size_Y], resample=Image.LANCZOS)
            # Blend the new original frame with the most recent generated frame. Use that as the initial image for the upcoming frame.
            if generated_frame_init_blend:
                # open the last frame of generated video
                pil_image_previous_generated_frame = Image.open(last_video_frame_generated).convert('RGB').resize([output_size_X,output_size_Y], resample=Image.LANCZOS)
                pil_init_image = Image.blend(pil_image_new_frame,pil_image_previous_generated_frame,generated_frame_init_blend)
            else:
                pil_init_image = pil_image_new_frame
            eng.convert_image_to_init_image(pil_init_image)

            # Optionally use the current source video frame, and the previous generate frames, as input prompts
            eng.clear_all_prompts()
            current_prompt_number = 0
            eng.encode_and_append_prompts(current_prompt_number, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)
            if current_source_frame_prompt_weight:
                eng.encode_and_append_pil_image(pil_image_new_frame, weight=current_source_frame_prompt_weight)
            if previous_generated_frame_prompt_weight:
                eng.encode_and_append_pil_image(pil_image_previous_generated_frame, weight=previous_generated_frame_prompt_weight)

            # Setup for this frame is complete. Configure the optimizer for this z.
            eng.configure_optimizer()

            # Generate a new image
            for iteration_num in range(1,iterations+1):
                #perform iterations of train()
                lossAll = eng.train(iteration_num)
                # TODO reimplement save_every
                # if change_prompt_every and iteration_num % change_prompt_every == 0:
                #     # change prompts if every change_prompt_every iterations
                #     current_prompt_number += 1
                #     eng.clear_all_prompts()
                #     eng.encode_and_append_prompts(current_prompt_number, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)
                #     # eng.encode_and_append_pil_image(pil_image_new_frame, weight=current_frame_prompt_weight)
                #     eng.configure_optimizer()

                if save_every and iteration_num % save_every == 0:
                    # save a frame of video every .save_every iterations
                    losses_str = ', '.join(f'{loss.item():7.3f}' for loss in lossAll)
                    tqdm.write(f'iteration:{iteration_num:6d}\tvideo frame: {video_frame_num:6d}\tloss sum: {sum(lossAll).item():7.3f}\tloss for each prompt:{losses_str}')
                    eng.save_current_output(filepath_to_save)

            # save a frame of video every iterations
            # display some statistics about how the GAN training is going whever we save an image
            losses_str = ', '.join(f'{loss.item():7.3f}' for loss in lossAll)
            tqdm.write(f'iteration:{iteration_num:6d}\tvideo frame: {video_frame_num:6d}\tloss sum: {sum(lossAll).item():7.3f}\tloss for each prompt:{losses_str}')

            # metadata to save to PNG file as data chunks
            png_info =  [('text_prompts',text_prompts),
                ('image_prompts',image_prompts),
                ('noise_prompts',noise_prompts),
                ('iterations',iterations),
                ('init_image',video_frame),
                ('save_every',save_every),
                ('change_prompt_every','N/A'),
                ('seed',eng.conf.seed),
                ('current_source_frame_prompt_weight',f'{current_source_frame_prompt_weight:2.2f}'),
                ('previous_generated_frame_prompt_weight',f'{previous_generated_frame_prompt_weight:2.2f}'),
                ('generated_frame_init_blend',f'{generated_frame_init_blend:2.2f}')]
            # if making a video, save a frame named for the video step
            if z_smoother:
                smoothed_z.append(eng._z.clone())
                output_tensor = eng.synth(smoothed_z._mid_ewma())
                Engine.save_tensor_as_image(output_tensor,filepath_to_save,VF.png_info_chunks(png_info))
            else:
                eng.save_current_output(filepath_to_save,VF.png_info_chunks(png_info))
            last_video_frame_generated = filepath_to_save
            video_frame_num += 1
    except KeyboardInterrupt:
        pass
    config_info=f'iterations: {iterations}, '\
            f'image_prompts: {image_prompts}, '\
            f'noise_prompts: {noise_prompts}, '\
            f'init_weight_method: {eng_config.init_image_method}, '\
            f'init_weight {eng_config.init_weight:1.2f}, '\
            f'init_image {generated_video_frames_path}, '\
            f'cut_method {eng_config.cut_method}, '\
            f'current_source_frame_prompt_weight {current_source_frame_prompt_weight:2.2f}, '\
            f'previous_generated_frame_prompt_weight {previous_generated_frame_prompt_weight:2.2f}, '\
            f'generated_frame_init_blend {generated_frame_init_blend:2.2f}, '\
            f'seed {eng.conf.seed}'
    return config_info

def video_frames(num_video_frames,
        iterations_per_frame = 30,
        iterations_for_first_frame = 100,
        eng_config=VQGAN_CLIP_Config(),
        text_prompts = [],
        image_prompts = [],
        noise_prompts = [],
        change_prompts_on_frame = None,
        init_image = None,
        generated_video_frames_path='./video_frames',
        zoom_scale=1.0,
        shift_x=0, 
        shift_y=0,
        z_smoother=False,
        z_smoother_buffer_len=5,
        z_smoother_alpha=0.9,
        verbose=False,
        leave_progress_bar = True):
    """Generate a series of PNG-formatted images using VQGAN+CLIP where each image is related to the previous image so they can be combined into a video. 
    The configuration of the VQGAN+CLIP algorithms is done via a VQGAN_CLIP_Config instance.

    Args:
        * num_video_frames (int) : Number of video frames to be generated.  
        * iterations_per_frame (int, optional) : Number of iterations of train() to perform on each generated video frame. Default = 30
        * iterations_for_first_frame (int, optional) : Number of additional iterations of train() to perform on the first frame so that the image is not a gray/random field. Default = 30
        * eng_config (VQGAN_CLIP_Config, optional): An instance of VQGAN_CLIP_Config with attributes customized for your use. See the documentation for VQGAN_CLIP_Config().
        * text_prompts (str, optional) : Text that will be turned into a prompt via CLIP. Default = []  
        * image_prompts (str, optional) : Path to image that will be turned into a prompt via CLIP. Default = []
        * noise_prompts (str, optional) : Random number seeds can be used as prompts using the same format as a text prompt. E.g. \'123:0.1|234:0.2|345:0.3\' Stories (^) are supported. Default = []
        * change_prompts_on_frame (list(int)) : All prompts (separated by "^") will be cycled forward on the video frames provided here. Defaults to None.
        * init_image (str, optional) : Path to an image file that will be used as the seed to generate output (analyzed for pixels).
        * video_frames_path (str, optional) : Path where still images should be saved as they are generated before being combined into a video. Defaults to './video_frames'.
        * zoom_scale (float) : Every save_every iterations, a video frame is saved. That frame is shifted scaled by a factor of zoom_scale, and used as the initial image to generate the next frame. Default = 1.0
        * shift_x (int, optional) : Every save_every iterations, a video frame is saved. That frame is shifted shift_x pixels in the x direction, and used as the initial image to generate the next frame. Default = 0
        * shift_y (int, optional) : Every save_every iterations, a video frame is saved. That frame is shifted shift_y pixels in the y direction, and used as the initial image to generate the next frame. Default = 0
        * z_smoother (boolean, optional) : If true, smooth the latent vectors (z) used for image generation by combining multiple z vectors through an exponentially weighted moving average (EWMA). Defaults to False.
        * z_smoother_buffer_len (int, optional) : How many images' latent vectors should be combined in the smoothing algorithm. Bigger numbers will be smoother, and have more blurred motion. Must be an odd number. Defaults to 3.
        * z_smoother_alpha (float, optional) : When combining multiple latent vectors for smoothing, this sets how important the "keyframe" z is. As frames move further from the keyframe, their weight drops by (1-z_smoother_alpha) each frame. Bigger numbers apply more smoothing. Defaults to 0.7.
        * verbose (boolean, optional) : When true, prints diagnostic data every time a video frame is saved. Defaults to False.
        * leave_progress_bar (boolean, optional) : When False, the tqdm progress bar will disappear when the work is completed. Useful for nested loops.
    """
    if zoom_scale != 1.0 or shift_x or shift_y:
        if iterations_per_frame < 10:
            warnings.warn('When using zoom_scale or shift_x/shift_y, iterations_per_frame should be above 10')
    if init_image:
        eng_config.init_image = init_image
    parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts = VF.parse_all_prompts(text_prompts, image_prompts, noise_prompts)
    # suppress stdout to keep the progress bar clear
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            eng = Engine(eng_config)
            eng.initialize_VQGAN_CLIP()
    current_prompt_number = 0
    eng.encode_and_append_prompts(current_prompt_number, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)
    eng.configure_optimizer()

    # if the location for the interim video frames doesn't exist, create it
    if not os.path.exists(generated_video_frames_path):
        os.mkdir(generated_video_frames_path)
    else:
        VF.delete_files(generated_video_frames_path)

    # Smooth the latent vector z with recent results. Maintain a list of recent latent vectors.
    smoothed_z = Z_Smoother(buffer_len=z_smoother_buffer_len, alpha=z_smoother_alpha)
    output_image_size_x, output_image_size_y = eng.calculate_output_image_size()
    # generate images
    try:
        # without an initial image, the first frame usually takes more iterations to converge away from a gray field.
        if not init_image and iterations_for_first_frame:
            for iteration_num in tqdm(range(iterations_for_first_frame),unit='iteration',desc='first frame',leave=False):
                lossAll = eng.train(iteration_num)

        # generate the video frames
        for video_frame_num in tqdm(range(1,num_video_frames+1),unit='frame',desc='video frames',leave=leave_progress_bar):
            for iteration_num in tqdm(range(iterations_per_frame),unit='iteration',desc='generating frame',leave=False):
                lossAll = eng.train(iteration_num)

            if change_prompts_on_frame is not None:
                if video_frame_num in change_prompts_on_frame:
                    # change prompts if the current frame number is in the list of change frames
                    current_prompt_number += 1
                    eng.clear_all_prompts()
                    eng.encode_and_append_prompts(current_prompt_number, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)

            # Zoom / shift the generated image
            if zoom_scale != 1.0 or shift_x or shift_y:
                pil_image = TF.to_pil_image(eng.output_tensor[0].cpu())
                if zoom_scale != 1.0:
                    new_pil_image = VF.zoom_at(pil_image, output_image_size_x/2, output_image_size_y/2, zoom_scale)
                else:
                    new_pil_image = pil_image

                if shift_x or shift_y:
                    new_pil_image = ImageChops.offset(new_pil_image, shift_x, shift_y)
                
                # Re-encode and use this as the new initial image for the next iteration
                eng.convert_image_to_init_image(new_pil_image)
                eng.configure_optimizer()

            if verbose:
                # display some statistics about how the GAN training is going whever we save an interim image
                losses_str = ', '.join(f'{loss.item():7.3f}' for loss in lossAll)
                tqdm.write(f'iteration:{iteration_num:6d}\tvideo frame: {video_frame_num:6d}\tloss sum: {sum(lossAll).item():7.3f}\tloss for each prompt:{losses_str}')

            # metadata to save to PNG file as data chunks
            img_info =  [('text_prompts',text_prompts),
                ('image_prompts',image_prompts),
                ('noise_prompts',noise_prompts),
                ('iterations',iterations_per_frame),
                ('init_image',video_frame_num),
                ('cut_method',eng_config.cut_method),
                ('seed',eng.conf.seed),
                ('zoom_scale',zoom_scale),
                ('shift_x',shift_x),
                ('shift_y',shift_y),
                ('z_smoother',z_smoother),
                ('z_smoother_buffer_len',z_smoother_buffer_len),
                ('z_smoother_alpha',z_smoother_alpha)]
            # if making a video, save a frame named for the video step
            filepath_to_save = os.path.join(generated_video_frames_path,f'frame_{video_frame_num:012d}.jpg')
            if z_smoother:
                smoothed_z.append(eng._z.clone())
                output_tensor = eng.synth(smoothed_z._mean())
                Engine.save_tensor_as_image(output_tensor,filepath_to_save,img_info)
            else:
                eng.save_current_output(filepath_to_save,img_info)

    except KeyboardInterrupt:
        pass
    # metadata to return so that it can be saved to the video file using e.g. ffmpeg.
    config_info=f'iterations: {iterations_per_frame}, '\
            f'image_prompts: {image_prompts}, '\
            f'noise_prompts: {noise_prompts}, '\
            f'init_weight_method: {eng_config.init_image_method}, '\
            f'init_weight {eng_config.init_weight:1.2f}, '\
            f'cut_method {eng_config.cut_method}, '\
            f'init_image {init_image}, '\
            f'seed {eng.conf.seed}, '\
            f'zoom_scale {zoom_scale}, '\
            f'shift_x {shift_x}, '\
            f'shift_y {shift_y}, '\
            f'z_smoother {z_smoother}, '\
            f'z_smoother_buffer_len {z_smoother_buffer_len}, '\
            f'z_smoother_alpha {z_smoother_alpha}'
    return config_info


def _filename_to_jpg(file_path):
    dir = os.path.dirname(file_path)
    filename_without_path = os.path.basename(file_path)
    basename_without_ext, ext = os.path.splitext(filename_without_path)
    if ext.lower() not in ['.jpg','']:
        warnings.warn('vqgan_clip_generator can only create and save .jpg files.')
    path_str = os.path.join(dir,basename_without_ext+'.jpg')
    return f'{path_str}'

def style_transfer(video_frames,
    eng_config=VQGAN_CLIP_Config(),
    text_prompts = 'Covered in spiders | Surreal:0.5',
    image_prompts = [],
    noise_prompts = [],
    iterations_per_frame = 15,
    iterations_for_first_frame = 15,
    current_source_frame_image_weight = 2.0,
    change_prompts_on_frame = None,
    generated_video_frames_path='./video_frames',
    current_source_frame_prompt_weight=0.0,
    z_smoother=False,
    z_smoother_buffer_len=3,
    z_smoother_alpha=0.7,
    verbose=False,
    leave_progress_bar = True):
    """Apply a style to existing video frames using VQGAN+CLIP.
    Set values of iteration_per_frame to determine how much the style transfer effect will be.
    Set values of source_frame_weight to determine how closely the result will match the source image. Balance iteration_per_frame and source_frame_weight to influence output.
    Set z_smoother to True to apply some latent-vector-based motion smoothing that will increase frame-to-frame consistency further at the cost of adding some motion blur.
    Set current_source_frame_prompt_weight >0 to have the generated content CLIP-match the source image.

    Args:
    * video_frames (list of str) : List of paths to the video frames that will be restyled.
    * eng_config (VQGAN_CLIP_Config, optional): An instance of VQGAN_CLIP_Config with attributes customized for your use. See the documentation for VQGAN_CLIP_Config().
    * text_prompts (str, optional) : Text that will be turned into a prompt via CLIP. Default = []  
    * image_prompts (str, optional) : Path to image that will be turned into a prompt via CLIP. Default = []
    * noise_prompts (str, optional) : Random number seeds can be used as prompts using the same format as a text prompt. E.g. \'123:0.1|234:0.2|345:0.3\' Stories (^) are supported. Default = []
    * change_prompts_on_frame (list(int)) : All prompts (separated by "^" will be cycled forward on the video frames provided here. Defaults to None.
    * iterations_per_frame (int, optional) : Number of iterations of train() to perform for each frame of video. Default = 15 
    * iterations_for_first_frame (int, optional) : Number of additional iterations of train() to perform on the first frame so that the image is not a gray/random field. Default = 30
    * generated_video_frames_path (str, optional) : Path where still images should be saved as they are generated before being combined into a video. Defaults to './video_frames'.
    * current_source_frame_image_weight (float) : Assigns a loss weight to make the output image look like the source image itself. Default = 0.0
    * current_source_frame_prompt_weight (float) : Assigns a loss weight to make the output image look like the CLIP representation of the source image. Default = 0.0
    * z_smoother (boolean, optional) : If true, smooth the latent vectors (z) used for image generation by combining multiple z vectors through an exponentially weighted moving average (EWMA). Defaults to False.
    * z_smoother_buffer_len (int, optional) : How many images' latent vectors should be combined in the smoothing algorithm. Bigger numbers will be smoother, and have more blurred motion. Must be an odd number. Defaults to 3.
    * z_smoother_alpha (float, optional) : When combining multiple latent vectors for smoothing, this sets how important the "keyframe" z is. As frames move further from the keyframe, their weight drops by (1-z_smoother_alpha) each frame. Bigger numbers apply more smoothing. Defaults to 0.6.
    * leave_progress_bar (boolean, optional) : When False, the tqdm progress bar will disappear when the work is completed. Useful for nested loops.
"""
    eng_config.init_weight = current_source_frame_image_weight

    # by default, run the first frame for the same number of iterations as the rest of the frames. It can be useful to use more though.
    if not iterations_for_first_frame:
        iterations_for_first_frame = iterations_per_frame

    output_size_X, output_size_Y = VF.filesize_matching_aspect_ratio(video_frames[0], eng_config.output_image_size[0], eng_config.output_image_size[1])
    eng_config.output_image_size = [output_size_X, output_size_Y]

    # Let's generate a single image to initialize the video. Otherwise it takes a few frames for the new video to stabilize on the generated imagery.
    init_image = 'init_image.jpg'
    eng_config_init_img = eng_config
    eng_config_init_img.init_image_method = 'original'
    image(output_filename=init_image,
        eng_config=eng_config_init_img,
        text_prompts=text_prompts,
        image_prompts = image_prompts,
        noise_prompts = noise_prompts,
        init_image = video_frames[0],
        iterations = iterations_for_first_frame,
        save_every = None,
        verbose = False,
        leave_progress_bar = False)

    parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts = VF.parse_all_prompts(text_prompts, image_prompts, noise_prompts)

    # lock in a seed to use for each frame
    if not eng_config.seed:
        # note, retreiving torch.seed() also sets the torch seed
        eng_config.seed = torch.seed()

    # if the location for the generated video frames doesn't exist, create it
    if not os.path.exists(generated_video_frames_path):
        os.mkdir(generated_video_frames_path)
    else:
        VF.delete_files(generated_video_frames_path)

    output_size_X, output_size_Y = VF.filesize_matching_aspect_ratio(video_frames[0], eng_config.output_image_size[0], eng_config.output_image_size[1])
    eng_config.output_image_size = [output_size_X, output_size_Y]
    # alternate_img_target is required for restyling video. alternate_img_target_decay is experimental.
    if eng_config.init_image_method not in ['alternate_img_target_decay', 'alternate_img_target']:
        eng_config.init_image_method = 'alternate_img_target'

    # suppress stdout to keep the progress bar clear
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            eng = Engine(eng_config)
            eng.initialize_VQGAN_CLIP()

    if z_smoother:
        # Populate the z smoother with the initial image
        init_image_pil = Image.open(init_image).convert('RGB').resize([output_size_X,output_size_Y], resample=Image.LANCZOS)
        # init_img_z = eng.pil_image_to_latent_vector(init_image_pil)
        smoothed_z = Z_Smoother(buffer_len=z_smoother_buffer_len, alpha=z_smoother_alpha)

    # generate images
    video_frame_num = 1
    current_prompt_number = 0
    try:
        # To generate the first frame of video, either use the init_image argument, or the first frame of source video.
        pil_image_previous_generated_frame = Image.open(init_image).convert('RGB').resize([output_size_X,output_size_Y], resample=Image.LANCZOS)
        eng.convert_image_to_init_image(pil_image_previous_generated_frame)
        eng.configure_optimizer()
        video_frames_loop = tqdm(video_frames,unit='image',desc='style transfer',leave=leave_progress_bar)
        for video_frame in video_frames_loop:
            filename_to_save = os.path.basename(os.path.splitext(video_frame)[0]) + '.jpg'
            filepath_to_save = os.path.join(generated_video_frames_path,filename_to_save)

            # INIT IMAGE
            # Alternate aglorithm - init image is unchanged from the previous output. We are not resetting the tensor gradient.
            # alternate_image_target is the new source frame of video. Apply a loss in Engine using conf.init_image_method == 'alternate_img_target'
            # The previous output will be trained to change toward the new source frame.
            pil_image_new_frame = Image.open(video_frame).convert('RGB').resize([output_size_X,output_size_Y], resample=Image.LANCZOS)
            eng.set_alternate_image_target(pil_image_new_frame)

            # Optionally use the current source video frame, and the previous generate frames, as input prompts
            eng.clear_all_prompts()
            if change_prompts_on_frame is not None:
                if video_frame_num in change_prompts_on_frame:
                    # change prompts if the current frame number is in the list of change frames
                    current_prompt_number += 1
            eng.encode_and_append_prompts(current_prompt_number, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)
            if current_source_frame_prompt_weight:
                eng.encode_and_append_pil_image(pil_image_new_frame, weight=current_source_frame_prompt_weight)

            # Generate a new image
            for iteration_num in tqdm(range(1,iterations_per_frame+1),unit='iteration',desc='generating frame',leave=False):
                #perform iterations of train()
                lossAll = eng.train(iteration_num)          

            if verbose:
                # display some statistics about how the GAN training is going whever we save an image
                losses_str = ', '.join(f'{loss.item():7.3f}' for loss in lossAll)
                tqdm.write(f'iteration:{iteration_num:6d}\tvideo frame: {video_frame_num:6d}\tloss sum: {sum(lossAll).item():7.3f}\tloss for each prompt:{losses_str}')

            # save a frame of video
            # metadata to save to PNG file as data chunks
            img_info =  [('text_prompts',text_prompts),
                ('image_prompts',image_prompts),
                ('noise_prompts',noise_prompts),
                ('iterations_per_frame',iterations_per_frame),
                ('iterations_for_first_frame',iterations_for_first_frame),
                ('cut_method',eng_config.cut_method),
                ('init_image',video_frame),
                ('seed',eng.conf.seed),
                ('z_smoother',z_smoother),
                ('z_smoother_buffer_len',z_smoother_buffer_len),
                ('z_smoother_alpha',z_smoother_alpha),
                ('current_source_frame_prompt_weight',f'{current_source_frame_prompt_weight:2.2f}'),
                ('current_source_frame_image_weight',f'{current_source_frame_image_weight:2.2f}')]
            if z_smoother:
                smoothed_z.append(eng._z.clone())
                output_tensor = eng.synth(smoothed_z._mid_ewma())
                Engine.save_tensor_as_image(output_tensor,filepath_to_save,img_info)
            else:
                eng.save_current_output(filepath_to_save,img_info)
            last_video_frame_generated = filepath_to_save
            video_frame_num += 1
    except KeyboardInterrupt:
        pass

    config_info=f'iterations_per_frame: {iterations_per_frame}, '\
            f'image_prompts: {image_prompts}, '\
            f'noise_prompts: {noise_prompts}, '\
            f'init_weight {eng_config.init_weight:1.2f}, '\
            f'init_image {init_image}, '\
            f'current_source_frame_prompt_weight {current_source_frame_prompt_weight:2.2f}, '\
            f'current_source_frame_image_weight {current_source_frame_image_weight:2.2f}, '\
            f'cut_method {eng_config.cut_method}, '\
            f'z_smoother {z_smoother:2.2f}, '\
            f'z_smoother_buffer_len {z_smoother_buffer_len:2.2f}, '\
            f'z_smoother_alpha {z_smoother_alpha:2.2f}, '\
            f'seed {eng.conf.seed}'

    return config_info
