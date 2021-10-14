# This module is the interface for creating images and video from text prompts
# This should also serve as examples of how you can use the Engine class to create images and video using your own creativity.
from vqgan_clip.engine import Engine, VQGAN_CLIP_Config
from tqdm import tqdm
import glob, os, sys, io
import subprocess
import contextlib

from PIL import ImageFile, Image, ImageChops
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import functional as TF
from . import _functional as VF

def single_image(eng_config=VQGAN_CLIP_Config(),
        text_prompts = [],
        image_prompts = [],
        noise_prompts = [],
        iterations = 100,
        save_every = 50,
        output_filename = 'output' + os.sep + 'output',
        change_prompt_every = 0):
    """Generate an image using VQGAN+CLIP. The configuration of the algorithms is done via a VQGAN_CLIP_Config instance.

    Args:
        * eng_config (VQGAN_CLIP_Config, optional): An instance of VQGAN_CLIP_Config with attributes customized for your use. See the documentation for VQGAN_CLIP_Config().
        * text_prompts (str, optional) : Text that will be turned into a prompt via CLIP. Default = []  
        * image_prompts (str, optional) : Path to image that will be turned into a prompt via CLIP. Default = []
        * noise_prompts (str, optional) : Random number seeds can be used as prompts using the same format as a text prompt. E.g. \'123:0.1|234:0.2|345:0.3\' Stories (^) are supported. Default = []
        * iterations (int, optional) : Number of iterations of train() to perform before stopping. Default = 100 
        * save_every (int, optional) : An interim image will be saved to the output location every save_every iterations, and training stats will be displayed. Default = 50  
        * output_filename (str, optional) : location to save the output image. Omit the file extension. Default = \'output\' + os.sep + \'output\'  
        * change_prompt_every (int, optional) : Serial prompts, sepated by ^, will be cycled through every change_prompt_every iterations. Prompts will loop if more cycles are requested than there are prompts. Default = 0
    """
    eng = Engine(eng_config)
    eng.initialize_VQGAN_CLIP()
    parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts = VF.parse_all_prompts(text_prompts, image_prompts, noise_prompts)
    eng.encode_and_append_prompts(0, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)
    eng.configure_optimizer()
    output_file = output_filename + '.png'
    # generate the image
    current_prompt_number = 0
    try:
        for iteration_num in tqdm(range(1,iterations+1),unit='iteration'):
            #perform iterations of train()
            lossAll = eng.train(iteration_num)
            if change_prompt_every and iteration_num % change_prompt_every == 0:
                # change prompts if every change_prompt_every iterations
                current_prompt_number += 1
                eng.clear_all_prompts()
                eng.encode_and_append_prompts(current_prompt_number, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)
            if save_every and iteration_num % save_every == 0:
                # display some statistics about how the GAN training is going whever we save an interim image
                losses_str = ', '.join(f'{loss.item():7.3f}' for loss in lossAll)
                tqdm.write(f'iteration:{iteration_num:6d}\tloss sum: {sum(lossAll).item():7.3f}\tloss for each prompt:{losses_str}')
                # save an interim copy of the image so you can look at it as it changes if you like
                eng.save_current_output(output_file) 
        # Always save the output at the end
        eng.save_current_output(output_file) 
    except KeyboardInterrupt:
        pass


def multiple_images(eng_config=VQGAN_CLIP_Config(),
        text_prompts = [],
        image_prompts = [],
        noise_prompts = [],
        iterations = 100,
        save_every = 101,
        change_prompt_every = 0,
        num_images_to_generate = 10,
        output_images_path='./video_frames'):
    """Generate multiple images using VQGAN+CLIP, each with a different random seed. The configuration of the algorithms is done via a VQGAN_CLIP_Config instance.  
    The use case for this function is to generate a lot of variants on the same prompt, and then look through the output folder for \'keepers.\'

    Args:
        * eng_config (VQGAN_CLIP_Config, optional): An instance of VQGAN_CLIP_Config with attributes customized for your use. See the documentation for VQGAN_CLIP_Config().
        * text_prompts (str, optional) : Text that will be turned into a prompt via CLIP. Default = []  
        * image_prompts (str, optional) : Path to image that will be turned into a prompt via CLIP. Default = []
        * noise_prompts (str, optional) : Random number seeds can be used as prompts using the same format as a text prompt. E.g. \'123:0.1|234:0.2|345:0.3\' Stories (^) are supported. Default = []
        * iterations (int, optional) : Number of iterations of train() to perform before stopping. Default = 100 
        * save_every (int, optional) : An interim image will be saved to the output location every save_every iterations, and training stats will be displayed. Default = 50  
        * change_prompt_every (int, optional) : Serial prompts, sepated by ^, will be cycled through every change_prompt_every iterations. Prompts will loop if more cycles are requested than there are prompts. Default = 0
        * num_images_to_generate (int, optional) : Number of images to generates. Default = 10
        * output_images_path (str, optional) : Path to save all generated images. Default = './video_frames'
    """
    parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts = VF.parse_all_prompts(text_prompts, image_prompts, noise_prompts)

    # if the location for the images doesn't exist, create it
    if not os.path.exists(output_images_path):
        os.mkdir(output_images_path)
    else:
        VF.delete_files(output_images_path)

    # generate the image
    current_prompt_number = 0
    try:
        files = tqdm(range(1,num_images_to_generate+1),unit='file')
        for file_num in files:
            # files.desc = f'Multiple Files {file_num}'
            # load a fresh copy of the VQGAN model for training on the new image
            # suppress stdout so the progressbar looks nice
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    eng = Engine(eng_config)
                    eng.initialize_VQGAN_CLIP()
            eng.encode_and_append_prompts(0, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)
            eng.configure_optimizer()
            for iteration_num in range(1,iterations+1):
                #perform iterations of train()
                lossAll = eng.train(iteration_num)
                if change_prompt_every and iteration_num % change_prompt_every == 0:
                    # change prompts if every change_prompt_every iterations
                    current_prompt_number += 1
                    eng.clear_all_prompts()
                    eng.encode_and_append_prompts(current_prompt_number, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)
                if save_every and iteration_num % save_every == 0:
                    # display some statistics about how the GAN training is going whever we save an interim image
                    losses_str = ', '.join(f'{loss.item():7.3f}' for loss in lossAll)
                    tqdm.write(f'iteration:{iteration_num:6d}\tfiles generated: {file_num:6d}\tloss sum: {sum(lossAll).item():7.3f}\tloss for each prompt:{losses_str}')
                    # save an interim copy of the image so you can look at it as it changes if you like
                    eng.save_current_output(output_images_path + os.sep + str(file_num) + '.png')
            #Always save a file at the end
            eng.save_current_output(output_images_path + os.sep + str(file_num) + '.png')
    except KeyboardInterrupt:
        pass

def restyle_video(input_video_path,
        extraction_framerate = 30,
        eng_config=VQGAN_CLIP_Config(),
        text_prompts = [],
        image_prompts = [],
        noise_prompts = [],
        iterations = 100,
        save_every = None,
        output_filename = 'output' + os.sep + 'output',
        change_prompt_every = 0,
        video_frames_path='./video_frames',
        extracted_video_frames_path='./extracted_video_frames',
        output_framerate=30, 
        assumed_input_framerate=None,
        copy_audio=False):
    """Apply a style to an existing video using VQGAN+CLIP. The still image frames from the original video are extracted, and used as initial images for VQGAN+CLIP. The resulting folder of stills are then encoded into an HEVC video file. The audio from the original may optionally be transferred.
    The configuration of the VQGAN+CLIP algorithms is done via a VQGAN_CLIP_Config instance.

    Args:
        Args:
        * input_video_path (str) : Path to the video that will be restyled.
        * extraction_framerate (int) : The number of frames per second of original video to extract. Interpolation will be used to extract an arbitrary number of frames.
        * eng_config (VQGAN_CLIP_Config, optional): An instance of VQGAN_CLIP_Config with attributes customized for your use. See the documentation for VQGAN_CLIP_Config().
        * text_prompts (str, optional) : Text that will be turned into a prompt via CLIP. Default = []  
        * image_prompts (str, optional) : Path to image that will be turned into a prompt via CLIP. Default = []
        * noise_prompts (str, optional) : Random number seeds can be used as prompts using the same format as a text prompt. E.g. \'123:0.1|234:0.2|345:0.3\' Stories (^) are supported. Default = []
        * iterations (int, optional) : Number of iterations of train() to perform before stopping. Default = 100 
        * output_filename (str, optional) : location to save the output image. Omit the file extension. Default = \'output\' + os.sep + \'output\'  
        * change_prompt_every (int, optional) : Serial prompts, sepated by ^, will be cycled through every change_prompt_every iterations. Prompts will loop if more cycles are requested than there are prompts. Default = 0
        * video_frames_path (str, optional) : Path where still images should be saved as they are generated before being combined into a video. Defaults to './video_frames'.
        * output_framerate (int, optional) : Desired framerate of the output video. Defaults to 30.
        * assumed_input_framerate (int, optional) : An assumed framerate to use for the still images. If an assumed input framerate is provided, the output video will be interpolated to the specified output framerate. Defaults to None.
        * copy_audio (boolean) : If True, attempt to copy the audio from the original video to the output video. The durations of the two videos should be the same.
    """
    eng = Engine(eng_config)
    parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts = VF.parse_all_prompts(text_prompts, image_prompts, noise_prompts)
    eng.initialize_VQGAN_CLIP()
    eng.encode_and_append_prompts(0, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)

    # if the location for the interim video frames doesn't exist, create it
    if not os.path.exists(video_frames_path):
        os.mkdir(video_frames_path)
    else:
        VF.delete_files(video_frames_path)

    video_frames = VF.extract_video_frames(input_video_path, extraction_framerate, extracted_video_frames_path)
    output_size_X, output_size_Y = VF.filesize_matching_aspect_ratio(video_frames[0], eng_config.output_image_size[0], eng_config.output_image_size[1])
    eng_config.output_image_size = [output_size_X, output_size_Y]

    # generate images
    current_prompt_number = 0
    video_frame_num = 1
    try:
        for video_frame in tqdm(video_frames,unit='image'):
            # suppress stdout to keep the progress bar clear
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    eng.load_model()
            # Use the next frame of video as an initial image for VQGAN+CLIP
            pil_image = Image.open(video_frame).convert('RGB')
            eng.convert_image_to_init_image(pil_image)
            eng.configure_optimizer()

            # Generate a new image
            for iteration_num in range(1,iterations+1):
                #perform iterations of train()
                lossAll = eng.train(iteration_num)
                if change_prompt_every and iteration_num % change_prompt_every == 0:
                    # change prompts if every change_prompt_every iterations
                    current_prompt_number += 1
                    eng.clear_all_prompts()
                    eng.encode_and_append_prompts(current_prompt_number, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)

                if save_every and iteration_num % save_every == 0:
                    # save a frame of video every .save_every iterations
                    losses_str = ', '.join(f'{loss.item():7.3f}' for loss in lossAll)
                    tqdm.write(f'iteration:{iteration_num:6d}\tvideo frame: {video_frame_num:6d}\tloss sum: {sum(lossAll).item():7.3f}\tloss for each prompt:{losses_str}')
                    eng.save_current_output(video_frames_path + os.sep + str(video_frame_num) + '.png')

            # save a frame of video every iterations
            # display some statistics about how the GAN training is going whever we save an image
            losses_str = ', '.join(f'{loss.item():7.3f}' for loss in lossAll)
            tqdm.write(f'iteration:{iteration_num:6d}\tvideo frame: {video_frame_num:6d}\tloss sum: {sum(lossAll).item():7.3f}\tloss for each prompt:{losses_str}')

            # if making a video, save a frame named for the video step
            eng.save_current_output(video_frames_path + os.sep + str(video_frame_num) + '.png')
            video_frame_num += 1
        tqdm.write('Generating video...')
    except KeyboardInterrupt:
        pass

    # Encode the video even if the user aborts generating stills using CTRL+C
    if not assumed_input_framerate:
        # The framerate used for encoding should match the framerate used for extraction, unless otherwise specified by the calling function. This ensures the output video is the same duration as the input.
        assumed_input_framerate = extraction_framerate
   
    output_file = output_filename + '.mp4'
    encode_video(output_file=output_file,
        path_to_stills=video_frames_path, 
        metadata=text_prompts,
        output_framerate=output_framerate,
        assumed_input_framerate=assumed_input_framerate)

    if copy_audio:
        os.rename(output_file,'original_no_audio.mp4')
        VF.copy_video_audio(input_video_path, 'original_no_audio.mp4', output_file)
        os.remove('original_no_audio.mp4')

def restyle_video2(input_video_path,
        extraction_framerate = 30,
        eng_config=VQGAN_CLIP_Config(),
        text_prompts = [],
        image_prompts = [],
        noise_prompts = [],
        iterations = 100,
        save_every = None,
        output_filename = 'output' + os.sep + 'output',
        change_prompt_every = 0,
        video_frames_path='./video_frames',
        extracted_video_frames_path='./extracted_video_frames',
        output_framerate=30, 
        assumed_input_framerate=None,
        copy_audio=False):
    """Apply a style to an existing video using VQGAN+CLIP. The still image frames from the original video are extracted, and used as initial images for VQGAN+CLIP. The resulting folder of stills are then encoded into an HEVC video file. The audio from the original may optionally be transferred.
    The configuration of the VQGAN+CLIP algorithms is done via a VQGAN_CLIP_Config instance. Unlike restyle_video, restyle_video2 uses the current frame of source
    video as an image prompt.

    Args:
        Args:
        * input_video_path (str) : Path to the video that will be restyled.
        * extraction_framerate (int) : The number of frames per second of original video to extract. Interpolation will be used to extract an arbitrary number of frames.
        * eng_config (VQGAN_CLIP_Config, optional): An instance of VQGAN_CLIP_Config with attributes customized for your use. See the documentation for VQGAN_CLIP_Config().
        * text_prompts (str, optional) : Text that will be turned into a prompt via CLIP. Default = []  
        * image_prompts (str, optional) : Path to image that will be turned into a prompt via CLIP. Default = []
        * noise_prompts (str, optional) : Random number seeds can be used as prompts using the same format as a text prompt. E.g. \'123:0.1|234:0.2|345:0.3\' Stories (^) are supported. Default = []
        * iterations (int, optional) : Number of iterations of train() to perform before stopping. Default = 100 
        * output_filename (str, optional) : location to save the output image. Omit the file extension. Default = \'output\' + os.sep + \'output\'  
        * change_prompt_every (int, optional) : Serial prompts, sepated by ^, will be cycled through every change_prompt_every iterations. Prompts will loop if more cycles are requested than there are prompts. Default = 0
        * video_frames_path (str, optional) : Path where still images should be saved as they are generated before being combined into a video. Defaults to './video_frames'.
        * output_framerate (int, optional) : Desired framerate of the output video. Defaults to 30.
        * assumed_input_framerate (int, optional) : An assumed framerate to use for the still images. If an assumed input framerate is provided, the output video will be interpolated to the specified output framerate. Defaults to None.
        * copy_audio (boolean) : If True, attempt to copy the audio from the original video to the output video. The durations of the two videos should be the same.
    """
    parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts = VF.parse_all_prompts(text_prompts, image_prompts, noise_prompts)
    output_file = output_filename + '.mp4'
    eng = Engine(eng_config)
    eng.initialize_VQGAN_CLIP()

    # if the location for the interim video frames doesn't exist, create it
    if not os.path.exists(video_frames_path):
        os.mkdir(video_frames_path)
    else:
        VF.delete_files(video_frames_path)

    video_frames = VF.extract_video_frames(input_video_path, extraction_framerate, extracted_video_frames_path)
    output_size_X, output_size_Y = VF.filesize_matching_aspect_ratio(video_frames[0], eng_config.output_image_size[0], eng_config.output_image_size[1])
    eng_config.output_image_size = [output_size_X, output_size_Y]

    # generate images
    current_prompt_number = 0
    video_frame_num = 1
    try:
        for video_frame in tqdm(video_frames,unit='image'):
            # load a fresh VQGAN model to train for the new image
            # suppress stdout so the progressbar looks nice
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    eng.load_model()
            eng.encode_and_append_prompts(0, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)
            
            # Use the next frame of video as an initial image for VQGAN+CLIP
            pil_image = Image.open(video_frame).convert('RGB')
            eng.convert_image_to_init_image(pil_image)
            # Also use the current source video frame as an input prompt
            eng.encode_and_append_pil_image(pil_image, weight=1)
            eng.configure_optimizer()

            # Generate a new image
            for iteration_num in range(1,iterations+1):
                #perform iterations of train()
                lossAll = eng.train(iteration_num)
                if change_prompt_every and iteration_num % change_prompt_every == 0:
                    # change prompts if every change_prompt_every iterations
                    current_prompt_number += 1
                    eng.clear_all_prompts()
                    eng.encode_and_append_prompts(current_prompt_number, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)

                if save_every and iteration_num % save_every == 0:
                    # save a frame of video every .save_every iterations
                    losses_str = ', '.join(f'{loss.item():7.3f}' for loss in lossAll)
                    tqdm.write(f'iteration:{iteration_num:6d}\tvideo frame: {video_frame_num:6d}\tloss sum: {sum(lossAll).item():7.3f}\tloss for each prompt:{losses_str}')
                    eng.save_current_output(video_frames_path + os.sep + str(video_frame_num) + '.png')

            # save a frame of video every iterations
            # display some statistics about how the GAN training is going whever we save an image
            losses_str = ', '.join(f'{loss.item():7.3f}' for loss in lossAll)
            tqdm.write(f'iteration:{iteration_num:6d}\tvideo frame: {video_frame_num:6d}\tloss sum: {sum(lossAll).item():7.3f}\tloss for each prompt:{losses_str}')

            # if making a video, save a frame named for the video step
            eng.save_current_output(video_frames_path + os.sep + str(video_frame_num) + '.png')
            video_frame_num += 1
        tqdm.write('Generating video...')
    except KeyboardInterrupt:
        pass

    # Encode the video even if the user aborts generating stills using CTRL+C
    if not assumed_input_framerate:
        # The framerate used for encoding should match the framerate used for extraction, unless otherwise specified by the calling function. This ensures the output video is the same duration as the input.
        assumed_input_framerate = extraction_framerate

    encode_video(output_file=output_file,
        path_to_stills=video_frames_path, 
        metadata=text_prompts,
        output_framerate=output_framerate,
        assumed_input_framerate=assumed_input_framerate)

    if copy_audio:
        os.rename(output_file,'original_no_audio.mp4')
        VF.copy_video_audio(input_video_path, 'original_no_audio.mp4', output_file)
        os.remove('original_no_audio.mp4')

def video(eng_config=VQGAN_CLIP_Config(),
        text_prompts = [],
        image_prompts = [],
        noise_prompts = [],
        iterations = 100,
        save_every = 50,
        output_filename = 'output' + os.sep + 'output',
        change_prompt_every = 0,
        video_frames_path='./video_frames', 
        output_framerate=30, 
        assumed_input_framerate=None):
    """Generate a video using VQGAN+CLIP. The configuration of the VQGAN+CLIP algorithms is done via a VQGAN_CLIP_Config instance.

    Args:
        Args:
        * eng_config (VQGAN_CLIP_Config, optional): An instance of VQGAN_CLIP_Config with attributes customized for your use. See the documentation for VQGAN_CLIP_Config().
        * text_prompts (str, optional) : Text that will be turned into a prompt via CLIP. Default = []  
        * image_prompts (str, optional) : Path to image that will be turned into a prompt via CLIP. Default = []
        * noise_prompts (str, optional) : Random number seeds can be used as prompts using the same format as a text prompt. E.g. \'123:0.1|234:0.2|345:0.3\' Stories (^) are supported. Default = []
        * iterations (int, optional) : Number of iterations of train() to perform before stopping. Default = 100 
        * save_every (int, optional) : An interim image will be saved to the output location every save_every iterations, and training stats will be displayed. Default = 50  
        * output_filename (str, optional) : location to save the output image. Omit the file extension. Default = \'output\' + os.sep + \'output\'  
        * change_prompt_every (int, optional) : Serial prompts, sepated by ^, will be cycled through every change_prompt_every iterations. Prompts will loop if more cycles are requested than there are prompts. Default = 0
        * video_frames_path (str, optional) : Path where still images should be saved as they are generated before being combined into a video. Defaults to './video_frames'.
        * output_framerate (int, optional) : Desired framerate of the output video. Defaults to 30.
        * assumed_input_framerate (int, optional) : An assumed framerate to use for the still images. If an assumed input framerate is provided, the output video will be interpolated to the specified output framerate. Defaults to None.
    """
    eng = Engine(eng_config)
    eng.initialize_VQGAN_CLIP()
    parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts = VF.parse_all_prompts(text_prompts, image_prompts, noise_prompts)
    eng.encode_and_append_prompts(0, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)
    eng.configure_optimizer()
    output_file = output_filename + '.mp4'

    # if the location for the interim video frames doesn't exist, create it
    if not os.path.exists(video_frames_path):
        os.mkdir(video_frames_path)
    else:
        VF.delete_files(video_frames_path)

    # generate images
    current_prompt_number = 0
    video_frame_num = 1
    try:
        for iteration_num in tqdm(range(1,iterations+1),unit='iteration'):
            #perform iterations of train()
            lossAll = eng.train(iteration_num)
            if change_prompt_every and iteration_num % change_prompt_every == 0:
                # change prompts if every change_prompt_every iterations
                current_prompt_number += 1
                eng.clear_all_prompts()
                eng.encode_and_append_prompts(current_prompt_number, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)

            if save_every and iteration_num % save_every == 0:
                # save a frame of video every .save_every iterations
                # display some statistics about how the GAN training is going whever we save an interim image
                losses_str = ', '.join(f'{loss.item():7.3f}' for loss in lossAll)
                tqdm.write(f'iteration:{iteration_num:6d}\tvideo frame: {video_frame_num:6d}\tloss sum: {sum(lossAll).item():7.3f}\tloss for each prompt:{losses_str}')

                # if making a video, save a frame named for the video step
                eng.save_current_output(video_frames_path + os.sep + str(video_frame_num) + '.png')
                video_frame_num += 1
        tqdm.write('Generating video...')
    except KeyboardInterrupt:
        pass

    # Encode the video even if the user aborts generating stills using CTRL+C
    encode_video(output_file=output_file,
        path_to_stills=video_frames_path, 
        metadata=text_prompts,
        output_framerate=output_framerate,
        assumed_input_framerate=assumed_input_framerate)

def zoom_video(eng_config=VQGAN_CLIP_Config(),
        text_prompts = [],
        image_prompts = [],
        noise_prompts = [],
        iterations = 100,
        save_every = 50,
        output_filename = 'output' + os.sep + 'output',
        change_prompt_every = 0,
        video_frames_path='./video_frames',
        output_framerate=30,
        assumed_input_framerate=None,
        zoom_scale=1.0,
        shift_x=0, 
        shift_y=0):
    """Generate a video using VQGAN+CLIP where each frame moves relative to the previous frame. The configuration of the VQGAN+CLIP algorithms is done via a VQGAN_CLIP_Config instance.

    Args:
        * eng_config (VQGAN_CLIP_Config, optional): An instance of VQGAN_CLIP_Config with attributes customized for your use. See the documentation for VQGAN_CLIP_Config().
        * text_prompts (str, optional) : Text that will be turned into a prompt via CLIP. Default = []  
        * image_prompts (str, optional) : Path to image that will be turned into a prompt via CLIP. Default = []
        * noise_prompts (str, optional) : Random number seeds can be used as prompts using the same format as a text prompt. E.g. \'123:0.1|234:0.2|345:0.3\' Stories (^) are supported. Default = []
        * iterations (int, optional) : Number of iterations of train() to perform before stopping. Default = 100 
        * save_every (int, optional) : An interim image will be saved to the output location every save_every iterations, and training stats will be displayed. Default = 50  
        * output_filename (str, optional) : location to save the output image. Omit the file extension. Default = \'output\' + os.sep + \'output\'  
        * change_prompt_every (int, optional) : Serial prompts, sepated by ^, will be cycled through every change_prompt_every iterations. Prompts will loop if more cycles are requested than there are prompts. Default = 0
        * video_frames_path (str, optional) : Path where still images should be saved as they are generated before being combined into a video. Defaults to './video_frames'.
        * output_framerate (int, optional) : Desired framerate of the output video. Defaults to 30.
        * assumed_input_framerate (int, optional) : An assumed framerate to use for the still images. If an assumed input framerate is provided, the output video will be interpolated to the specified output framerate. Defaults to None.
        * zoom_scale (float) : Every save_every iterations, a video frame is saved. That frame is shifted scaled by a factor of zoom_scale, and used as the initial image to generate the next frame. Default = 1.0
        * shift_x (int) : Every save_every iterations, a video frame is saved. That frame is shifted shift_x pixels in the x direction, and used as the initial image to generate the next frame. Default = 0
        * shift_y (int) : Every save_every iterations, a video frame is saved. That frame is shifted shift_y pixels in the y direction, and used as the initial image to generate the next frame. Default = 0
    """
    parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts = VF.parse_all_prompts(text_prompts, image_prompts, noise_prompts)
    eng = Engine(eng_config)
    eng.initialize_VQGAN_CLIP()
    eng.encode_and_append_prompts(0, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)
    eng.configure_optimizer()
    output_file = output_filename + '.mp4'

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
        for iteration_num in tqdm(range(1,iterations+1),unit='iteration'):
            #perform iterations of train()
            lossAll = eng.train(iteration_num)

            if change_prompt_every and iteration_num % change_prompt_every == 0:
                # change prompts if every change_prompt_every iterations
                current_prompt_number += 1
                eng.clear_all_prompts()
                eng.encode_and_append_prompts(current_prompt_number, parsed_text_prompts, parsed_image_prompts, parsed_noise_prompts)

            if save_every and iteration_num % save_every == 0:
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
                # with nostdout():
                #     eng.load_model()
                eng.configure_optimizer()

                # save a frame of video every .save_every iterations
                # display some statistics about how the GAN training is going whever we save an interim image
                losses_str = ', '.join(f'{loss.item():7.3f}' for loss in lossAll)
                tqdm.write(f'iteration:{iteration_num:6d}\tvideo frame: {video_frame_num:6d}\tloss sum: {sum(lossAll).item():7.3f}\tloss for each prompt:{losses_str}')

                # if making a video, save a frame named for the video step
                eng.save_current_output(video_frames_path + os.sep + str(video_frame_num) + '.png')
                video_frame_num += 1
        tqdm.write('Generating video...')
    except KeyboardInterrupt:
        pass

    # Encode the video even if the user aborts generating stills using CTRL+C
    encode_video(output_file=output_file,
        path_to_stills=video_frames_path, 
        metadata=text_prompts,
        output_framerate=output_framerate,
        assumed_input_framerate=assumed_input_framerate)

def encode_video(output_file=f'.\\output\\output.mp4', path_to_stills=f'.\\steps', metadata='', output_framerate=30, assumed_input_framerate=None):
    """Encodes a folder of PNG images to a video in HEVC format using ffmpeg with optional interpolation. Input stills must be sequentially numbered png files starting from 1. E.g. 1.png 2.png etc.

    Args:
        output_file (str, optional): Location to save the resulting mp4 video file. Defaults to f'.\output\output.mp4'.
        path_to_stills (str, optional): Path to still images. Defaults to f'.\steps'.
        metadata (str, optional): Metadata to be added to the comments field of the resulting video file. Defaults to ''.
        output_framerate (int, optional): The desired framerate of the output video. Defaults to 30.
        assumed_input_framerate (int, optional): An assumed framerate to use for the input stills. If the assumed input framerate is different than the desired output, then ffpmeg will interpolate to generate extra frames. For example, an assumed input of 10 and desired output of 60 will cause the resulting video to have five interpolated frames for every original frame. Defaults to [].
    """
    if assumed_input_framerate and assumed_input_framerate != output_framerate:
        # Hardware encoding and video frame interpolation
        print("Creating interpolated frames...")
        ffmpeg_filter = f"minterpolate='mi_mode=mci:me=hexbs:me_mode=bidir:mc_mode=aobmc:vsbmc=1:mb_size=8:search_param=32:fps={str(output_framerate)}'"
        subprocess.call(['ffmpeg',
            '-y',
            '-f', 'image2',
            '-r', str(assumed_input_framerate),               
            '-i', f'{path_to_stills+os.sep}%d.png',
            '-vcodec', 'libx265',
            '-pix_fmt', 'yuv420p',
            '-strict', '-2',
            '-filter:v', f'{ffmpeg_filter}',
            '-metadata', f'comment={metadata}',
            output_file])
    else:
        # no interpolation
        subprocess.call(['ffmpeg',
            '-y',
            '-f', 'image2',
            '-i', f'{path_to_stills+os.sep}%d.png',
            '-r', str(output_framerate),
            '-vcodec', 'libx265',
            '-pix_fmt', 'yuv420p',
            '-strict', '-2',
            '-metadata', f'comment={metadata}',
            output_file])


def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                func(*a, **ka)
    return wrapper