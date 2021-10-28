import os, subprocess, glob


def extract_video_frames(input_video_path, extraction_framerate, extracted_video_frames_path='./extracted_video_frames'):
    """Wrapper for ffmpeg. Parse original video file into individual frames formatted frame_%12d.jpg.

    Args:
        input_video_path (str): Location of video file to process.
        extraction_framerate (int): number of frames per second to extract from the original video
        extracted_video_frames_path (str, optional): Where to save extracted still images. Defaults to './extracted_video_frames'.

    Returns:
        List of str: List of paths to result frames, sorted by filename.
    """
    # Parse original video file into individual frames
    # original_video = 'video_restyle\\original_video\\20211004_132008000_iOS.MOV'
    # extraction_framerate = '30' # number of frames per second to extract from the original video

    # purge previously extracted original frames
    if not os.path.exists(extracted_video_frames_path):
        os.mkdir(extracted_video_frames_path)
    else:
        files = glob.glob(extracted_video_frames_path+os.sep+'*')
        for f in files:
            os.remove(f)

    # print("Extracting image frames from original video")
    # extract original video frames
    subprocess.call(['ffmpeg',
        '-i', input_video_path,
        '-filter:v', 'fps='+str(extraction_framerate),
        '-hide_banner',
        '-loglevel', 'error',
        extracted_video_frames_path+os.sep+'frame_%12d.png'])

    video_frames = sorted(glob.glob(extracted_video_frames_path+os.sep+'*.png'))
    if not len(video_frames):
        raise NameError('No video frames were extracted')
    return video_frames

def copy_video_audio(original_video, destination_file_without_audio, output_file):
    extracted_original_audio = 'extracted_original_audio.aac' # audio file, if any, from the original video file

    # extract original audio
    try:
        subprocess.call(['ffmpeg',
            '-i', original_video,
            '-vn', 
            '-acodec', 'copy',
            '-hide_banner',
            '-loglevel', 'error',
            extracted_original_audio])
    except:
        print("Audio extraction failed")

    # if there is extracted audio from the original file, re-merge it here
    # note that in order for the audio to sync up, extracted_video_fps must have matched the original video framerate
    subprocess.call(['ffmpeg',
        '-i', destination_file_without_audio,
        '-i', extracted_original_audio,
        '-c', 'copy', 
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-hide_banner',
        '-loglevel', 'error',
    output_file])
    
    # clean up
    os.remove(extracted_original_audio)


def encode_video(output_file, input_framerate, path_to_stills=f'./video_frames', metadata_title='', metadata_comment='', output_framerate=None, crf=23, vcodec='libx264'):
    """Wrapper for FFMPEG. Encodes a folder of PNG images to a video in HEVC format using ffmpeg with optional interpolation. Input stills must be sequentially numbered png files named in the format frame_%12d.png.
    Note that this wrapper will print to the command line the exact ffmpeg command that was used. You can copy this and run it from the command line with any tweaks necessary.

    Args:
        output_file (str, optional): Location to save the resulting mp4 video file. Defaults to f'.\output\output.mp4'.
        path_to_stills (str, optional): Path to still images. Defaults to f'.\steps'.
        metadata (str, optional): Metadata to be added to the comments field of the resulting video file. Defaults to ''.
        output_framerate (int, optional): The desired framerate of the output video. Defaults to 30.
        input_framerate (int, optional): An assumed framerate to use for the input stills. If the assumed input framerate is different than the desired output, then ffpmeg will interpolate to generate extra frames. For example, an assumed input of 10 and desired output of 60 will cause the resulting video to have five interpolated frames for every original frame. Defaults to [].
        crf (int, optional): The -crf parameter value to pass to ffmpeg. Appropriate values depend on the codec, and image resolution. See ffmpeg documentation for guidance. Defaults to 23.
        vcodec (str, optional): The video codec (-vcodec) to pass to ffmpeg. Any valid video codec for ffmpeg is valid. Defaults to 'libx264'.
    """
    if input_framerate and output_framerate and input_framerate != output_framerate:
        # a different input and output framerate are specified. Use interpolation
        input_framerate_option = f'-r {input_framerate}'
        output_framerate_option = f"-filter:v minterpolate='mi_mode=mci:me=hexbs:me_mode=bidir:mc_mode=aobmc:vsbmc=1:mb_size=8:search_param=32:fps={str(output_framerate)}'"
    else:
        # no interpolation
        input_framerate_option = ''
        output_framerate_to_use = output_framerate if output_framerate else input_framerate
        output_framerate_option = f'-r {output_framerate_to_use}'
    metadata_option = f'-metadata title=\"{metadata_title}\" -metadata comment=\"{metadata_comment}\" -metadata description=\"Generated with https://github.com/rkhamilton/vqgan-clip-generator\"'
    ffmpeg_command = f'ffmpeg -y -f image2 {input_framerate_option} -i {path_to_stills}\\frame_%12d.png {output_framerate_option} -vcodec {vcodec} -crf {crf} -pix_fmt yuv420p -hide_banner -loglevel error {metadata_option} {output_file}'
    subprocess.Popen(ffmpeg_command,shell=True).wait()
    print(f'FFMPEG command used was:\n{ffmpeg_command}')
