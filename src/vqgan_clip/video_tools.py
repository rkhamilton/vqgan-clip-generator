import os
import subprocess
import glob
import cv2


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
    # original_video = 'video_restyle{os.sep}original_video{os.sep}20211004_132008000_iOS.MOV'
    # extraction_framerate = '30' # number of frames per second to extract from the original video

    # clean up arguments for use on shell if needed
    # extracted_video_frames_path = sanitize_path_for_cmd(extracted_video_frames_path)
    # input_video_path = sanitize_path_for_cmd(input_video_path)
    extracted_video_frames_path = f'{extracted_video_frames_path}'
    input_video_path = f'{input_video_path}'

    # purge previously extracted original frames
    if not os.path.exists(enquote_paths_with_spaces(extracted_video_frames_path)):
        os.makedirs(extracted_video_frames_path, exist_ok=True)
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

    video_frames = sorted(
        glob.glob(extracted_video_frames_path+os.sep+'*.png'))
    if not len(video_frames):
        raise NameError('No video frames were extracted')
    return video_frames


def copy_video_audio(original_video, destination_file_without_audio, output_file):
    # audio file, if any, from the original video file
    extracted_original_audio = os.path.join(os.path.dirname(
        destination_file_without_audio), 'extracted_original_audio.aac')
    extracted_original_audio = f'{extracted_original_audio}'
    if os.path.exists(extracted_original_audio):
        os.remove(extracted_original_audio)
    if os.path.exists(output_file):
        os.remove(output_file)

    # extract original audio
    try:
        ffmpeg_command = f'ffmpeg -i {enquote_paths_with_spaces(original_video)} -vn -acodec copy -hide_banner -loglevel error {enquote_paths_with_spaces(extracted_original_audio)}'
        print(f'FFMPEG command used was:\n{ffmpeg_command}')
        subprocess.Popen(ffmpeg_command, shell=True).wait()
        assert(os.path.exists(extracted_original_audio))
    except:
        print("Audio extraction failed")

    # if there is extracted audio from the original file, re-merge it here
    try:
        ffmpeg_command = f'ffmpeg -i {enquote_paths_with_spaces(destination_file_without_audio)} -i {enquote_paths_with_spaces(extracted_original_audio)} -c copy -map 0:v:0 -map 1:a:0 -hide_banner -loglevel error {enquote_paths_with_spaces(output_file)}'
        print(f'FFMPEG command used was:\n{ffmpeg_command}')
        subprocess.Popen(ffmpeg_command, shell=True).wait()
        assert(os.path.exists(output_file))
    except:
        print("Generating output file failed")

    # clean up
    os.remove(extracted_original_audio)


def encode_video(output_file, input_framerate, path_to_stills=f'video_frames', output_framerate=None, metadata_title='', metadata_comment='', vcodec='libx264', crf=23):
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
        output_framerate_option = f"-filter:v minterpolate='mi_mode=mci:me=hexbs:me_mode=bidir:mc_mode=aobmc:vsbmc=1:mb_size=8:search_param=32:fps={str(output_framerate)}'"
    else:
        # no interpolation
        output_framerate_to_use = output_framerate if output_framerate else input_framerate
        output_framerate_option = f'-r {output_framerate_to_use}'
    metadata_option = f'-metadata title=\"{metadata_title}\" -metadata comment=\"{metadata_comment}\" -metadata description=\"Generated with https://github.com/rkhamilton/vqgan-clip-generator\"'
    input_path = enquote_paths_with_spaces(f'{path_to_stills}{os.sep}frame_%12d.png')
    ffmpeg_command = f'ffmpeg -y -f image2 -r {input_framerate} -i {input_path} {output_framerate_option} -vcodec {vcodec} -crf {crf} -pix_fmt yuv420p -hide_banner -loglevel error {metadata_option} {enquote_paths_with_spaces(output_file)}'
    subprocess.Popen(ffmpeg_command, shell=True).wait()
    print(f'FFMPEG command used was:\n{ffmpeg_command}')
    if not os.path.exists(output_file):
        raise NameError('encode_video failed to generate an output file')


def RIFE_interpolation(input, output, interpolation_factor=4, metadata_title='', metadata_comment=''):
    """Perform optical flow interpolation with arXiv2020-RIFE.

    Args:
        input (str): Path to input video.
        output (str): Output file path.
        interpolation_factor (int, optional): Fold-increase in framerate desired. Options are 4 or 16. Defaults to 4.
        metadata_title (str, optional): String to store in the resulting mp4 title metadata field. Defaults to ''.
        metadata_comment (str, optional): String to store in the resulting mp4 title metadata field. Defaults to ''.
    """
    # This section runs RIFE optical flow interpolation and then compresses the resulting (uncompressed) video to h264 format.
    # Valid choices are 4 or 16

    if not os.path.exists(f'arXiv2020-RIFE{os.sep}inference_video.py'):
        raise NameError('RIFE NOT FOUND at arXiv2020-RIFE{os.sep}inference_video.py')
    if not os.path.exists(f'arXiv2020-RIFE{os.sep}train_log{os.sep}flownet.pkl'):
        raise NameError('RIFE MODEL NOT FOUND in arXiv2020-RIFE{os.sep}train_log{os.sep}')

    if os.path.exists(output):
        os.remove(output)

    # we need the framerate of the source video to know what name RIFE will use in the output
    input_framerate = cv2.VideoCapture(input).get(cv2.CAP_PROP_FPS)

    of_cmnd = f'python arXiv2020-RIFE{os.sep}inference_video.py --exp={2 if interpolation_factor==4 else 4} --model=arXiv2020-RIFE{os.sep}train_log --video={enquote_paths_with_spaces(input)}'
    subprocess.Popen(of_cmnd, shell=True).wait()
    # print(f'RIFE optical flow command used was:\n{of_cmnd}')

    # Re-encode the RIFE output to a compressed format
    metadata_option = f'-metadata title=\"{metadata_title}\" -metadata comment=\"{metadata_comment}\" -metadata description=\"Generated with https://github.com/rkhamilton/vqgan-clip-generator\"'
    # RIFE appends a string to the original filename of the form "original filename_4X_120fps.mp4"
    RIFE_output_filename = f'{os.path.splitext(input)[0]}_{interpolation_factor}X_{int(input_framerate*interpolation_factor)}fps.mp4'
    if not os.path.exists(RIFE_output_filename):
        raise NameError('RIFE_interpolation failed to generate an output file')

    ffmpeg_command = f'ffmpeg -y -i {enquote_paths_with_spaces(RIFE_output_filename)} -vcodec libx264 -crf 23 -pix_fmt yuv420p -hide_banner -loglevel error {metadata_option} {enquote_paths_with_spaces(output)}'
    subprocess.Popen(ffmpeg_command, shell=True).wait()
    if not os.path.exists(output):
        raise NameError('RIFE_interpolation failed to generate an output file')
    os.remove(RIFE_output_filename)
    # print(f'FFMPEG command used was:\t{ffmpeg_command}')


def enquote_paths_with_spaces(path):
    """Clean up a python path that contains spaces to be enclosed in double quotes for use on the command line.

    Args:
        path (str): a path string

    Returns:
        str: a path string enclosed in double quotes if needed
    """
    if ' ' in path:
        if path[0] != '\"' and path[-1] != '\"':
            clean_path = '\"' + path.strip('\"').strip('\'') + '\"'
    else:
        clean_path = path
    return clean_path
