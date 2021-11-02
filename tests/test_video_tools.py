import pytest
import os, glob
from vqgan_clip import video_tools


TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files',
    )

TEST_VIDEO = os.path.join(TEST_DATA_DIR,'small.mp4')

def test_video_tools(tmpdir):
    output_images_path = str(tmpdir.mkdir('video_frames'))
    original_video_frames = video_tools.extract_video_frames(TEST_VIDEO, 
        extraction_framerate = 30,
        extracted_video_frames_path=output_images_path)
        
    output_files = glob.glob(os.path.join(output_images_path,'*.jpg'))
    assert len(output_files) == len(original_video_frames)

    # Non-interpolated video
    output_video_filename = os.path.join(output_images_path,'test_video.mp4')
    video_tools.encode_video(output_file=output_video_filename,
            input_framerate=60,
            path_to_stills=output_images_path,
            metadata_title='test title',
            metadata_comment='test comment')
    assert os.path.exists(output_video_filename)
    os.remove(output_video_filename)

    # Interpolated video
    video_tools.encode_video(output_file=output_video_filename,
        path_to_stills=output_images_path,
        metadata_title='test title',
        metadata_comment='test comment',
        output_framerate=60,
        input_framerate=30)
    assert os.path.exists(output_video_filename)

    # Test copying audio from one video to another
    final_video_with_audio = os.path.join(output_images_path,'final_video.mp4')
    video_tools.copy_video_audio(TEST_VIDEO, output_video_filename, final_video_with_audio)
    assert os.path.exists(final_video_with_audio)

    # remove generated images
    os.remove(final_video_with_audio)
    os.remove(output_video_filename)
    for f in output_files:
        os.remove(f)

def test_RIFE_wrapper(tmpdir):
    output_images_path = str(tmpdir.mkdir('video_frames'))
    output_video_filename = os.path.join(output_images_path,'test_video.mp4')

    RIFE_output_filename = f'{os.path.splitext(output_video_filename)[0]}_RIFE.mp4'
    video_tools.RIFE_interpolation(input=TEST_VIDEO,
                       output=RIFE_output_filename,
                       interpolation_factor=4,
                       metadata_title='text_prompts',
                       metadata_comment='metadata_comment')

    assert os.path.exists(RIFE_output_filename)
    os.remove(RIFE_output_filename)