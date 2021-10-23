import pytest
import os, glob
from vqgan_clip import video_tools


TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files',
    )

TEST_VIDEO = os.path.join(TEST_DATA_DIR,'small.mp4')

def test_extract_and_encode_video_frames(tmpdir):
    output_images_path = str(tmpdir.mkdir('video_frames'))
    original_video_frames = video_tools.extract_video_frames(TEST_VIDEO, 
        extraction_framerate = 30,
        extracted_video_frames_path=output_images_path)
        
    output_files = glob.glob(os.path.join(output_images_path,'*.png'))
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
    os.remove(output_video_filename)

    # remove generated images
    for f in output_files:
        os.remove(f)




def test_copy_audio():
    assert False