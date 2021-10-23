
import pytest
import vqgan_clip.generate
from vqgan_clip import esrgan
import os
import vqgan_clip._functional as VF
import glob
from vqgan_clip import video_tools

TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files',
    )

SMALL_IMAGE = os.path.join(TEST_DATA_DIR,'small.png')
TEST_VIDEO = os.path.join(TEST_DATA_DIR,'small.mp4')

def test_upscale_image(tmpdir):
    upscaled_images_path = str(tmpdir.mkdir('upscaled_video_frames'))
    esrgan.inference_realesrgan(input=SMALL_IMAGE,
        output_images_path=upscaled_images_path,
        face_enhance=False,
        purge_existing_files=True,
        netscale=4,
        outscale=4)
    
    output_files = glob.glob(upscaled_images_path + os.sep + '*.png')
    assert len(output_files) == 1
    for f in output_files:
        os.remove(f)

def test_upscale_folder(tmpdir):
    # extract video frames to generate images to upscale
    extracted_images_path = str(tmpdir.mkdir('extracted_video_frames'))
    original_video_frames = video_tools.extract_video_frames(TEST_VIDEO, 
        extraction_framerate = 2,
        extracted_video_frames_path=extracted_images_path)
    
    upscaled_images_path = str(tmpdir.mkdir('upscaled_video_frames'))
    esrgan.inference_realesrgan(input=extracted_images_path,
        output_images_path=upscaled_images_path,
        face_enhance=False,
        purge_existing_files=True,
        netscale=4,
        outscale=4)
    
    output_files = glob.glob(upscaled_images_path + os.sep + '*.png')
    assert len(output_files) > 0
    assert len(output_files) == len(original_video_frames)
    for f in output_files:
        os.remove(f)
    for f in original_video_frames:
        os.remove(f)