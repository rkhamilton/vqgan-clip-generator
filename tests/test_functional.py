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

SMALL_IMAGE_PNG = os.path.join(TEST_DATA_DIR,'small.png')
SMALL_IMAGE_JPG = os.path.join(TEST_DATA_DIR,'small.jpg')
TEST_VIDEO = os.path.join(TEST_DATA_DIR,'small.mp4')

def test_metadata_copy_png():
    