import pytest
import subprocess, os

THIS_FILE_PATH = os.path.dirname(__file__)
EXAMPLES_PATH = os.path.join(THIS_FILE_PATH,'..'+os.sep+'examples')

@pytest.mark.slow
def test_examples_custom_zoom_video():
    cmnd = 'python ' + os.path.join(EXAMPLES_PATH,'custom_zoom_video.py')
    subprocess.check_output(cmnd)

def test_examples_image_prompt():
    cmnd = 'python ' + os.path.join(EXAMPLES_PATH,'image_prompt.py')
    subprocess.check_output(cmnd)

@pytest.mark.slow
def test_examples_multiple_same_prompt():
    cmnd = 'python ' + os.path.join(EXAMPLES_PATH,'multiple_same_prompt.py')
    subprocess.check_output(cmnd)

# @pytest.mark.slow
# def test_examples_restyle_video():
#     cmnd = 'python ' + os.path.join(EXAMPLES_PATH,'restyle_video.py')
#     subprocess.check_output(cmnd)

def test_examples_single_image():
    cmnd = 'python ' + os.path.join(EXAMPLES_PATH,'single_image.py')
    subprocess.check_output(cmnd)

@pytest.mark.slow
def test_examples_restyle_video_naive():
    cmnd = 'python ' + os.path.join(EXAMPLES_PATH,'restyle_video.py')
    subprocess.check_output(cmnd)

@pytest.mark.slow
def test_examples_zoom_video():
    cmnd = 'python ' + os.path.join(EXAMPLES_PATH,'style_transfer_exploration.py')
    subprocess.check_output(cmnd)
  
@pytest.mark.slow  
def test_examples_upscaling_video():
    cmnd = 'python ' + os.path.join(EXAMPLES_PATH,'upscaling_video.py')
    subprocess.check_output(cmnd)

@pytest.mark.slow
def test_examples_video():
    cmnd = 'python ' + os.path.join(EXAMPLES_PATH,'video.py')
    subprocess.check_output(cmnd)

@pytest.mark.slow
def test_examples_zoom_video():
    cmnd = 'python ' + os.path.join(EXAMPLES_PATH,'zoom_video.py')
    subprocess.check_output(cmnd)


