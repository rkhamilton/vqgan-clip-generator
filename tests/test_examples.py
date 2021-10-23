import pytest
import subprocess, os

THIS_FILE_PATH = os.path.dirname(__file__)
EXAMPLES_PATH = os.path.join(THIS_FILE_PATH,'..'+os.sep+'examples')

def test_examples():
    '''Confirm all example scripts run without error
    '''
    examples = ['single_image.py',
                'custom_zoom_video.py',
                'image_prompt.py',
                'multiple_same_prompt.py',
                'restyle_video_naive.py',
                'restyle_video.py',
                'single_image.py',
                'upscaling_video.py',
                'video.py',
                'zoom_video.py']    
    for example in examples:
        cmnd = 'python ' + os.path.join(EXAMPLES_PATH,example)
        # subprocess.call(cmnd, shell=True)
        subprocess.check_output(cmnd)