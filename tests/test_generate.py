import pytest
import vqgan_clip.generate
import os

@pytest.fixture
def prompt_text():
    return 'A field of flowers'


# generate a single image based on a text prompt
def test_generate_single_image(prompt_text):
    vqgan_clip.generate.single_image(prompt_text)
    output_file = 'outputs'+os.sep+'output.png'
    assert os.path.exists(output_file)
    os.remove(output_file)