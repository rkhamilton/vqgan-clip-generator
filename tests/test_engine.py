import pytest
from vqgan_clip.engine import Engine



# generate a single image based on a text prompt
def test_generate_single_image(prompt_text):
    eng = Engine()
    eng.conf.prompts = 'This is a field of flowers'
    eng.do_it()
    