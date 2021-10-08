# This module is the interface for creating images and video from text prompts
from vqgan_clip.engine import Engine

def single_image(prompt_text):
    eng = Engine()
    eng.set('prompt',prompt_text)
    print(eng.config('prompt'))
    eng.do_it()
