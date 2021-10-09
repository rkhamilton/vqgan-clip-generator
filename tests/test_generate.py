import pytest
import vqgan_clip.generate
from vqgan_clip.engine import VQGAN_CLIP_Config
import os



@pytest.fixture
def testing_config(tmpdir_factory):
    config = VQGAN_CLIP_Config()
    config.text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1'
    config.vqgan_checkpoint = f'C:\\Users\\ryanh\\Documents\\src\\vqgan_lib_dev\\models\\vqgan_imagenet_f16_16384.ckpt'
    config.vqgan_config = f'C:\\Users\\ryanh\\Documents\\src\\vqgan_lib_dev\\models\\vqgan_imagenet_f16_16384.yaml'
    config.iterations = 5
    config.output_filename = str(tmpdir_factory.mktemp('output').join('output.png'))
    return config

def test_generate_single_image(testing_config):
    '''Generate a single image based on a text prompt
    '''
    config = testing_config
    vqgan_clip.generate.single_image(config)
    assert os.path.exists(config.output_filename)
    os.remove(config.output_filename)

def test_generate_single_image_story(testing_config):
    '''Generate a single image based on a text prompt changing every 10 iterations
    '''
    config = testing_config
    config.change_prompt_every = 10
    config.iterations = 100
    vqgan_clip.generate.single_image(config)
    assert os.path.exists(config.output_filename)
    os.remove(config.output_filename)

def test_generate_single_image_noise_prompt(testing_config):
    '''Generate a single image based on a noise prompt
    '''
    config = testing_config
    config.text_prompts = []
    config.noise_prompts = '123:0.1|234:0.2|345:0.3'
    vqgan_clip.generate.single_image(config)
    assert os.path.exists(config.output_filename)
    os.remove(config.output_filename)


def test_generate_single_image_noise_prompt_story(testing_config):
    '''Generate a single image based on a noise prompt changing every 10 iterations
    '''
    config = testing_config
    config.text_prompts = []
    config.noise_prompts = '123:0.1|234:0.2|345:0.3'
    config.change_prompt_every = 10
    config.iterations = 100
    vqgan_clip.generate.single_image(config)
    assert os.path.exists(config.output_filename)
    os.remove(config.output_filename)

def test_generate_single_image_image_prompt(testing_config):
    assert False

# TODO add tests for noise prompts only, image only, text only, combinations
