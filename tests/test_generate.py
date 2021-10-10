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

@pytest.fixture
def image_prompts():
    return f'\\Users\\ryanh\\Documents\src\\vqgan_lib_dev\\vqgan-clip-generator\\tests\\images\\prompt1.jpg:0.5|\\Users\\ryanh\\Documents\src\\vqgan_lib_dev\\vqgan-clip-generator\\tests\\images\\prompt2.jpg:0.3'

def test_single_image(testing_config):
    '''Generate a single image based on a text prompt
    '''
    config = testing_config
    vqgan_clip.generate.single_image(config)
    output = config.output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)

def test_single_image_story(testing_config):
    '''Generate a single image based on a text prompt changing every 10 iterations
    '''
    config = testing_config
    config.change_prompt_every = 10
    config.iterations = 100
    vqgan_clip.generate.single_image(config)
    output = config.output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)

def test_single_image_noise_prompt(testing_config):
    '''Generate a single image based on a noise prompt
    '''
    config = testing_config
    config.text_prompts = []
    config.noise_prompts = '123:0.1|234:0.2|345:0.3'
    vqgan_clip.generate.single_image(config)
    output = config.output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)

def test_single_image_noise_prompt_story(testing_config):
    '''Generate a single image based on a noise prompt changing every 10 iterations
    '''
    config = testing_config
    config.text_prompts = []
    config.noise_prompts = '123:0.1|234:0.2|345:0.3'
    config.change_prompt_every = 10
    config.iterations = 100
    vqgan_clip.generate.single_image(config)
    output = config.output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)

def test_single_image_image_prompt(testing_config, image_prompts):
    '''Generate a single image based on a image prompt prompt
    '''
    config = testing_config
    config.text_prompts = []
    config.image_prompts = image_prompts
    vqgan_clip.generate.single_image(config)
    output = config.output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)

def test_single_image_image_prompt_story(testing_config, image_prompts):
    '''Generate a single image based on a image prompt prompt
    '''
    config = testing_config
    config.text_prompts = []
    config.image_prompts = image_prompts
    config.change_prompt_every = 10
    config.iterations = 100
    vqgan_clip.generate.single_image(config)
    output = config.output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)

def test_single_image_all_prompts(testing_config, image_prompts):
    '''Generate a single image based on a text prompt, image prompt, and noise prompt.
    '''
    config = testing_config
    config.text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1'
    config.noise_prompts = '123:0.1|234:0.2|345:0.3'
    config.image_prompts = image_prompts
    vqgan_clip.generate.single_image(config)
    output = config.output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)

def test_video_single_prompt(testing_config):
    '''Generate a video file based on a text prompt
    '''
    config = testing_config
    config.text_prompts = 'A painting of flowers in the renaissance style'
    config.output_filename = 'output' + os.sep + 'output'
    config.save_every = 10
    config.iterations = 100
    config.output_image_size = [128,128]

    vqgan_clip.generate.video(config)
    assert os.path.exists(config.output_filename + '.mp4')
    os.remove(config.output_filename + '.mp4')

def test_video_single_prompt_interpolation(testing_config):
    '''Generate a video file based on a text prompt and interpolate to a higher framerate
    '''
    config = testing_config
    config.text_prompts = 'A painting of flowers in the renaissance style'
    config.output_filename = 'output' + os.sep + 'output'
    config.save_every = 10
    config.iterations = 100
    config.output_image_size = [128,128]

    vqgan_clip.generate.video(config,video_frames_path='./steps', output_framerate=30, assumed_input_framerate=5)
    assert os.path.exists(config.output_filename + '.mp4')
    os.remove(config.output_filename + '.mp4')