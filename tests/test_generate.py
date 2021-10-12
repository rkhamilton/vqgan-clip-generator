import pytest
import vqgan_clip.generate
from vqgan_clip.engine import VQGAN_CLIP_Config
import os
import vqgan_clip._functional as VF
import glob

@pytest.fixture
def testing_config():
    config = VQGAN_CLIP_Config()
    config.vqgan_checkpoint = f'C:\\Users\\ryanh\\Documents\\src\\vqgan_lib_dev\\models\\vqgan_imagenet_f16_16384.ckpt'
    config.vqgan_config = f'C:\\Users\\ryanh\\Documents\\src\\vqgan_lib_dev\\models\\vqgan_imagenet_f16_16384.yaml'
    return config

@pytest.fixture
def image_prompts():
    return f'\\Users\\ryanh\\Documents\src\\vqgan_lib_dev\\vqgan-clip-generator\\tests\\images\\prompt1.jpg:0.5|\\Users\\ryanh\\Documents\src\\vqgan_lib_dev\\vqgan-clip-generator\\tests\\images\\prompt2.jpg:0.3'


def test_single_image(testing_config, tmpdir):
    '''Generate a single image based on a text prompt
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output'))
    vqgan_clip.generate.single_image(config,
        text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1',
        image_prompts = [],
        noise_prompts = [],
        iterations = 5,
        save_every = 50,
        output_filename = output_filename,
        change_prompt_every = 0)
    output = output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)

def test_single_image_story(testing_config, tmpdir):
    '''Generate a single image based on a text prompt changing every 10 iterations
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output'))
    vqgan_clip.generate.single_image(config,
        text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1',
        iterations = 100,
        save_every = 50,
        output_filename = output_filename ,
        change_prompt_every = 10)
    output = output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)

def test_single_image_noise_prompt(testing_config, tmpdir):
    '''Generate a single image based on a noise prompt
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output'))
    vqgan_clip.generate.single_image(config,
        noise_prompts = '123:0.1|234:0.2|345:0.3',
        iterations = 100,
        save_every = 50,
        output_filename = output_filename)
    output = output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)

def test_single_image_noise_prompt_story(testing_config, tmpdir):
    '''Generate a single image based on a noise prompt changing every 10 iterations
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output'))
    vqgan_clip.generate.single_image(config,
        noise_prompts = '123:0.1|234:0.2|345:0.3^700',
        iterations = 100,
        output_filename = output_filename ,
        change_prompt_every = 10)
    output = output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)

def test_single_image_image_prompt(testing_config, image_prompts, tmpdir):
    '''Generate a single image based on a image prompt
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output'))
    vqgan_clip.generate.single_image(config,
        image_prompts = image_prompts,
        iterations = 5,
        output_filename = output_filename)
    output = output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)

def test_single_image_image_prompt_story(testing_config, image_prompts, tmpdir):
    '''Generate a single image based on a image prompt prompt
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output'))
    vqgan_clip.generate.single_image(config,
        image_prompts = image_prompts,
        iterations = 100,
        output_filename = output_filename,
        change_prompt_every = 10)
    output = output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)

def test_single_image_all_prompts(testing_config, image_prompts, tmpdir):
    '''Generate a single image based on a text prompt, image prompt, and noise prompt.
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output'))
    vqgan_clip.generate.single_image(config,
        text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1',
        image_prompts = image_prompts,
        noise_prompts = '123:0.1|234:0.2|345:0.3^700',
        iterations = 100,
        output_filename = output_filename)
    output = output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)

def test_single_image_all_prompts_story(testing_config, image_prompts, tmpdir):
    '''Generate a single image based on a text prompt, image prompt, and noise prompt.
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output'))
    vqgan_clip.generate.single_image(config,
        text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1',
        image_prompts = image_prompts,
        noise_prompts = '123:0.1|234:0.2|345:0.3^700',
        iterations = 100,
        output_filename = output_filename,
        change_prompt_every = 10)
    output = output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)

def test_multiple_images(testing_config, tmpdir):
    '''Generate multiple images based on the same text prompt, each with a different random seed
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_images_path = str(tmpdir.mkdir('video_frames'))
    num_images_to_generate = 3
    vqgan_clip.generate.multiple_images(config,
        text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1',
        iterations = 5,
        save_every = 6,
        num_images_to_generate=num_images_to_generate,
        output_images_path=output_images_path)
    output_files = glob.glob(output_images_path + os.sep + '*.png')
    assert len(output_files) == num_images_to_generate
    for f in output_files:
        os.remove(f)

def test_video_single_prompt(testing_config, tmpdir):
    '''Generate a video file based on a text prompt
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output'))
    steps_path = str(tmpdir.mkdir('video_frames'))
    vqgan_clip.generate.video(config,
        text_prompts = 'A painting of flowers in the renaissance style',
        iterations = 100,
        save_every = 10,
        output_filename = output_filename,
        change_prompt_every = 300,
        video_frames_path=steps_path, 
        output_framerate=30, 
        assumed_input_framerate=5)
    assert os.path.exists(output_filename + '.mp4')
    os.remove(output_filename + '.mp4')

def test_video_single_prompt_interpolation(testing_config, tmpdir):
    '''Generate a video file based on a text prompt and interpolate to a higher framerate
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output'))
    steps_path = str(tmpdir.mkdir('video_frames'))
    vqgan_clip.generate.video(config,
        text_prompts = 'A painting of flowers in the renaissance style',
        iterations = 100,
        save_every = 10,
        output_filename = output_filename,
        video_frames_path=steps_path, 
        output_framerate=30, 
        assumed_input_framerate=5)
    assert os.path.exists(output_filename + '.mp4')
    os.remove(output_filename + '.mp4')


def test_video_multiple_prompt_interpolation(testing_config, tmpdir):
    '''Generate a video file based on a text prompt and interpolate to a higher framerate
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output'))
    steps_path = str(tmpdir.mkdir('video_frames'))
    vqgan_clip.generate.video(config,
        text_prompts = 'A painting of flowers in the renaissance style^a black dog in a cave',
        iterations = 300,
        save_every = 10,
        output_filename = output_filename,
        change_prompt_every = 100,
        video_frames_path=steps_path, 
        output_framerate=30, 
        assumed_input_framerate=5)
    assert os.path.exists(output_filename + '.mp4')
    os.remove(output_filename + '.mp4')


def test_zoom_video(testing_config, tmpdir):
    '''Generate a zoom video based on a text prompt changing every 10 iterations
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output'))
    steps_path = str(tmpdir.mkdir('video_frames'))

    vqgan_clip.generate.zoom_video(config,
        text_prompts = 'A painting of flowers in the renaissance style',
        image_prompts = [],
        noise_prompts = [],
        iterations = 200,
        save_every = 5,
        output_filename = output_filename,
        change_prompt_every = 50,
        video_frames_path=steps_path, 
        output_framerate=30, 
        assumed_input_framerate=5,
        zoom_scale=1.02, 
        shift_x=1, 
        shift_y=1)
    assert os.path.exists(output_filename + '.mp4')
    os.remove(output_filename + '.mp4')

def test_zoom_video_all_prompts(testing_config, image_prompts, tmpdir):
    '''Generate a zoom video based on a text prompt changing every 10 iterations
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output'))
    steps_path = str(tmpdir.mkdir('video_frames'))

    vqgan_clip.generate.zoom_video(config,
        text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1',
        image_prompts = image_prompts,
        noise_prompts = '123:0.1|234:0.2|345:0.3^700',
        iterations = 200,
        save_every = 5,
        output_filename = output_filename,
        change_prompt_every = 50,
        video_frames_path=steps_path, 
        output_framerate=30, 
        assumed_input_framerate=5,
        zoom_scale=1.02, 
        shift_x=1, 
        shift_y=1)
    assert os.path.exists(output_filename + '.mp4')
    os.remove(output_filename + '.mp4')