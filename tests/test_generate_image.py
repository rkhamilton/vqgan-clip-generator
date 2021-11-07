import pytest
import vqgan_clip.generate
from vqgan_clip.engine import VQGAN_CLIP_Config
import os
import vqgan_clip._functional as VF
import glob
from vqgan_clip import video_tools

@pytest.fixture
def testing_config():
    config = VQGAN_CLIP_Config()
    config.vqgan_checkpoint = f'C:\\Users\\ryanh\\Documents\\src\\vqgan_lib_dev\\models\\vqgan_imagenet_f16_16384.ckpt'
    config.vqgan_config = f'C:\\Users\\ryanh\\Documents\\src\\vqgan_lib_dev\\models\\vqgan_imagenet_f16_16384.yaml'
    return config

TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files',
    )

IMAGE_1 = os.path.join(TEST_DATA_DIR,'prompt1.jpg')
IMAGE_2 = os.path.join(TEST_DATA_DIR,'prompt2.jpg')
IMAGE_PROMPTS = f'{IMAGE_1}:0.5|{IMAGE_2}:0.5'
TEST_VIDEO = os.path.join(TEST_DATA_DIR,'small.mp4')

def test_image_invalid_input(testing_config, tmpdir):
    '''Confirm we get an exception when given invalid prompts
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output.png'))
    with pytest.raises(ValueError, match='text_prompts must be a string'):
        vqgan_clip.generate.image(eng_config=config,
            text_prompts = 3,
            image_prompts = [],
            noise_prompts = [],
            init_image = [],
            iterations = 5,
            save_every = 50,
            output_filename = output_filename)
    with pytest.raises(ValueError, match='image_prompts must be a string'):
        vqgan_clip.generate.image(eng_config=config,
            text_prompts = [],
            image_prompts = 3,
            noise_prompts = [],
            init_image = [],
            iterations = 5,
            save_every = 50,
            output_filename = output_filename)
    with pytest.raises(ValueError, match='noise_prompts must be a string'):
        vqgan_clip.generate.image(eng_config=config,
            text_prompts = [],
            image_prompts = [],
            noise_prompts = 3,
            init_image = [],
            iterations = 5,
            save_every = 50,
            output_filename = output_filename)
    init_image = output_filename
    with pytest.raises(ValueError, match=f'init_image does not exist.'):
        vqgan_clip.generate.image(eng_config=config,
            text_prompts = [],
            image_prompts = [],
            noise_prompts = [],
            init_image = init_image,
            iterations = 5,
            save_every = 50,
            output_filename = output_filename)
    with pytest.raises(ValueError, match=f'save_every must be an int.'):
        vqgan_clip.generate.image(eng_config=config,
            text_prompts = 'test prompt',
            image_prompts = [],
            noise_prompts = [],
            init_image = [],
            iterations = 5,
            save_every = [50],
            output_filename = output_filename)
    with pytest.raises(ValueError, match='No valid prompts were provided'):
        vqgan_clip.generate.image(eng_config=config,
            text_prompts =  [],
            image_prompts = [],
            noise_prompts = [],
            init_image = [],
            iterations = 5,
            save_every = 50,
            output_filename = output_filename)

def test_image_png(testing_config, tmpdir):
    '''Generate a single png image based on a text prompt
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output.png'))
    vqgan_clip.generate.image(eng_config=config,
        text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1',
        image_prompts = [],
        noise_prompts = [],
        init_image = [],
        iterations = 5,
        save_every = 50,
        output_filename = output_filename)
    assert os.path.exists(output_filename)
    os.remove(output_filename)

def test_image_bmp(testing_config, tmpdir):
    '''Generate a single bmp image based on a text prompt. Testing a format with no package metadata support.
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output.bmp'))
    vqgan_clip.generate.image(eng_config=config,
        text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1',
        image_prompts = [],
        noise_prompts = [],
        init_image = [],
        iterations = 5,
        save_every = 50,
        output_filename = output_filename)
    assert os.path.exists(output_filename)
    os.remove(output_filename)

def test_image_jpg_save_every(testing_config, tmpdir):
    '''Generate a single jpg image based on a text prompt
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output.jpg'))
    vqgan_clip.generate.image(eng_config=config,
        text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1',
        image_prompts = [],
        noise_prompts = [],
        init_image = [],
        iterations = 5,
        save_every = 2,
        output_filename = output_filename)
    assert os.path.exists(output_filename)
    os.remove(output_filename)

def test_image_jpg(testing_config, tmpdir):
    '''Generate a single jpg image based on a text prompt, no save every
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output.jpg'))
    vqgan_clip.generate.image(eng_config=config,
        text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1',
        image_prompts = [],
        noise_prompts = [],
        init_image = [],
        iterations = 5,
        output_filename = output_filename)
    assert os.path.exists(output_filename)
    os.remove(output_filename)

def test_image_story(testing_config, tmpdir):
    '''Generate a single image based on a text prompt changing every 10 iterations
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output.jpg'))
    vqgan_clip.generate.image(eng_config=config,
        text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1',
        iterations = 100,
        save_every = 50,
        output_filename = output_filename)
    assert os.path.exists(output_filename)
    os.remove(output_filename)

def test_image_noise_prompt(testing_config, tmpdir):
    '''Generate a single image based on a noise prompt with save_every==50
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output.jpg'))
    vqgan_clip.generate.image(eng_config=config,
        noise_prompts = '123:0.1|234:0.2|345:0.3',
        iterations = 100,
        save_every = 50,
        output_filename = output_filename)
    assert os.path.exists(output_filename)
    os.remove(output_filename)

def test_image_noise_prompt_story(testing_config, tmpdir):
    '''Generate a single image based on a noise prompt no save every
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output.jpg'))
    vqgan_clip.generate.image(eng_config=config,
        noise_prompts = '123:0.1|234:0.2|345:0.3^700',
        iterations = 100,
        output_filename = output_filename)
    assert os.path.exists(output_filename)
    os.remove(output_filename)

def test_image_image_prompt(testing_config, tmpdir):
    '''Generate a single image based on a image prompt
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output.jpg'))
    vqgan_clip.generate.image(eng_config=config,
        image_prompts = IMAGE_PROMPTS,
        iterations = 5,
        output_filename = output_filename)
    assert os.path.exists(output_filename)
    os.remove(output_filename)

def test_image_init_image(testing_config, tmpdir):
    '''Generate a single image based on a image prompt
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output.jpg'))
    init_image = IMAGE_1
    vqgan_clip.generate.image(eng_config=config,
        text_prompts = 'A painting of flowers in the renaissance style',
        init_image = init_image,
        iterations = 5,
        output_filename = output_filename)
    assert os.path.exists(output_filename)
    os.remove(output_filename)

def test_image_init_image_weight(testing_config, tmpdir):
    '''Generate a single image based on a image prompt
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output.jpg'))
    init_image = IMAGE_1
    vqgan_clip.generate.image(eng_config=config,
        text_prompts = 'A painting of flowers in the renaissance style',
        init_image = init_image,
        init_weight= 0.5,
        iterations = 5,
        output_filename = output_filename)
    assert os.path.exists(output_filename)
    os.remove(output_filename)

def test_image_all_prompts(testing_config, tmpdir):
    '''Generate a single image based on a text prompt, image prompt, and noise prompt.
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output.jpg'))
    vqgan_clip.generate.image(eng_config=config,
        text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1',
        image_prompts = IMAGE_PROMPTS,
        noise_prompts = '123:0.1|234:0.2|345:0.3^700',
        iterations = 100,
        output_filename = output_filename)
    assert os.path.exists(output_filename)
    os.remove(output_filename)

def test_image_all_prompts_story(testing_config, tmpdir):
    '''Generate a single image based on a text prompt, image prompt, and noise prompt.
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output.jpg'))
    vqgan_clip.generate.image(eng_config=config,
        text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1',
        image_prompts = IMAGE_PROMPTS,
        noise_prompts = '123:0.1|234:0.2|345:0.3^700',
        iterations = 100,
        output_filename = output_filename)
    assert os.path.exists(output_filename)
    os.remove(output_filename)

def test_image_no_folder(testing_config):
    '''Output filename specified without a folder
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str('output.jpg')
    vqgan_clip.generate.image(eng_config=config,
        text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1',
        image_prompts = [],
        noise_prompts = [],
        init_image = [],
        iterations = 5,
        save_every = 50,
        output_filename = output_filename)
    assert os.path.exists(output_filename)
    os.remove(output_filename)

