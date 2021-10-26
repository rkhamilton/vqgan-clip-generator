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
        init_image = [],
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

def test_single_image_image_prompt(testing_config, tmpdir):
    '''Generate a single image based on a image prompt
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output'))
    vqgan_clip.generate.single_image(config,
        image_prompts = IMAGE_PROMPTS,
        iterations = 5,
        output_filename = output_filename)
    output = output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)

def test_single_image_init_image(testing_config, tmpdir):
    '''Generate a single image based on a image prompt
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output'))
    init_image = IMAGE_1
    vqgan_clip.generate.single_image(config,
        text_prompts = 'A painting of flowers in the renaissance style',
        init_image = init_image,
        iterations = 5,
        output_filename = output_filename)
    output = output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)

def test_single_image_image_prompt_story(testing_config, tmpdir):
    '''Generate a single image based on a image prompt prompt
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output'))
    vqgan_clip.generate.single_image(config,
        image_prompts = IMAGE_PROMPTS,
        iterations = 100,
        output_filename = output_filename,
        change_prompt_every = 10)
    output = output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)

def test_single_image_all_prompts(testing_config, tmpdir):
    '''Generate a single image based on a text prompt, image prompt, and noise prompt.
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output'))
    vqgan_clip.generate.single_image(config,
        text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1',
        image_prompts = IMAGE_PROMPTS,
        noise_prompts = '123:0.1|234:0.2|345:0.3^700',
        iterations = 100,
        output_filename = output_filename)
    output = output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)

def test_single_image_all_prompts_story(testing_config, tmpdir):
    '''Generate a single image based on a text prompt, image prompt, and noise prompt.
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output'))
    vqgan_clip.generate.single_image(config,
        text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1',
        image_prompts = IMAGE_PROMPTS,
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

def test_video(testing_config, tmpdir):
    '''Generate a zoom video based on a text prompt changing every 10 iterations
    '''
    config = testing_config
    config.output_image_size = [128,128]
    steps_path = str(tmpdir.mkdir('video_frames'))
    iterations_per_frame = 15
    num_video_frames = 5

    vqgan_clip.generate.video_frames(eng_config=config,
        text_prompts = 'A painting of flowers in the renaissance style',
        image_prompts = [],
        noise_prompts = [],
        num_video_frames=num_video_frames,
        iterations_per_frame = iterations_per_frame,
        video_frames_path=steps_path, 
        zoom_scale=1.0, 
        shift_x=0, 
        shift_y=0)
    output_files = glob.glob(steps_path + os.sep + '*.png')
    assert len(output_files) == num_video_frames

    # test generating video
    output_filename = str(tmpdir.mkdir('output').join('output.mp4'))
    video_tools.encode_video(output_file=output_filename,
        path_to_stills=steps_path,
        metadata_title='a test comment',
        input_framerate=30)
    assert os.path.exists(output_filename)
    for f in output_files:
        os.remove(f)
    os.remove(output_filename)

def test_video_zoomed(testing_config, tmpdir):
    '''Generate a zoom video based on a text prompt changing every 10 iterations
    '''
    config = testing_config
    config.output_image_size = [128,128]
    steps_path = str(tmpdir.mkdir('video_frames'))
    iterations_per_frame = 15
    num_video_frames = 5

    vqgan_clip.generate.video_frames(eng_config=config,
        text_prompts = 'A painting of flowers in the renaissance style',
        image_prompts = [],
        noise_prompts = [],
        num_video_frames=num_video_frames,
        iterations_per_frame = iterations_per_frame,
        video_frames_path=steps_path, 
        zoom_scale=1.02, 
        shift_x=1, 
        shift_y=1)
    output_files = glob.glob(steps_path + os.sep + '*.png')
    assert len(output_files) == num_video_frames

    # test generating video
    output_filename = str(tmpdir.mkdir('output').join('output.mp4'))
    video_tools.encode_video(output_file=output_filename,
        path_to_stills=steps_path,
        metadata_title='a test comment',
        input_framerate=30)
    assert os.path.exists(output_filename)
    for f in output_files:
        os.remove(f)
    os.remove(output_filename)

def test_video_smoothed(testing_config, tmpdir):
    '''Generate a zoom video based on a text prompt changing every 10 iterations
    '''
    config = testing_config
    config.output_image_size = [128,128]
    steps_path = str(tmpdir.mkdir('video_frames'))
    iterations_per_frame = 15
    num_video_frames = 5

    vqgan_clip.generate.video_frames(eng_config=config,
        text_prompts = 'A painting of flowers in the renaissance style',
        image_prompts = [],
        noise_prompts = [],
        num_video_frames=num_video_frames,
        iterations_per_frame = iterations_per_frame,
        video_frames_path=steps_path, 
        zoom_scale=1.02, 
        shift_x=1, 
        shift_y=1,
        z_smoother=True)
    output_files = glob.glob(steps_path + os.sep + '*.png')
    assert len(output_files) == num_video_frames

    # test generating video
    output_filename = str(tmpdir.mkdir('output').join('output.mp4'))
    video_tools.encode_video(output_file=output_filename,
        path_to_stills=steps_path,
        metadata_title='a test comment',
        output_framerate=30,
        input_framerate=30)
    assert os.path.exists(output_filename)
    for f in output_files:
        os.remove(f)
    os.remove(output_filename)

def test_video_story_prompts(testing_config, tmpdir):
    '''Generate a zoom video based on a text prompt changing over time
    '''
    config = testing_config
    config.output_image_size = [128,128]
    steps_path = str(tmpdir.mkdir('video_frames'))
    iterations_per_frame = 15
    num_video_frames = 10

    vqgan_clip.generate.video_frames(eng_config=config,
        text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1',
        video_frames_path = steps_path,
        image_prompts = IMAGE_PROMPTS,
        noise_prompts = '123:0.1|234:0.2|345:0.3^700',
        change_prompts_on_frame = [3, 5, 9],
        num_video_frames=num_video_frames,
        iterations_per_frame = iterations_per_frame,
        zoom_scale=1.02, 
        shift_x=1, 
        shift_y=1,
        z_smoother=True)
    output_files = glob.glob(steps_path + os.sep + '*.png')
    assert len(output_files) == num_video_frames

    # test generating video
    output_filename = str(tmpdir.mkdir('output').join('output.mp4'))
    video_tools.encode_video(output_file=output_filename,
        path_to_stills=steps_path,
        metadata_title='a test comment',
        output_framerate=30,
        input_framerate=30)
    assert os.path.exists(output_filename)
    for f in output_files:
        os.remove(f)
    os.remove(output_filename)

def test_video_all_prompts(testing_config, tmpdir):
    '''Generate a zoom video based on a text prompt changing every 10 iterations
    '''
    config = testing_config
    config.output_image_size = [128,128]
    steps_path = str(tmpdir.mkdir('video_frames'))
    iterations_per_frame = 30
    num_video_frames = 5

    vqgan_clip.generate.video_frames(eng_config=config,
        text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1',
        image_prompts = IMAGE_PROMPTS,
        noise_prompts = '123:0.1|234:0.2|345:0.3^700',
        change_prompts_on_frame = [3, 5, 9],
        num_video_frames=num_video_frames,
        iterations_per_frame = iterations_per_frame,
        video_frames_path = steps_path,
        zoom_scale=1.02, 
        shift_x=1, 
        shift_y=1,
        z_smoother=True)
    output_files = glob.glob(steps_path + os.sep + '*')
    assert len(output_files) == num_video_frames

    # test generating video
    output_filename = str(tmpdir.mkdir('output').join('output.mp4'))
    video_tools.encode_video(output_file=output_filename,
        path_to_stills=steps_path,
        metadata_title='a test comment',
        output_framerate=30,
        input_framerate=30)
    assert os.path.exists(output_filename)
    for f in output_files:
        os.remove(f)
    os.remove(output_filename)

def test_restyle_video_naive(testing_config, tmpdir):
    output_images_path = str(tmpdir.mkdir('video_frames'))
    original_video_frames = video_tools.extract_video_frames(TEST_VIDEO, 
        extraction_framerate = 2,
        extracted_video_frames_path=output_images_path)

    # Restyle the video by applying VQGAN to each frame independently
    generated_video_frames_path = str(tmpdir.mkdir('generated_video_frames'))
    vqgan_clip.generate.restyle_video_frames_naive(original_video_frames,
            eng_config=testing_config,
            text_prompts = 'a red rose|a fish^the last horse',
            iterations = 5,
            save_every=None,
            generated_video_frames_path = generated_video_frames_path)

    output_files = glob.glob(generated_video_frames_path + os.sep + '*.png')
    assert len(output_files) > 0
    assert len(output_files) == len(original_video_frames)
    for f in output_files:
        os.remove(f)
    for f in original_video_frames:
        os.remove(f)

def test_restyle_video(testing_config, tmpdir):
    output_images_path = str(tmpdir.mkdir('video_frames'))
    original_video_frames = video_tools.extract_video_frames(TEST_VIDEO, 
        extraction_framerate = 2,
        extracted_video_frames_path=output_images_path)

    # Restyle the video by applying VQGAN to each frame independently
    generated_video_frames_path = str(tmpdir.mkdir('generated_video_frames'))
    vqgan_clip.generate.restyle_video_frames(original_video_frames,
            eng_config=testing_config,
            text_prompts = 'a red rose|a fish^the last horse',
            iterations = 5,
            save_every=None,
            generated_video_frames_path = generated_video_frames_path,
            current_source_frame_prompt_weight=0.1,
            previous_generated_frame_prompt_weight=0.1,
            generated_frame_init_blend=0.1)

    output_files = glob.glob(generated_video_frames_path + os.sep + '*.png')
    assert len(output_files) > 0
    assert len(output_files) == len(original_video_frames)
    for f in output_files:
        os.remove(f)
    for f in original_video_frames:
        os.remove(f)

def test_restyle_video_smoothed(testing_config, tmpdir):
    output_images_path = str(tmpdir.mkdir('video_frames'))
    original_video_frames = video_tools.extract_video_frames(TEST_VIDEO, 
        extraction_framerate = 2,
        extracted_video_frames_path=output_images_path)

    # Restyle the video by applying VQGAN to each frame independently
    generated_video_frames_path = str(tmpdir.mkdir('generated_video_frames'))
    vqgan_clip.generate.restyle_video_frames(original_video_frames,
            eng_config=testing_config,
            text_prompts = 'a red rose|a fish^the last horse',
            iterations = 5,
            save_every=None,
            generated_video_frames_path = generated_video_frames_path,
            current_source_frame_prompt_weight=0.1,
            previous_generated_frame_prompt_weight=0.1,
            generated_frame_init_blend=0.1,
            z_smoother=True)

    output_files = glob.glob(generated_video_frames_path + os.sep + '*.png')
    assert len(output_files) > 0
    assert len(output_files) == len(original_video_frames)
    for f in output_files:
        os.remove(f)
    for f in original_video_frames:
        os.remove(f)

def test_single_image_no_folder(testing_config, tmpdir):
    '''Generate a single image based on a text prompt
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str('output')
    vqgan_clip.generate.single_image(config,
        text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1',
        image_prompts = [],
        noise_prompts = [],
        init_image = [],
        iterations = 5,
        save_every = 50,
        output_filename = output_filename,
        change_prompt_every = 0)
    output = output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)

def test_single_image_output_jpg(testing_config, tmpdir):
    '''Generate a single image based on a text prompt
    '''
    config = testing_config
    config.output_image_size = [128,128]
    output_filename = str(tmpdir.mkdir('output').join('output'))
    vqgan_clip.generate.single_image(config,
        text_prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1',
        image_prompts = [],
        noise_prompts = [],
        init_image = [],
        iterations = 5,
        save_every = 50,
        output_filename = output_filename+'.jpg',
        change_prompt_every = 0)
    output = output_filename + '.png'
    assert os.path.exists(output)
    os.remove(output)