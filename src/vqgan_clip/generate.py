# This module is the interface for creating images and video from text prompts
from vqgan_clip.engine import Engine, EngineConfig
from tqdm import tqdm

def single_image(config):
    eng = Engine()
    eng.conf = config

    # load the models
    eng.initialize_VQGAN_CLIP()       
    
    # CLIP tokenize/encode prompts from text, input images, and noise parameters
    text_prompts, story_phrases_all_prompts = eng.parse_story_prompts(eng.conf.text_prompts)
    for prompt in text_prompts:
        eng.encode_and_append_text_prompt(prompt)
    
    # Split target images using the pipe character (weights are split later)
    image_prompts, image_prompts_all = eng.parse_story_prompts(eng.conf.image_prompts)
    # if we had image prompts, encode them with CLIP
    for prompt in image_prompts:
        eng.encode_and_append_image_prompt(prompt)

    # Split noise prompts using the pipe character (weights are split later)
    noise_prompts, noise_prompts_all = eng.parse_story_prompts(eng.conf.image_prompts)
    for prompt in noise_prompts:
        eng.encode_and_append_noise_prompt(prompt)

    # generate the image
    eng.configure_optimizer()
    try:
        for iteration_num in tqdm(range(1,eng.conf.iterations+1)):
            eng.train(iteration_num)
    except KeyboardInterrupt:
        pass
