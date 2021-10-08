# This module is the interface for creating images and video from text prompts
from vqgan_clip.engine import Engine, EngineConfig
from tqdm import tqdm

save_every = 50 # an interim image will be saved to the output location every save_every iterations


def single_image(eng_config=EngineConfig()):
    eng = Engine()
    eng.conf = eng_config

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
            lossAll = eng.train(iteration_num)
            if iteration_num % save_every == 0:
                # TODO move this to outer loop
                losses_str = ', '.join(f'{loss.item():g}' for loss in lossAll)
                tqdm.write(f'i: {iteration_num}, loss: {sum(lossAll).item():g}, lossAll: {losses_str}')
                eng.save_current_output(eng.conf.output_filename) 

                # # if making a video, save a frame named for the video step
                # if eng.conf.make_video:
                #     eng.save_current_output('./steps/' + str(iteration_num) + '.png') 
    except KeyboardInterrupt:
        pass
