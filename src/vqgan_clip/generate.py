# This module is the interface for creating images and video from text prompts
from vqgan_clip.engine import Engine, VQGAN_CLIP_Config
from tqdm import tqdm

def single_image(eng_config=VQGAN_CLIP_Config()):
    eng = Engine(eng_config)
    eng.initialize_VQGAN_CLIP() 
    
    eng.parse_all_prompts()
    eng.encode_and_append_prompts(0)

    # generate the image
    eng.configure_optimizer()
    try:
        for iteration_num in tqdm(range(1,eng.conf.iterations+1)):
            lossAll = eng.train(iteration_num)
            if iteration_num % eng_config.save_every == 0:
                # TODO move this to outer loop
                losses_str = ', '.join(f'{loss.item():g}' for loss in lossAll)
                tqdm.write(f'i: {iteration_num}, loss: {sum(lossAll).item():g}, lossAll: {losses_str}')
                eng.save_current_output(eng.conf.output_filename) 

                # # if making a video, save a frame named for the video step
                # if eng.conf.make_video:
                #     eng.save_current_output('./steps/' + str(iteration_num) + '.png') 
    except KeyboardInterrupt:
        pass


