# This module is the interface for creating images and video from text prompts
from vqgan_clip.engine import Engine, VQGAN_CLIP_Config
from tqdm import tqdm

def single_image(eng_config=VQGAN_CLIP_Config()):
    eng = Engine(eng_config)
    eng.initialize_VQGAN_CLIP() 
    eng.parse_all_prompts()
    eng.encode_and_append_prompts(0)
    eng.configure_optimizer()

    # generate the image
    try:
        for iteration_num in tqdm(range(1,eng.conf.iterations+1)):
            #perform eng.conf.iterations of train()
            lossAll = eng.train(iteration_num)
            if iteration_num % eng_config.save_every == 0:
                # display some statistics about how the GAN training is going whever we save an interim image
                losses_str = ', '.join(f'{loss.item():7.3f}' for loss in lossAll)
                tqdm.write(f'iteration:{iteration_num:6d}\tloss sum: {sum(lossAll).item():7.3f}\tloss for all prompts:{losses_str}')
                # save an interim copy of the image so you can look at it as it changes if you like
                eng.save_current_output(eng.conf.output_filename) 

                # # if making a video, save a frame named for the video step
                # if eng.conf.make_video:
                #     eng.save_current_output('./steps/' + str(iteration_num) + '.png') 
        # Always save the output at the end
        eng.save_current_output(eng.conf.output_filename) 
    except KeyboardInterrupt:
        pass
