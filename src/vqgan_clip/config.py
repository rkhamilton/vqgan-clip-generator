# A set of configuration options used to execute VQGAN+CLIP.

config = {
    'prompt' : 'A painting of flowers in the renaissance style', # text prompt used to generate an image
    'image_prompts' : None, # an input image can be parsed by CLIP and used as a prompt to generate another image
    'iterations' : 500, # number of iterations of train() to perform before stopping.
    'save_every' : 50, # an interim image will be saved to the output location at an iteration interval defined here
    'output_image_size' : [256,256], # x/y dimensions of the output image in pixels.
    'init_image' : None, # a seed image that can be used to start the training. Without an initial image, random noise will be used.
    'init_weight' : 0.0,
    'clip_model' : 'ViT-B/32',
    'vqgan_config' : f'checkpoints/vqgan_imagenet_f16_16384.yaml',
    'vqgan_checkpoint' : f'checkpoints/vqgan_imagenet_f16_16384.ckpt',
    'noise_prompt_seeds' : [],
    'noise_prompt_weights' : [],
    'learning_rate' : 0.1,
    'cut_method' : 'latest',
    'num_cuts' : 32,
    'cut_power' : 1.0,
    'seed' : None,
    'optimiser' : 'Adam',
    'output_filename' : 'output.png',
    'make_video' : False,
    'make_zoom_video' : False,
    'zoom_start' : 0,
    'zoom_save_every' : 50,
    'zoom_scale' : 1.02,
    'zoom_shift_x' : 0,
    'zoom_shift_y' : 0,
    'change_prompt_every' : 0,
    'output_video_fps' : 60,
    'input_video_fps' : 15,
    'cudnn_determinism' : False,
    'augments' : [],
    'video_style_dir' : None,
    'cuda_device' : 'cuda:0'
}