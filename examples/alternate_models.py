# Demonstrate how to use the alternate models
# Note that the Gumbel model requires an additional flag.

from vqgan_clip import generate
from vqgan_clip.engine import VQGAN_CLIP_Config
import os

config = VQGAN_CLIP_Config()


config.output_image_size = [684, 384]
text_prompts = 'A pastoral landscape painting by Rembrandt'


# model vqgan_imagenet_f16_16384 (the default)
config.vqgan_model_name = 'vqgan_imagenet_f16_16384'
config.vqgan_model_yaml_url = f'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1'
config.vqgan_model_ckpt_url = f'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1'
output_filename = f'example media{os.sep}vqgan_imagenet_f16_16384.jpg'
metadata_comment = generate.image(eng_config=config,
                                  text_prompts=text_prompts,
                                  iterations=400,
                                  output_filename=output_filename)


# model vqgan_imagenet_f16_1024
config.vqgan_model_name = 'vqgan_imagenet_f16_1024'
config.vqgan_model_yaml_url = f'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1'
config.vqgan_model_ckpt_url = f'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt&dl=1'
output_filename = f'example media{os.sep}vqgan_imagenet_f16_1024.jpg'
metadata_comment = generate.image(eng_config=config,
                                  text_prompts=text_prompts,
                                  iterations=400,
                                  output_filename=output_filename)

# model sflckr
config.vqgan_model_name = 'sflckr'
config.vqgan_model_yaml_url = f'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1'
config.vqgan_model_ckpt_url = f'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1'
output_filename = f'example media{os.sep}sflckr.jpg'
metadata_comment = generate.image(eng_config=config,
                                  text_prompts=text_prompts,
                                  iterations=400,
                                  output_filename=output_filename)

# model coco_transformer
config.vqgan_model_name = 'coco_transformer'
config.vqgan_model_yaml_url = f'https://dl.nmkd.de/ai/clip/coco/coco.yaml'
config.vqgan_model_ckpt_url = f'https://dl.nmkd.de/ai/clip/coco/coco.ckpt'
output_filename = f'example media{os.sep}coco_transformer.jpg'
metadata_comment = generate.image(eng_config=config,
                                  text_prompts=text_prompts,
                                  iterations=400,
                                  output_filename=output_filename)

