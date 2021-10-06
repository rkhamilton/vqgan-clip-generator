# download datasets for VQGAN-CLIP
import urllib.request

def imagenet_f16_16384():
    # TODO fix this it's not downloading
    urllib.request.urlretrieve(
        'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1',
        'checkpoints/vqgan_imagenet_f16_16384.yaml')

    urllib.request.urlretrieve(
        'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1',
        'checkpoints/vqgan_imagenet_f16_16384.ckpt')

