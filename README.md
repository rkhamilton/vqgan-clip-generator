# VQGAN-CLIP-GENERATOR Overview

A package for running VQGAN+CLIP locally. This package was a complete refactor of the code provided by [NerdyRodent](https://github.com/nerdyrodent/), which started out as a Katherine Crowson VQGAN+CLIP derived Google colab notebook.

In addition to refactoring NerdyRodent's code into a pythonic package to improve usability, this project adds unit tests, and adds improvements to the ability to restyle an existing video.

Original notebook: [![Open In Colab][colab-badge]][colab-notebook]

[colab-notebook]: <https://colab.research.google.com/drive/1ZAus_gn2RhTZWzOWUpPERNC0Q8OhZRTZ>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

[NerdyRodent VQGAN+CLIP repository](https://github.com/nerdyrodent/)

Some example images:

<img src="./samples/A child throwing the ducks into a wood chipper painting by Rembrandt initial.png" width="256px"></img>
<img src="./samples/Pastoral landscape painting in the impressionist style initial.png" width="256px"></img>
<img src="./samples/The_sadness_of_Colonel_Sanders_by_Thomas_Kinkade.png" width="256px"></img>

Environment:

* Tested on Windows 10 build 19043
* GPU: Nvidia RTX 3080
* Typical VRAM requirements:
  * 24 GB for a 900x900 image
  * 10 GB for a 512x512 image
  * 8 GB for a 380x380 image

## Set up

This example uses [Anaconda](https://www.anaconda.com/products/individual#Downloads) to manage virtual Python environments.

Create a new virtual Python environment for VQGAN-CLIP-GENERATOR. Then, install the VQGAN-CLIP-GENERATOR package using pip.

```sh
conda create --name vqgan python=3.9 pip numpy pytest tqdm git pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda activate vqgan
pip install git+https://github.com/openai/CLIP.git taming-transformers ftfy regex tqdm pytorch-lightning kornia imageio omegaconf taming-transformers torch_optimizer
pip install vqgan-clip-generator
```

## Download model

The VQGAN algorithm requires use of a compatible model file. These files are not provided with the pip intallation, and must be downloaded separately. You can either download them manually, or use the provided download method.

```sh
mkdir models

curl -L -o models/vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 16384
curl -L -o models/vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 16384
```

Or, using the provided download method

```python
vqgan_clip.download(".\models\")
``` 

Note that when using this package you must specify the location where you've saved these model files
```python
import vqgan_clip.generate
from vqgan_clip.engine import VQGAN_CLIP_Config

config = VQGAN_CLIP_Config()
config.vqgan_config = f'models/vqgan_imagenet_f16_16384.yaml' # path to model yaml file
config.vqgan_checkpoint = f'models/vqgan_imagenet_f16_16384.ckpt' # path to model checkpoint file
config.text_prompts = 'A pastoral landscape painting by Rembrandt:1.0|A red cow:0.1'
vqgan_clip.generate.single_image(config)
```

### Using an AMD graphics card

Instructions courtesy of NerdyRodent. Note: This hasn't been tested yet.

ROCm can be used for AMD graphics cards instead of CUDA. You can check if your card is supported here:
<https://github.com/RadeonOpenCompute/ROCm#supported-gpus>

Install ROCm accordng to the instructions and don't forget to add the user to the video group:
<https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html>

The usage and set up instructions above are the same, except for the line where you install Pytorch.
Instead of `pip install torch==1.9.0+cu111 ...`, use the one or two lines which are displayed here (select Pip -> Python-> ROCm):
<https://pytorch.org/get-started/locally/>

### Using the CPU

If no graphics card can be found, the CPU is automatically used and a warning displayed.

Regardless of an available graphics card, the CPU can also be used by adding this command line argument: `-cd cpu`

This works with the CUDA version of Pytorch, even without CUDA drivers installed, but doesn't seem to work with ROCm as of now.

### Uninstalling

Remove the Python enviroment:

```sh
conda remove --name vqgan --all
```

## Run

To generate images from text, specify your text prompt as shown in the example below:
```sh
# TODO 
```

## Citations

```bibtex
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}
```

```bibtex
@misc{esser2020taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2020},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Katherine Crowson - <https://github.com/crowsonkb>
NerdyRodent - <https://github.com/nerdyrodent/>

Public Domain images from Open Access Images at the Art Institute of Chicago - <https://www.artic.edu/open-access/open-access-images>