# For Python Beginners
This is a very rough outline of how to get started if you are a Windows user and have never used Python before, don't really care about Python, and just want to get to a basic place of creating images. I won't be able to offer one-on-one help with this, but hopefully this guide is better than nothing.

## One-time setup
1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) for your platform.
2. Open your miniconda prompt (e.g. from the start menu on Windows) and paste in the following text. This will install all of the python packages you need to be able to run this package and create images. You will not see anything change on your computer. No files or folders will appear in your current working directory. They are all being installed elsewhere so that you can run them from any folder on your PC.
```sh
conda create --name vqgan python=3.9 pip ffmpeg numpy pytest tqdm git pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c conda-forge ffmpeg
conda activate vqgan
pip install git+https://github.com/openai/CLIP.git taming-transformers ftfy regex tqdm pytorch-lightning kornia imageio omegaconf torch_optimizer
pip install git+https://github.com/rkhamilton/vqgan-clip-generator.git
pip install opencv-python scipy
pip install basicsr
pip install facexlib
pip install gfpgan
pip install git+https://github.com/xinntao/Real-ESRGAN
pip install sk-video
pip install opencv-python
pip install moviepy
git clone git@github.com:hzwer/arXiv2020-RIFE.git
```
3. Create a folder on your PC where you want to create your images. Into that folder, download one or more of the [example python files](https://github.com/rkhamilton/vqgan-clip-generator/tree/main/examples) provided in the repository. I will use the single_image.py file for the rest of this example.
## Creating images
1. Edit single_image.py in notepad (or better, VS Code) and change the line that says text_prompt to describe whatever you want the software to create for you.
```python
text_prompts = 'A pastoral landscape painting by Rembrandt'
```
2. Open the Anaconda command prompt and actiave your Python environment where you installed all of those tools. You will need to do this any time you want to run these tools.
```sh
conda activate vqgan
```
3. Run the editing python script from the Anaconda command prompt. You can run this command over and over with no other setup.
```sh
python single_image.py
```

That's it! You should see the generated image apear in a new subfolder called output. The basic workflow is to edit an example python file and run it as shown in step 3.

## Uninstalling
If you ever want to get rid of all of this stuff, run the following command and it will all be uninstalled. Also delete the contents of C:\\Users\\{your_windows_username}\\.cache\\torch\\hub\\models
```sh
conda deactivate
conda remove --name vqgan --all
```
# Troubleshooting
If you see an error message:
```sh
RuntimeError: CUDA out of memory.
```
Then your GPU doesn't have enough memory to create the image size you selected. Edit the output_image_size line to try to create a smaller file.
```python
config.output_image_size = [587,330]
```
