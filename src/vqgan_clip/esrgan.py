# adapted from https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py
# modified to convert to a callable function
# This requires Real-ESRGAN to be installed.
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
import re
from tqdm import tqdm

from realesrgan import RealESRGANer
from vqgan_clip.download import load_file_from_url

def inference_realesrgan(input='./video_frames',
                         model_filename='RealESRGAN_x4plus.pth',
                         model_url=f'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                         output_images_path='./upscaled_video_frames',
                         purge_existing_files=False,
                         netscale=4,
                         outscale=4,
                         tile=0,
                         tile_pad=10,
                         pre_pad=0,
                         face_enhance=False,
                         half=True,
                         block=23,
                         ext='auto',
                         model_dir=None):
    """Applies a machine learning image restoration model using Real-ESRGAN. The default is a general purpose upscaler, but many 
    models are available that are trained for different types of content. See https://upscale.wiki/wiki/Model_Database.

    Args:
        * input (str, optional): Input image or folder. Defaults to './video_frames'.
        * model_filename (str, optional): Filename for the pre-trained model. Defaults to 'RealESRGAN_x4plus.pth'.
        * model_url (str, optional): URL to download the model file. If the model is not cached locally it will be downloaded and cached. Defaults to 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'.
        * output_images_path (str, optional): Output folder. Defaults to './upscaled_video_frames'.
        * purge_existing_files (boolean): If True, all of the files in output_images_path will be deleted before new files are created.
        * netscale (int, optional): Upsample scale factor of the network. Defaults to 4.
        * outscale (int, optional): The final upsampling scale of the image. Defaults to 4.
        * tile (int, optional): Tile size, 0 for no tile during testing. Defaults to 0.
        * tile_pad (int, optional): Tile padding. Defaults to 10.
        * pre_pad (int, optional): Pre padding size at each border. Defaults to 0.
        * face_enhance (bool, optional): Use GFPGAN to enhance face, while also upsampling. Defaults to False.
        * half (bool, optional): Use half precision during inference. Defaults to True.
        * block (int, optional): num_block in RRDB. Defaults to 23.
        * alpha_upsampler (str, optional): The upsampler for the alpha channels. Options: realesrgan | bicubic. Defaults to 'realesrgan'
        * ext (str, optional): Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Defaults to 'auto'
        * model_dir (str, optional): If set to a folder name (e.g. 'models') then model files will be downloaded to a subfolder of the current working directory. Defaults to None.
    """
    # load the file from disk if available, otherwise download it.
    model_filename = load_file_from_url(model_url, model_dir=model_dir, progress=True, file_name=model_filename)

    if 'RealESRGAN_x4plus_anime_6B.pth' in model_filename:
        block = 6
    elif 'RealESRGAN_x2plus.pth' in model_filename:
        netscale = 2

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=block, num_grow_ch=32, scale=netscale)

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_filename,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half)

    if face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth',
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)

    # purge previously extracted original frames
    if not os.path.exists(output_images_path):
        os.mkdir(output_images_path)
    else:
        if purge_existing_files:
            for f in glob.glob(output_images_path+os.sep+'*'):
                os.remove(f)

    if os.path.isfile(input):
        paths = [input]
    else:
        # do a natural sort on the image filenames to handle files named 1.png, 2.png ... 10.png, 11.png.
        paths = sorted(glob.glob(os.path.join(input, '*')),
                       key=lambda x: int(re.sub('\D', '', x)))

    for path in tqdm(paths, unit='image', desc='Real-ESRGAN'):
        imgname, extension = os.path.splitext(os.path.basename(path))

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        h, w = img.shape[0:2]
        if max(h, w) > 1000 and netscale == 4:
            import warnings
            warnings.warn(
                'The input image is large, try X2 model for better performance.')
        if max(h, w) < 500 and netscale == 2:
            import warnings
            warnings.warn(
                'The input image is small, try X4 model for better performance.')

        try:
            if face_enhance:
                _, _, model_output = face_enhancer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                model_output, _ = upsampler.enhance(img, outscale=outscale)
        except Exception as error:
            print('Error', error)
            print(
                'If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            if ext == 'auto':
                extension = extension[1:]
            else:
                extension = ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            if os.path.exists(os.path.join(output_images_path, f'{imgname}.{extension}')):
                save_path = os.path.join(
                    output_images_path, f'{imgname}_upscaled.{extension}')
            else:
                save_path = os.path.join(
                    output_images_path, f'{imgname}.{extension}')

            cv2.imwrite(save_path, model_output)
