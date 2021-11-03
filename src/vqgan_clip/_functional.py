# Various functions and classes
import torch
from torch import nn, optim
import math
from PIL import ImageFile, Image, PngImagePlugin, ImageChops
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from torch.nn import functional as F
import kornia.augmentation as K
from torchvision.transforms import functional as TF
from torchvision import transforms
from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan
import glob, os
import subprocess
import contextlib
import piexif

def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


# For zoom video
def zoom_at(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2, 
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.LANCZOS)


# NR: Testing with different intital images
def make_random_noise_image(w,h):
    random_image = Image.fromarray(np.random.randint(0,255,(w,h,3),dtype=np.dtype('uint8')))
    return random_image


def gradient_2d(start, stop, width, height, is_horizontal):
    """create initial gradient image
    """
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=float)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = gradient_2d(start, stop, width, height, is_horizontal)

    return result

    
def make_random_gradient_image(w,h):
    array = gradient_3d(w, h, (0, 0, np.random.randint(0,255)), (np.random.randint(1,255), np.random.randint(2,255), np.random.randint(3,128)), (True, False, False))
    random_image = Image.fromarray(np.uint8(array))
    return random_image


# Used in older MakeCutouts
def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    replace_grad = ReplaceGrad.apply
    return replace_grad(x_q, x)

def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        replace_grad = ReplaceGrad.apply
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


# An updated version with Kornia augments, but no pooling:
class MakeCutoutsKornia(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            K.RandomSharpness(0.3,p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2,p=0.4),)
        self.noise_fac = 0.1


    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch

class MakeCutoutsSG3(torch.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


# This is the original version (No pooling)
class MakeCutoutsOrig(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        clamp_with_grad = ClampWithGrad.apply
        return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)

normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])

def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model

def delete_files(path_to_delete):
    """Delete all files in the folder passed as an argument

    Args:
        frame_path (str): Folder path
    """
    files = glob.glob(path_to_delete + os.sep + '*')
    for f in files:
        os.remove(f)

def parse_all_prompts(text_prompts, image_prompts, noise_prompts):
    """Split prompt strings into lists of lists of prompts.
    Apply parse_story_prompts() to each of conf.text_prompts, conf.image_prompts, and conf.noise_prompts
    """
    # 
    if text_prompts:
        text_prompts = parse_story_prompts(text_prompts)
    else:
        text_prompts = []
    
    # Split target images using the pipe character (weights are split later)
    if image_prompts:
        image_prompts = parse_story_prompts(image_prompts)
    else:
        image_prompts = []

    # Split noise prompts using the pipe character (weights are split later)
    if noise_prompts:
        noise_prompts = parse_story_prompts(noise_prompts)
    else: 
        noise_prompts = []
    
    return text_prompts, image_prompts, noise_prompts

def parse_story_prompts(prompt):
    """This method splits the input string by ^, then by |, and returns the full set of substrings as a list.
    Story prompts, in the form of images, text, noise, are provided to the class as a string containing a series of phrases and clauses separated by | and ^. The set of all such groups of text is the story.
    
    example parse_story_prompts("a field:0.2^a pile of leaves|painting|red")
    would return [['a field:0.2'],['a pile of leaves','painting','red']]

    Args:
        prompt (string): A string containing a series of phrases and separated by ^ and |

    Returns:
        all_prompts (list of lists): A list of lists of all substrings from the input prompt, first split by ^, then by |
    """
    # 

    all_prompts = []

    # Split text prompts using the pipe character (weights are split later)
    if prompt:
        # For stories, there will be many phrases separated by ^ 
        # e.g. "a field:0.2^a pile of leaves|painting|red" would parse into two phrases 'a field:0.2' and 'a pile of leaves|painting|red'
        story_phrases = [phrase.strip() for phrase in prompt.split("^")]
        
        # Make a list of all phrases.
        for phrase in story_phrases:
            all_prompts.append(phrase.split("|"))

    return all_prompts

def split_prompt(prompt):
    """Split an input string of the form 'string:float' into three returned objects: string, float, -inf

    Args:
        prompt (string): String of the form 'string:float' E.g. 'a red boat:1.2'

    Returns:
        text (str): The substring from prompt prior to the colon, e.g. 'A red boat'
        weight (float): The string after the colon is converted to a float, e.g 1.2
        stop (int): Returns -inf. I have never seen this value used, but it is provided in the original algorithm.
    """
    # handle the case where we get a full path with a drive letter
    drive, path = os.path.splitdrive(prompt)

    #NR: Split prompts and weights
    vals = path.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return drive+vals[0], float(vals[1]), float(vals[2])

def filesize_matching_aspect_ratio(file_name, desired_x, desired_y):
    """Calculate image sizes (X,Y) that have an area equal to the area of your desired_x, desired_y image, but with an aspect ratio matching the image in file_name.

    Args:
        file_name (str): Path to an image file. We want to match the aspect ratio of this file.
        desired_x (int): Desired image size X, without knowing a target aspect ratio.
        desired_y (int): Desired image size Y, without knowing a target aspect ratio.

    Returns:
        restyled_image_x (int) : Width of an image file matching the aspect ratio of the input image filename.
        restyled_image_y (int) : Height of an image file matching the aspect ratio of the input image filename.
    """
    # get the source video file dimensions
    files = glob.glob(file_name)
    img=Image.open(files[0])
    source_img_x,source_img_y=img.size
    img.close()
    source_aspect_ratio = source_img_x/source_img_y
    # calculate the dimensions of the new restyled video such that it has an area that you can process in GAN
    restyled_image_y = int(math.sqrt(desired_x*desired_y/source_aspect_ratio))
    restyled_image_x = int(source_aspect_ratio * restyled_image_y)
    return restyled_image_x, restyled_image_y

def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                func(*a, **ka)
    return wrapper

def png_info_chunks(list_of_info):
    """Create a series of PNG info chunks that contain various details of image generation.

    Args:
        * list_of_info (list): A list of data to encode in a PNG file. Structure is [['chunk_name']['chunk_data'],['chunk_name']['chunk_data'],...]

    Returns:
        [PngImagePlugin.PngInfo()]: A PngImagePlugin.PngInfo() object populated with various information about the image generation.
    """
    # model_output = self.synth()
    info = PngImagePlugin.PngInfo()
    for chunk_tuple in list_of_info:
        encode_me = chunk_tuple[1] if chunk_tuple[1] else ''  
        info.add_text(chunk_tuple[0], str(encode_me))
    return info

def info_to_jpg_exif(list_of_info):
    """Creates a PIL exif object from a list of info

    Args:
        * list_of_info (list): A list of data to encode in a PNG file. Structure is [['chunk_name']['chunk_data'],['chunk_name']['chunk_data'],...]
    """
    # concatenate everything in the info dict into one string
    concat_info = ''
    for item in list_of_info:
        encode_me = item[1] if item[1] else '[]'  
        concat_info = concat_info + f'{item[0]}: {str(encode_me)}, '

    xptitle = list_of_info[0][1] if list_of_info[0][1] else ''
    # assemble the structure for piexif to process
    zeroth_ifd = {
        piexif.ImageIFD.Software: "https://github.com/rkhamilton/vqgan-clip-generator",
        piexif.ImageIFD.XPTitle: xptitle.encode('utf_16_le'),
        piexif.ImageIFD.XPComment: concat_info[:-2].encode('utf_16_le'),  
    }
    exif_ifd = {
        piexif.ExifIFD.DateTimeOriginal: u"2099:09:29 10:10:10",
        piexif.ExifIFD.UserComment: concat_info[:-2].encode('utf_16_le')
    }
    exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd}
    exif_bytes = piexif.dump(exif_dict)
    
    return exif_bytes

def copy_image_metadata(files_with_metadata_path,files_needing_metadata_path):
    """Copy metadata between files. From one file to one file, or from one folder to another.

    Args:
        files_with_metadata_path ([type]): [description]
        files_needing_metadata_path ([type]): [description]
    """
    # one file to one file
    if os.path.isfile(files_with_metadata_path):
        ext = os.path.splitext(files_with_metadata_path)[1].lower()
        assert( ext in ['.png', '.jpg'] )
        source_files = [files_with_metadata_path]
        if not os.path.isfile(files_needing_metadata_path):
            raise NameError('If source is a file, destination must also be a file.')
        dest_files = [files_needing_metadata_path]
    # copy from a folder of images (jpg or png or both)
    elif os.path.isdir(files_with_metadata_path):
        source_files_png = glob.glob(files_with_metadata_path + os.sep + '*.png')
        source_files_jpg = glob.glob(files_with_metadata_path + os.sep + '*.jpg')
        source_files = source_files_png + source_files_jpg
        assert(os.path.isdir(files_needing_metadata_path))
        
        # use source filenames
        dest_files = []
        for src in source_files:
            dest_files.append(os.path.join(files_needing_metadata_path,os.path.basename(src)))
    else:
        raise NameError('Input must be paths to files or folders.')
    
    assert len(source_files) == len(dest_files)

    # creat tuples for matching source and dest paths
    file_set = zip(source_files, dest_files)

    for source_file, dest_file in file_set:
        assert(os.path.exists(source_file))
        assert(os.path.exists(dest_file))
        # get file extension
        ext = os.path.splitext(source_file)[1]
        if ext.lower() == '.png':
            info = PngImagePlugin.PngInfo()
            pil_source = Image.open(source_file)
            pil_dest = Image.open(dest_file)
            
            for key, value in zip(pil_source.info.keys(), pil_source.info.values()):
                info.add_text(key, str(value))
            pil_dest.save(dest_file, "png", pnginfo=info)
        elif ext.lower() == '.jpg':
            piexif.transplant(source_file,dest_file)
