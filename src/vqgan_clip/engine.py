# This contains the original math to generate an image from VQGAN+CLIP. I don't fully understand what it's doing and don't expect to change it.
import vqgan_clip.vqgan_math as vm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from torch.cuda import get_device_properties
torch.backends.cudnn.benchmark = False		# NR: True is a bit faster, but can lead to OOM. False is more deterministic.
#torch.use_deterministic_algorithms(True)	# NR: grid_sampler_2d_backward_cuda does not have a deterministic implementation
from torch_optimizer import DiffGrad, AdamP, RAdam
import clip
from PIL import ImageFile, Image, PngImagePlugin, ImageChops
ImageFile.LOAD_TRUNCATED_IMAGES = True
import imageio
from tqdm import tqdm

from urllib.request import urlopen

import numpy as np


class Engine:
    config = {
        'prompt' : 'A painting of flowers in the renaissance style', # text prompt used to generate an image
        'image_prompts' : [], # an input image can be parsed by CLIP and used as a prompt to generate another image
        'iterations' : 100, # number of iterations of train() to perform before stopping.
        'save_every' : 50, # an interim image will be saved to the output location at an iteration interval defined here
        'output_image_size' : [256,256], # x/y dimensions of the output image in pixels.
        'init_image' : None, # a seed image that can be used to start the training. Without an initial image, random noise will be used.
        'init_noise' : None,
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
        'augments' : [['Af', 'Pe', 'Ji', 'Er']],
        'video_style_dir' : None,
        'cuda_device' : 'cuda:0'
    }

    def __init__(self):
        # self._optimiser = optim.Adam([self._z], lr=0.1)
        self._iteration_number = 0.0
        self._loss = 0.0


        self._gumbel = False

        self.replace_grad = vm.ReplaceGrad.apply
        self.clamp_with_grad = vm.ClampWithGrad.apply

        self.seed = torch.seed()

        self.pMs = []


    def set_seed(self, seed):
        self.seed = seed
        torch.manual_seed(seed)

    # Set the optimiser
    def set_optimiser(self, opt_name, opt_lr=0.1):
        if opt_name == "Adam":
            self._optimiser = optim.Adam([self._z], lr=opt_lr)	# LR=0.1 (Default)
        elif opt_name == "AdamW":
            self._optimiser = optim.AdamW([self._z], lr=opt_lr)	
        elif opt_name == "Adagrad":
            self._optimiser = optim.Adagrad([self._z], lr=opt_lr)	
        elif opt_name == "Adamax":
            self._optimiser = optim.Adamax([self._z], lr=opt_lr)	
        elif opt_name == "DiffGrad":
            self._optimiser = DiffGrad([self._z], lr=opt_lr, eps=1e-9, weight_decay=1e-9) # NR: Playing for reasons
        elif opt_name == "AdamP":
            self._optimiser = AdamP([self._z], lr=opt_lr)		    
        elif opt_name == "RAdam":
            self._optimiser = RAdam([self._z], lr=opt_lr)		    
        elif opt_name == "RMSprop":
            self._optimiser = optim.RMSprop([self._z], lr=opt_lr)
        else:
            print("Unknown optimiser.")
            self._optimiser = optim.Adam([self._z], lr=opt_lr)

    def train(self, i):
        #self._optimiser.zero_grad(set_to_none=True)
        lossAll = self.ascend_txt()
        
        if i % self.config['save_every'] == 0:
            with torch.inference_mode():
                # TODO move this to outer loop
                # losses_str = ', '.join(f'{loss.item():g}' for loss in lossAll)
                # tqdm.write(f'i: {i}, loss: {sum(lossAll).item():g}, lossAll: {losses_str}')
                out = self.synth()
                info = PngImagePlugin.PngInfo()
                info.add_text('comment', self.config['prompt'][0])
                TF.to_pil_image(out[0].cpu()).save(self.config['output_filename'], pnginfo=info) 	

        loss = sum(lossAll)
        loss.backward()
        self._optimiser.step()
        
        #with torch.no_grad():
        with torch.inference_mode():
            self._z.copy_(self._z.maximum(self.z_min).minimum(self.z_max))

    def ascend_txt(self):
        out = self.synth()
        iii = self._perceptor.encode_image(vm.normalize(self._make_cutouts(out))).float()
        
        result = []

        if self.config['init_weight']:
            # result.append(F.mse_loss(self._z, z_orig) * args.init_weight / 2)
            result.append(F.mse_loss(self._z, torch.zeros_like(self._z_orig)) * ((1/torch.tensor(self._iteration_number*2 + 1))*self.config['init_weight']) / 2)

        for prompt in self.pMs:
            result.append(prompt(iii))
        
        if self.config['make_video']:    
            img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
            img = np.transpose(img, (1, 2, 0))
            imageio.imwrite('./steps/' + str(self._iteration_number) + '.png', np.array(img))

        return result # return loss

    # Vector quantize
    def synth(self):
        if self._gumbel:
            z_q = vm.vector_quantize(self._z.movedim(1, 3), self._model.quantize.embed.weight).movedim(3, 1)
        else:
            z_q = vm.vector_quantize(self._z.movedim(1, 3), self._model.quantize.embedding.weight).movedim(3, 1)
        clamp_with_grad = vm.ClampWithGrad.apply
        return clamp_with_grad(self._model.decode(z_q).add(1).div(2), 0, 1)

    # main execution path from generate.py
    def do_it(self):
        # Split text prompts using the pipe character (weights are split later)
        # TODO stop overwriting config prompt
        if self.config['prompt']:
            # For stories, there will be many phrases
            story_phrases = [phrase.strip() for phrase in self.config['prompt'].split("^")]
            
            # Make a list of all phrases
            all_phrases = []
            for phrase in story_phrases:
                all_phrases.append(phrase.split("|"))
            
            # First phrase
            self.config['prompt'] = all_phrases[0]
            
        # Split target images using the pipe character (weights are split later)
        if self.config['image_prompts']:
            self.config['image_prompts'] = self.config['image_prompts'].split("|")
            self.config['image_prompts'] = [image.strip() for image in self.config['image_prompts']]
            # Check for GPU and reduce the default image size if low VRAM

        default_image_size = 512  # >8GB VRAM
        if not torch.cuda.is_available():
            default_image_size = 256  # no GPU found
        elif get_device_properties(0).total_memory <= 2 ** 33:  # 2 ** 33 = 8,589,934,592 bytes = 8 GB
            default_image_size = 318  # <8GB VRAM

        
        self._device = torch.device(self.config['cuda_device'])
        self._model = vm.load_vqgan_model(self.config['vqgan_config'], self.config['vqgan_checkpoint']).to(self._device)
        jit = True if float(torch.__version__[:3]) < 1.8 else False
        self._perceptor = clip.load(self.config['clip_model'], jit=jit)[0].eval().requires_grad_(False).to(self._device)

        cut_size = self._perceptor.visual.input_resolution
        f = 2**(self._model.decoder.num_resolutions - 1)

        # Cutout class options:
        # 'latest','original','updated' or 'updatedpooling'
        if self.config['cut_method'] == 'latest':
            self._make_cutouts = vm.MakeCutouts(cut_size, self.config['num_cuts'], self.config['augments'], cut_pow=self.config['cut_power'])
        elif self.config['cut_method'] == 'original':
            self._make_cutouts = vm.MakeCutoutsOrig(cut_size, self.config['num_cuts'], cut_pow=self.config['cut_power'])
        elif self.config['cut_method'] == 'updated':
            self._make_cutouts = vm.MakeCutoutsUpdate(cut_size, self.config['num_cuts'], cut_pow=self.config['cut_power'])
        elif self.config['cut_method'] == 'nrupdated':
            self._make_cutouts = vm.MakeCutoutsNRUpdate(cut_size, self.config['num_cuts'], self.config['augments'], cut_pow=self.config['cut_power'])
        else:
            self._make_cutouts = vm.MakeCutoutsPoolingUpdate(cut_size, self.config['num_cuts'], cut_pow=self.config['cut_power'])    

        toksX, toksY = self.config['output_image_size'][0] // f, self.config['output_image_size'][1] // f
        sideX, sideY = toksX * f, toksY * f

        # CLIP tokenize/encode   
        if self.config['prompt']:
            for prompt in self.config['prompt']:
                txt, weight, stop = vm.split_prompt(prompt)
                embed = self._perceptor.encode_text(clip.tokenize(txt).to(self._device)).float()
                self.pMs.append(vm.Prompt(embed, weight, stop).to(self._device))

        # Gumbel or not?
        if self._gumbel:
            e_dim = 256
            n_toks = self._model.quantize.n_embed
            self.z_min = self._model.quantize.embed.weight.min(dim=0).values[None, :, None, None]
            self.z_max = self._model.quantize.embed.weight.max(dim=0).values[None, :, None, None]
        else:
            e_dim = self._model.quantize.e_dim
            n_toks = self._model.quantize.n_e
            self.z_min = self._model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
            self.z_max = self._model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]


        if self.config['init_image']:
            if 'http' in self.config['init_image']:
                output = Image.open(urlopen(self.config['init_image']))
            else:
                output = Image.open(self.config['init_image'])
            pil_image = output.convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            pil_tensor = TF.to_tensor(pil_image)
            self._z, *_ = self._model.encode(pil_tensor.to(self._device).unsqueeze(0) * 2 - 1)
        elif self.config['init_noise'] == 'pixels':
            output = vm.random_noise_image(self.config['image_size'][0], self.config['image_size'][1])    
            pil_image = output.convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            pil_tensor = TF.to_tensor(pil_image)
            self._z, *_ = self._model.encode(pil_tensor.to(self._device).unsqueeze(0) * 2 - 1)
        elif self.config['init_noise'] == 'gradient':
            output = vm.random_gradient_image(self.config['image_size'][0], self.config['image_size'][1])
            pil_image = output.convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            pil_tensor = TF.to_tensor(pil_image)
            self._z, *_ = self._model.encode(pil_tensor.to(self._device).unsqueeze(0) * 2 - 1)
        else:
            one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=self._device), n_toks).float()
            # self._z = one_hot @ self._model.quantize.embedding.weight
            if self._gumbel:
                self._z = one_hot @ self._model.quantize.embed.weight
            else:
                self._z = one_hot @ self._model.quantize.embedding.weight

            self._z = self._z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2) 
            #self._z = torch.rand_like(self._z)*2						# NR: check

        self._z_orig = self._z.clone()
        self._z.requires_grad_(True)       
        
        # CLIP tokenize/encode   
        if self.config['prompt']:
            for prompt in self.config['prompt']:
                txt, weight, stop = vm.split_prompt(prompt)
                embed = self._perceptor.encode_text(clip.tokenize(txt).to(self._device)).float()
                self.pMs.append(vm.Prompt(embed, weight, stop).to(self._device))

        for prompt in self.config['image_prompts']:
            path, weight, stop = vm.split_prompt(prompt)
            output_image = Image.open(path)
            pil_image = output_image.convert('RGB')
            output_image = vm.resize_image(pil_image, (sideX, sideY))
            batch = self._make_cutouts(TF.to_tensor(output_image).unsqueeze(0).to(self._device))
            embed = self._perceptor.encode_image(vm.normalize(batch)).float()
            self.pMs.append(vm.Prompt(embed, weight, stop).to(self._device))

        for seed, weight in zip(self.config['noise_prompt_seeds'], self.config['noise_prompt_weights']):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self._perceptor.visual.output_dim]).normal_(generator=gen)
            self.pMs.append(vm.Prompt(embed, weight).to(self._device))

        iteration_num = 0 # Iteration counter
        zoom_video_frame_num = 0 # Zoom video frame counter
        phrase_counter = 1 # Phrase counter
        smoother_counter = 0 # Smoother counter
        video_styler_frame_num = 0 # for video styling

        self.set_optimiser(self.config['optimiser'])
        try:
            for iteration_num in tqdm(range(1,self.config['iterations']+1)):
                self.train(iteration_num)
        except KeyboardInterrupt:
            pass