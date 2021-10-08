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



class EngineConfig:
    def __init__(self):
        self.prompts = 'A painting of flowers in the renaissance style:0.5|rembrandt:0.5^fish:0.2|love:1'
        self.image_prompts = []
        self.iterations = 100 # number of iterations of train() to perform before stopping.
        self.save_every = 50 # an interim image will be saved to the output location at an iteration interval defined here
        self.output_image_size = [256,256] # x/y dimensions of the output image in pixels. This will be adjusted slightly based on the GAN model used.
        self.init_image = None # a seed image that can be used to start the training. Without an initial image, random noise will be used.
        self.init_noise = None
        self.init_weight = 0.0
        self.clip_model = 'ViT-B/32'
        self.vqgan_config = f'models/vqgan_imagenet_f16_16384.yaml'
        self.vqgan_checkpoint = f'models/vqgan_imagenet_f16_16384.ckpt'
        self.noise_prompt_seeds = []
        self.noise_prompt_weights = []
        self.learning_rate = 0.1
        self.cut_method = 'latest'
        self.num_cuts = 32
        self.cut_power = 1.0
        self.seed = None
        self.optimiser = 'Adam'
        self.output_filename = 'output.png'
        self.make_video = False
        self.make_zoom_video = False
        self.zoom_start = 0
        self.zoom_save_every = 50
        self.zoom_scale = 1.02
        self.zoom_shift_x = 0
        self.zoom_shift_y = 0
        self.change_prompt_every = 0
        self.output_video_fps = 60
        self.input_video_fps = 15
        self.cudnn_determinism = False
        self.augments = [['Af', 'Pe', 'Ji', 'Er']]
        self.video_style_dir = None
        self.cuda_device = 'cuda:0'

class Engine:
    def __init__(self):
        # self._optimiser = optim.Adam([self._z], lr=0.1)
        self.conf = EngineConfig()
        self._iteration_number = 0.0
        self._loss = 0.0


        self._gumbel = False

        self.replace_grad = vm.ReplaceGrad.apply
        self.clamp_with_grad = vm.ClampWithGrad.apply

        self.seed = torch.seed()

        self.pMs = []

        self.image_prompts = []
        self.all_prompts_story_phrases = []
        self.current_prompt_story_phrase = []

        # default_image_size = 512  # >8GB VRAM
        # if not torch.cuda.is_available():
        #     default_image_size = 256  # no GPU found
        # elif get_device_properties(0).total_memory <= 2 ** 33:  # 2 ** 33 = 8,589,934,592 bytes = 8 GB
        #     default_image_size = 318  # <8GB VRAM

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
        
        if i % self.conf.save_every == 0:
            with torch.inference_mode():
                # TODO move this to outer loop
                # losses_str = ', '.join(f'{loss.item():g}' for loss in lossAll)
                # tqdm.write(f'i: {i}, loss: {sum(lossAll).item():g}, lossAll: {losses_str}')
                out = self.synth()
                info = PngImagePlugin.PngInfo()
                info.add_text('comment', self.current_prompt_story_phrase[0])
                TF.to_pil_image(out[0].cpu()).save(self.conf.output_filename, pnginfo=info) 	

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

        if self.conf.init_weight:
            # result.append(F.mse_loss(self._z, z_orig) * args.init_weight / 2)
            result.append(F.mse_loss(self._z, torch.zeros_like(self._z_orig)) * ((1/torch.tensor(self._iteration_number*2 + 1))*self.conf.init_weight) / 2)

        for prompt in self.pMs:
            result.append(prompt(iii))
        
        if self.conf.make_video:    
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
        self.parse_text_prompts()
            
        # Split target images using the pipe character (weights are split later)
        if self.conf.image_prompts:
            self.image_prompts = self.conf.image_prompts.split("|")
            self.image_prompts = [image.strip() for image in self.image_prompts]



        
        self._device = torch.device(self.conf.cuda_device)
        self.load_model()
        jit = True if float(torch.__version__[:3]) < 1.8 else False
        self._perceptor = clip.load(self.conf.clip_model, jit=jit)[0].eval().requires_grad_(False).to(self._device)

        f = 2**(self._model.decoder.num_resolutions - 1)

        # Cutout class options:
        # 'latest','original','updated' or 'updatedpooling'
        if self.conf.cut_method == 'latest':
            self._make_cutouts = vm.MakeCutouts(self._perceptor.visual.input_resolution, self.conf.num_cuts, self.conf.augments, cut_pow=self.conf.cut_power)
        elif self.conf.cut_method == 'original':
            self._make_cutouts = vm.MakeCutoutsOrig(self._perceptor.visual.input_resolution, self.conf.num_cuts, cut_pow=self.conf.cut_power)
        elif self.conf.cut_method == 'updated':
            self._make_cutouts = vm.MakeCutoutsUpdate(self._perceptor.visual.input_resolution, self.conf.num_cuts, cut_pow=self.conf.cut_power)
        elif self.conf.cut_method == 'nrupdated':
            self._make_cutouts = vm.MakeCutoutsNRUpdate(self._perceptor.visual.input_resolution, self.conf.num_cuts, self.conf.augments, cut_pow=self.conf.cut_power)
        else:
            self._make_cutouts = vm.MakeCutoutsPoolingUpdate(self._perceptor.visual.input_resolution, self.conf.num_cuts, cut_pow=self.conf.cut_power)    

        toksX = self.conf.output_image_size[0] // f
        toksY = self.conf.output_image_size[1] // f
        self.output_image_size_X = (self.conf.output_image_size[0] // f) * f
        self.output_image_size_Y = (self.conf.output_image_size[1] // f) * f

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

        if self.conf.init_image:
            if 'http' in self.conf.init_image:
                self.convert_image_to_init_image(Image.open(urlopen(self.conf.init_image)))
            else:
                self.convert_image_to_init_image(Image.open(self.conf.init_image))
        elif self.conf.init_noise == 'pixels':
            self.convert_image_to_init_image(vm.make_random_noise_image(self.conf.image_size[0], self.conf.image_size[1]))
        elif self.conf.init_noise == 'gradient':
            self.convert_image_to_init_image(vm.make_random_gradient_image(self.conf.image_size[0], self.conf.image_size[1]))
        else:
            # this is the default that happens if no initialization image options are specified
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
        if self.current_prompt_story_phrase:
            for prompt in self.current_prompt_story_phrase:
                self.append_text_prompt(prompt)

        for prompt in self.image_prompts:
            self.append_image_prompt(prompt)

        for seed, weight in zip(self.conf.noise_prompt_seeds, self.conf.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self._perceptor.visual.output_dim]).normal_(generator=gen)
            self.pMs.append(vm.Prompt(embed, weight).to(self._device))

        iteration_num = 0 # Iteration counter
        zoom_video_frame_num = 0 # Zoom video frame counter
        phrase_counter = 1 # Phrase counter
        smoother_counter = 0 # Smoother counter
        video_styler_frame_num = 0 # for video styling

        self.set_optimiser(self.conf.optimiser)
        try:
            for iteration_num in tqdm(range(1,self.conf.iterations+1)):
                self.train(iteration_num)
        except KeyboardInterrupt:
            pass

    def convert_image_to_init_image(self, output):
        pil_image = output.convert('RGB')
        pil_image = pil_image.resize((self.output_image_size_X, self.output_image_size_Y), Image.LANCZOS)
        pil_tensor = TF.to_tensor(pil_image)
        self._z, *_ = self._model.encode(pil_tensor.to(self._device).unsqueeze(0) * 2 - 1)

    def append_image_prompt(self, prompt):
        # given an image prompt that is a filename followed by a weight e.g. 'prompt_image.png:0.5', load the image, encode it with CLIP, and append it to the list of prompts used for image generation
        path, weight, stop = self.split_prompt(prompt)
        output_image = Image.open(path)
        pil_image = output_image.convert('RGB')
        output_image = vm.resize_image(pil_image, (self.output_image_size_X, self.output_image_size_Y))
        batch = self._make_cutouts(TF.to_tensor(output_image).unsqueeze(0).to(self._device))
        embed = self._perceptor.encode_image(vm.normalize(batch)).float()
        self.pMs.append(vm.Prompt(embed, weight, stop).to(self._device))

    def append_text_prompt(self, prompt):
        # given a text prompt like 'a field of red flowers:0.5' parse that into text and weights, encode it with CLIP, and add it to the encoded prompts used for image generation
        txt, weight, stop = self.split_prompt(prompt)
        embed = self._perceptor.encode_text(clip.tokenize(txt).to(self._device)).float()
        self.pMs.append(vm.Prompt(embed, weight, stop).to(self._device))

    def load_model(self):
        # This step is slow, and does not need to be done each time an image is generated.
        self._model = vm.load_vqgan_model(self.conf.vqgan_config, self.conf.vqgan_checkpoint).to(self._device)

    def parse_text_prompts(self):
        # text prompts are provided to the class as a series of phrases and clauses separated by | and ^
        # This method separates this single string into lists of separate phrases to be processed by CLIP

        # Split text prompts using the pipe character (weights are split later)
        if self.conf.prompts:
            # For stories, there will be many phrases separated by ^ 
            # e.g. "a field:0.2^a pile of leaves|painting|red" would parse into two phrases 'a field:0.2' and 'a pile of leaves|painting|red'
            story_phrases = [phrase.strip() for phrase in self.conf.prompts.split("^")]
            
            # Make a list of all phrases.
            all_prompts_phrases = []
            for phrase in story_phrases:
                all_prompts_phrases.append(phrase.split("|"))
            self.all_prompts_story_phrases = all_prompts_phrases

            # First phrase
            self.current_prompt_story_phrase = self.all_prompts_story_phrases[0]
        else:
            self.all_prompts_story_phrases = []
            self.current_prompt_story_phrase = []

    @staticmethod
    def split_prompt(prompt):
        #NR: Split prompts and weights
        vals = prompt.rsplit(':', 2)
        vals = vals + ['', '1', '-inf'][len(vals):]
        return vals[0], float(vals[1]), float(vals[2])