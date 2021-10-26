# v2.0.0
**API changes**
* generate.zoom_video_frames and generate.video_frames have been combined to a single function: generate.video_frames. If you do not specify zoom_scale, shift_x, or shift_y, these values default to 0, and non-zooming images are generated.
* generate.video_frames arguments changed. iterations and save_every are removed. New arguments are provided to make it easier to calculate video durations.
  * num_video_frames : Set the number of video frames (images) to be generated.
  * iterations_per_frame : Set the number of vqgan training iterations to perform for each frame of video. Higher numbers are more stylized.
* 

**New Features**
* generate.zoom_video lets you specify specific video frames where prompts should be changed using the argument change_prompts_on_frame. E.g. to change prompts on frames 150 and 200, use change_prompts_on_frame = [150,200]. Examples are updated with this argument.
* video_tools now sets ffmpeg to output on error only

**Bug Fixes**
* upscaling video example file had a bug in the ffmpeg command. Fixed.

# v1.3.1
**New features:**
* Single_image generation creates an image that matches the aspect ratio of any init_image provided. The output will have the same number of pixels as your specified output_size, in order to stay within your memory constraints. 


# v1.3.0
This release adds smoothing to the output of video_frames and restyle_video_frames. The smoothing is done by combining a user-specifiable number of latent vectors (z) and averaging them together using a modified exponentially weighted moving average (EWMA). The approach used here creates a sliding window of z frames (of z_smoothing_buffer length). The center of this window is considered the key frame, and has the greatest weight in the result. As frames move away from the center of the buffer, they have exponentially decreasing weight, by factor (1-z_smoothing_alpha)**offset_from_center.

To increase the temporal smoothing, increase the buffer size. To increase the weight of the key frame of video, increase the z_smoothing_alpha. More smoothing will combine adjacent z vectors, which will blur rapid motion from frame to frame.

# v1.2.2
Test coverage increased to include all generate, esrgan, and video_tools functions.

**Bug Fixes**
* generate.extract_video_frames was still saving jpgs. Changed to only save png.

# v1.2.1
**New features:**
* Video metadata is encoded by the encode_video function in the title (text prompts) and comment (generator parameters) fields. 

**Bug Fixes**
* generate.restyle_video* functions no longer re-load the VQGAN network each frame, which results in a 300% speed-up in running this function. This means that training doesn't start over each frame, so the output will look somewhat different than in earlier versions.
* generate functions no longer throw a warning when the output file argument doesn't have an extension.
* v1.2.0 introduced a bug where images were saved to output/output/filename. This is fixed.

# v1.2.0
**Important change to handling initial images**
I discovered that the code that I started from had a major deviation in how it handled initial images, which I carried over in my code. The expected behavior is that passing any value for init_weight would drive the algorithm to preserve the original image in the output. The code I was using had changed this behavior completely to an (interesting) experimental approach so that the initial image feature was putting pressure on the output to drive it to an all grayscale, flat image, with a decay of this effect with iteration. If you set the init_weight very high, instead of ending up with your initial image, you would get a flat gray image.

The line of code used in all other VQGAN+CLIP repos returns the diffrence between the outut tensor z (the current output image) and the orginal output tensor (original image):
```python
F.mse_loss(self._z, self._z_orig) * self.conf.init_weight / 2
```

The line of code used in the upstream copy that I started from is very different, with an effect that decreases with more iterations:
```python
F.mse_loss(self._z, torch.zeros_like(self._z_orig)) * ((1/torch.tensor(iteration_number*2 + 1))*self.conf.init_weight) / 2
```

**New features:**
* Alternate methods for maintaining init_image are provided.
  * 'decay' is the method used in this package from v1.0.0 through v1.1.3, and remains the default. This gives a more stylized look. Try values of 0.1-0.3.
  * 'original' is the method from the original Katherine Crowson colab notebook, and is in common use in other notebooks. This gives a look that stays closer to the source image. Try values of 1-2.
  * specify the method using config.init_weight_method = 'original' if desired, or config.init_weight_method = 'decay' to specify the default.
* Story prompts no longer cycle back to the first prompt when the end is reached.
* encode_video syntax change. input_framerate is now required. As before, if output_framerate differs from input_framerate, interpolation will be used.
* PNG outputs have data chunks added which describe the generation conditions. You can view these properties using imagemagick. "magick identify -verbose my_image.png"


# v1.1.3
**Bug Fixes**
* generate.restyle_video* functions now no longer renames the source files. Original filenames are preserved. As part of this fix, the video_tools.extract_video_frames() now uses a different naming format, consistent with generate.restyle_video. All video tools now use the filename frames_%12d.png.


# v1.1.2
**Bug Fixes**
* When generating videos, the pytorch random number generator was getting a new seed every frame of video, instead of keeping the same seed. This is now fixed, and video is more consistent from frame to frame.

# v1.1.1
By user request, it is now possible to set an Engine.conf.model_dir to store downloaded models in a subfolder of the current working directory.
```python
esrgan.inference_realesrgan(input='.\\video_frames',
        output_images_path='upscaled_video_frames',
        face_enhance=False,
        model_dir='models')

config = VQGAN_CLIP_Config()
config.model_dir = 'models'
generate.single_image(eng_config = config,
        image_prompts = 'input_image.jpg',
        iterations = 500,
        save_every = 10,
        output_filename = output_filename)
```


# v1.1.0
**This is a significant change that breaks compatibility.**

**New features:**
* [Real-ESRGAN integration](https://github.com/rkhamilton/vqgan-clip-generator/tree/main/main/Real-ESRGAN.md) for upscaling images and video. This can be used on generated media or existing media.
* In order to accomodate flexible upscaling, all generate.\*_video() methods have been changed to only generate folders of images (and renamed generate.*_video_frames()). You will need to optionally include a call to the upscaler, followed by a call to the video encoder.
* All [examples](https://github.com/rkhamilton/vqgan-clip-generator/tree/main/examples) have been updated to include upscaling.
* Model files for VQGAN and Real-ESRGAN are dynamically downloaded and cached in your pytorch hub folder instead of your working folder ./models subfolder. You will provide a URL and filename for the model to the vqgan_clip_generator.Engine object, and if there is no local copy available it will be downloaded and used. If a local copy has already been downloaded, it will not be downloaded again. This should give you a cleaner project folder / working directory, and allow model reuse across multiple project folders. 
    * These files will need to be manually removed when you uninstall vqgan_clip_generator. On Windows, model files are stored in *~\\.cache\torch\hub\models*
    * You can copy your existing downloaded model files to *~\\.cache\torch\hub\models* and they will be used and not re-downloaded.
* Initial images (init_image) used to initialize VQGAN+CLIP image generation has been moved to be an argument of the generate.* methods, instead of being accessed as part of Engine configuration. It was confusing that initializing image generation required accessing Engine config. The philosophy is that Engine config should not need to be touched in most cases except to set your output image size. Internally, generate.* methods just copy the init_image to the Engine config structure, but it seemed more clear to expose this as a generate.* argument.

**Known issues:**
* Story prompts aren't working when restyling videos. Only the initial prompts (before the ^) are used. I need to change the prompt cycling to be based on video frame, not iteration, since the iterations reset for each frame.
* Unit tests don't cover Real-ESRGAN yet.
* The Colab notebook isn't fully tested for these changes yet.