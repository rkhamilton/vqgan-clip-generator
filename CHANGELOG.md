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