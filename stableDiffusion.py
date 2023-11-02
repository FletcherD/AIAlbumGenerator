import torch
import time
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

#model_id = "stabilityai/stable-diffusion-2-1"
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
refiner_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
n_steps = 100
high_noise_frac = 0.8

prompt = """Album artwork for the album "{title}" by {artist}, released in {year}, in the genres {genre}"""

def generateAlbumArtwork(release):
    base = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    #base.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    refiner = DiffusionPipeline.from_pretrained(
        refiner_id,
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
    )

    refiner.to("cuda")
    base.enable_model_cpu_offload()

    thisPrompt = prompt.format(**release)
    print(thisPrompt)

    #pipe.text2img(thisPrompt, width=768, height=768, max_embeddings_multiples=3).images[0]
    image = base(thisPrompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]

    imagePath = '{}.png'.format(int(time.time()))
    image.save(imagePath)
    return imagePath
