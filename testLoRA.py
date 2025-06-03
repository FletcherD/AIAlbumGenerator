import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_base = "runwayml/stable-diffusion-v1-5"
model_path = "album-cover-lora/checkpoint-1000"

pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

image = pipe(
    "Disco, Funk, Synth-pop, Electro album by Daft Punk titled 'Random Access Memories' from 2013, tracks include Get Lucky, Doin' It Right, Lose Yourself To Dance", num_inference_steps=25, guidance_scale=7.5, cross_attention_kwargs={"scale": 1.0}
).images[0]

image.save("LoRATest.png")
