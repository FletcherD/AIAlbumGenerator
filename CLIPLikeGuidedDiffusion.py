import random
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import ViTModel, DistilBertModel, DistilBertTokenizer, ResNetModel
from PIL import Image
import os
from datasets import load_dataset
from tqdm import tqdm

import DiscogsDataset
from CreateDataset import getDataset
from torch.utils.data import DataLoader, IterableDataset

from torchvision.models import resnet50, ResNet50_Weights

import CLIPLikeModel


class CLIPGuidedStableDiffusion(StableDiffusionPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_model = None  # We'll set this later

    def set_clip_model(self, clip_model):
        self.clip_model = clip_model

    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # Get text embeddings from our CLIPLikeModel
        text_embeddings = self.clip_model.encode_text(prompt).to(device)

        # Duplicate text embeddings for each generation per prompt
        text_embeddings = text_embeddings.repeat(num_images_per_prompt, 1, 1)

        # Classifier-free guidance
        if do_classifier_free_guidance:
            uncond_tokens = negative_prompt or ""
            uncond_embeddings = self.clip_model.encode_text(uncond_tokens).to(device)
            uncond_embeddings = uncond_embeddings.repeat(batch_size * num_images_per_prompt, 1, 1)
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def __call__(self, prompt, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, negative_prompt=None, num_images_per_prompt=1, **kwargs):
        if self.clip_model is None:
            raise ValueError("CLIP model has not been set. Call set_clip_model() first.")

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Prepare latents
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.unet.config.in_channels,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
        )

        # Denoising loop
        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        # convert to PIL Images
        image = self.numpy_to_pil(image)

        return {"images": image, "nsfw_content_detected": [False] * len(image)}

# Usage
def main():
    # Load your trained CLIPLikeModel
    clip_model = CLIPLikeModel.load_checkpoint("checkpoint_37.pth")
    clip_model.eval()

    # Load Stable Diffusion 2.1
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = CLIPGuidedStableDiffusion.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # Set our custom CLIP model
    pipe.set_clip_model(clip_model)

    # Generate an image
    prompt = "A serene lake surrounded by autumn trees, photorealistic style"
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    # Save the generated image
    image.save("clip_guided_sd_image.png")

if __name__ == "__main__":
    main()
