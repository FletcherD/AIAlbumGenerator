import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTokenizer
from PIL import Image
import numpy as np

# Assume we have our CLIPLikeModel defined and trained
import TrainClipLike

class CustomCLIPGuidedStableDiffusion(StableDiffusionPipeline):
    def __init__(self, vae, unet, clip_model, tokenizer):
        super().__init__(vae, unet, None, None, None, None)
        self.clip_model = clip_model
        self.tokenizer = tokenizer

    def encode_prompt(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)

        with torch.no_grad():
            _, text_embeddings = self.clip_model(None, text_input_ids, None)

        # Stable Diffusion expects the embeddings to have a specific shape
        # We might need to adjust the dimensions of our CLIP embeddings
        if text_embeddings.shape[1] != self.unet.config.cross_attention_dim:
            text_embeddings = torch.nn.functional.pad(
                text_embeddings,
                (0, self.unet.config.cross_attention_dim - text_embeddings.shape[1])
            )

        return text_embeddings

    def __call__(self, prompt, num_inference_steps=50, guidance_scale=7.5, **kwargs):
        # Encode the prompt using our custom CLIP model
        text_embeddings = self.encode_prompt(prompt)

        # Prepare unconditioned embeddings for classifier free guidance
        uncond_embeddings = self.encode_prompt([""] * len(prompt))
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Set up the diffusion process
        self.scheduler.set_timesteps(num_inference_steps)
        latents = torch.randn((1, self.unet.in_channels, 64, 64)).to(self.device)

        for t in self.progress_bar(self.scheduler.timesteps):
            # Expand latents for classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode the image
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

def load_custom_pipeline(clip_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the pre-trained Stable Diffusion components
    pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")


    # Load your custom CLIP model
    clip_model = TrainClipLike.CLIPLikeModel()  # Initialize your model architecture

    optimizer = torch.optim.Adam(clip_model.parameters(), lr=1e-4)

    clip_model, optimizer, epoch, loss = TrainClipLike.load_model_checkpoint(clip_model, optimizer, "checkpoint_epoch_37.pth", device)
    clip_model.eval()

    # Create the custom pipeline
    custom_pipeline = CustomCLIPGuidedStableDiffusion(
        vae=pipeline.vae,
        unet=pipeline.unet,
        clip_model=clip_model,
        tokenizer=pipeline.tokenizer,
    )

    return custom_pipeline.to("cuda")


prompt = """Rhythm - Find Yourself Somebody To Love
Genre: Funk / Soul, Soul
Year: 1975"""

# Usage
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = load_custom_pipeline("checkpoint_epoch_37.pth")

    image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5)[0]

    image.save("custom_clip_guided_image.png")
    print("Image generated and saved as 'custom_clip_guided_image.png'")

main()
