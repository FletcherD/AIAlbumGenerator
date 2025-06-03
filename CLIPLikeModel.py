import random
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, DistilBertModel, DistilBertTokenizer, ResNetModel
from PIL import Image
import os

import DiscogsDataset

from torchvision.models import resnet50, ResNet50_Weights


class CLIPLikeModel(nn.Module):
    def __init__(self, projection_dim=256):
        super().__init__()


        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        self.image_encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-1])

        image_embedding_dim = 2048
        text_embedding_dim = self.text_encoder.config.hidden_size

        self.image_projection = nn.Linear(image_embedding_dim, projection_dim)
        self.text_projection = nn.Linear(text_embedding_dim, projection_dim)


    def tokenize(self, texts):
        """
        Tokenize the provided text(s) using the model's tokenizer.

        Args:
        texts (str or List[str]): A single text string or a list of text strings to tokenize.

        Returns:
        dict: A dictionary containing 'input_ids' and 'attention_mask' tensors.
        """
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return encoded


    def encode_text(self, texts):
        """
        Encode the provided text(s) into the joint embedding space.

        Args:
        texts (str or List[str]): A single text string or a list of text strings to encode.

        Returns:
        torch.Tensor: The normalized text embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        tokenized = self.tokenize(texts)
        _, text_embeddings = self.forward(None, tokenized["input_ids"], tokenized["attention_mask"])

        return text_embeddings


    def encode_image(self, images):
        """
        Encode the provided image(s) into the joint embedding space.

        Args:
        images (torch.Tensor): A tensor of images of shape (batch_size, channels, height, width).

        Returns:
        torch.Tensor: The normalized image embeddings.
        """
        image_embeddings, _ = self.forward(images, None, None)
        return image_embeddings


    def forward(self, images, input_ids, attention_mask):
        if input_ids is not None:
            text_features = self.text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state[:,0,:]
            text_embeddings = self.text_projection(text_features)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        else:
            text_embeddings = None

        if images is not None:
            image_features = self.image_encoder(images).squeeze(-1).squeeze(-1)  # Shape: (batch_size, 2048)
            image_embeddings = self.image_projection(image_features)
            image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        else:
            image_embeddings = None

        return image_embeddings, text_embeddings



def save_model_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_model_checkpoint(optimizer, checkpoint_path, device=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    model = CLIPLikeModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Loss at checkpoint: {loss:.4f}")

    # Additional information
    print("\nAdditional Checkpoint Info:")
    for key, value in checkpoint.items():
        if key not in ['model_state_dict', 'optimizer_state_dict']:
            print(f"{key}: {value}")

    return model, optimizer, epoch, loss

