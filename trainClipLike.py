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


def contrastive_loss(image_embeddings, text_embeddings, temperature=0.07):
    logits = torch.matmul(image_embeddings, text_embeddings.t()) / temperature
    labels = torch.arange(len(image_embeddings), device=image_embeddings.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2


def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    #texts = [item['text'] for item in batch]  # Original texts
    return {'images': images, 'input_ids': input_ids, 'attention_mask': attention_mask}


def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return CreateDataset.imageTransform(image).unsqueeze(0)


def encode_text_corpus(model, text_corpus, tokenizer, device, batch_size=32, max_length=77):
    model.eval()
    text_embeddings = []

    for i in tqdm(range(0, len(text_corpus), batch_size), desc="Encoding text corpus"):
        batch = text_corpus[i:i + batch_size]
        encoded_text = model.tokenize(batch)
        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)

        with torch.no_grad():
            _, text_embedding = model(None, input_ids, attention_mask)

        text_embeddings.append(text_embedding)

    return torch.cat(text_embeddings, dim=0)


def find_closest_texts(image_embedding, text_embeddings, text_corpus, top_k=5):
    # Compute cosine similarity
    #similarity = np.abs(F.cosine_similarity(image_embedding.unsqueeze(0), text_embeddings))

    image_embedding = image_embedding.cpu()
    text_embeddings = text_embeddings.cpu()

    from scipy.spatial.distance import cosine
    similarity = [cosine(image_embedding, text_embeddings[i,:]) for i in range(text_embeddings.shape[0])]

    # Get top-k matches
    top_indices = sorted(range(len(similarity)), key=lambda i: abs(similarity[i]), reverse=False)[:top_k]

    return [(text_corpus[i], similarity[i]) for i in top_indices]




# Training function
def train_clip_like_model(model, dataloader, optimizer, device, epochs):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        # Create tqdm progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        for batch in progress_bar:
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            image_embeddings, text_embeddings = model(images, input_ids, attention_mask)
            loss = contrastive_loss(image_embeddings, text_embeddings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs} completed. Average Loss: {avg_loss:.4f}")

        # Save checkpoint after each epoch
        checkpoint_filename = f"checkpoint_epoch_{epoch + 1}.pth"
        save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_filename)


def getRandomImage():
    import glob
    image_list = glob.glob(os.path.join(DiscogsDataset.IMAGE_DIR_PATH, '*.jpg'))
    return random.choice(image_list)


# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and optimizer
    model = CLIPLikeModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 1000

    model, optimizer, epoch, loss = load_model_checkpoint(model, optimizer, "checkpoint_epoch_37.pth", device)
    print("\nCheckpoint loaded successfully!")
    print(f"Model is now at epoch {epoch} with loss {loss:.4f}")

    # You can now use the loaded model for inference or continue training
    model.eval()  # Set the model to evaluation mode

    dataset = getDataset(tokenizer)
    text_corpus = [r['text'] for r in dataset.data]
    # text_embeddings = encode_text_corpus(model, text_corpus, tokenizer, batch_size=128, device=device)
    # torch.save(text_embeddings, 'text_embeddings.pth')

    text_embeddings = torch.load('text_embeddings.pth')
    print(text_embeddings.shape)

    for i in range(10):

        n = random.choice(range(len(text_corpus)))
        text = text_corpus[n]
        print(text)

        embedding = text_embeddings[n,:]

        # Find closest text matches
        closest_matches = find_closest_texts(embedding, text_embeddings, text_corpus)

        # Print results
        for text, similarity in closest_matches:
            print(f"Similarity: {similarity:.4f} \n{text}")

        print('------------------')
    #
    #
    # dataloader = DataLoader(dataset, batch_size=72, collate_fn=collate_fn, num_workers=1, prefetch_factor=2, pin_memory=True)
    #
    # train_clip_like_model(model, dataloader, optimizer, device, epochs=num_epochs)

