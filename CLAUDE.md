# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIAlbumGenerator is a system that:
1. Generates fictional album descriptions (artist, title, genre, year, tracklist) using a fine-tuned GPT-2 model
2. Creates album artwork for these fictional albums using Stable Diffusion with LoRA fine-tuning
3. Posts the generated albums to Twitter

## Project Architecture

The system consists of several interconnected components:

### Data Collection
- `scrapeDiscogsData.py` and `scrapeDiscogsImages.py` - Scrape album data from Discogs
- `discogsApi.py` - Interface with Discogs API to validate generated albums and check artist popularity
- `postProcessDiscogsImages.py` - Process scraped album artwork
- `cleanImages.py` - Clean and standardize album artwork

### Dataset Creation
- `CreateDataset.py` - Creates training datasets for both text generation and image generation
- `createAlbumDescriptionDataset.py` - Prepares album metadata for GPT-2 fine-tuning
- `createTextToImageDataset.py` - Prepares text-image pairs for Stable Diffusion LoRA fine-tuning
- `DiscogsDataset.py` - Dataset utilities for the Discogs album data

### Model Training
- `trainGPT2.py` - Fine-tune GPT-2 to generate album descriptions
- `trainLoRA.sh` - Fine-tune Stable Diffusion using LoRA for album artwork generation
- `CLIPLikeModel.py` - Implementation of a CLIP-like model for image-text alignment
- `testLoRA.py` - Test the fine-tuned LoRA model

### Generation Pipeline
- `generate.py` - Main script that orchestrates the complete generation process
- `inferAlbumDescription.py` - Generate album descriptions using fine-tuned GPT-2
- `inferAlbumArtwork.py` - Generate album artwork using fine-tuned Stable Diffusion
- `ClipGuidedStableDiffusion.py` and `CLIPLikeGuidedDiffusion.py` - Custom implementations for guided diffusion
- `trimImage.sh` - Post-process generated images
- `twitterApi.py` - Post generated albums to Twitter

## Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Collection and Preprocessing
```bash
# Scrape album data from Discogs
python scrapeDiscogsData.py

# Scrape album artwork from Discogs
python scrapeDiscogsImages.py

# Clean and process images
python cleanImages.py
python postProcessDiscogsImages.py
```

### Dataset Creation
```bash
# Create dataset for GPT-2 fine-tuning
python createAlbumDescriptionDataset.py

# Create dataset for Stable Diffusion LoRA fine-tuning
python CreateDataset.py
```

### Training Models
```bash
# Fine-tune GPT-2 for album descriptions
python trainGPT2.py

# Fine-tune Stable Diffusion with LoRA for album artwork
bash trainLoRA.sh
```

### Generation and Posting
```bash
# Generate an album and post to Twitter
python generate.py

# Alternative using the shell script
bash generate.sh
```

### Testing and Debugging
```bash
# Test album description generation
python inferAlbumDescription.py --temperature 1.0

# Test album artwork generation
python inferAlbumArtwork.py

# Test LoRA model
python testLoRA.py
```

## Environment Variables

The project requires several environment variables to be set in a `.env` file:
- `DISCOGS_TOKEN` - Discogs API token
- `TWITTER_API_KEY` - Twitter API key
- `TWITTER_API_SECRET` - Twitter API secret
- `TWITTER_ACCESS_TOKEN` - Twitter access token
- `TWITTER_ACCESS_TOKEN_SECRET` - Twitter access token secret
- `TWITTER_BEARER_TOKEN` - Twitter bearer token

## File Structure
- `finetuned/` - Directory containing fine-tuned GPT-2 model
- `album-cover-lora/` - Directory containing fine-tuned LoRA models
- `generated_images/` - Directory where generated album artwork is saved