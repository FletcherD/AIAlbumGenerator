import math

import DiscogsDataset
import os
import csv
import re
import json
import random
import glob
import langdetect
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from datasets import load_dataset

def getArtistStr(release):
    r = ''
    if 'artists' not in release:
        return None
    for artist in release['artists']:
        artistName = artist['name']
        artistName = re.sub(r' \([0-9]*\)', '', artistName)
        r += artistName
        if artist['join'] != '':
            if artist['join'] != ',':
                r += ' '
            r += artist['join'] + ' '
    return r

def getGenreStr(release):
    genreList = []
    if 'genres' in release:
        genreList += release['genres']
    if 'styles' in release:
        genreList += release['styles']
    else:
        genreList = ['None']
    genreSubstitutions = {'Folk, World, & Country': "Folk / World / Country"}
    for i, genre in enumerate(genreList):
        if genre in genreSubstitutions:
            genreList[i] = genreSubstitutions[genre]
    #genreList = set(genreList)
    return ', '.join(genreList)

def getReleaseDescriptionStr(release):
    r = ''
    r += 'Artist: {}\n'.format(getArtistStr(release))
    r += 'Title: {}\n'.format(release['title'])
    r += 'Genre: {}\n'.format(getGenreStr(release))
    r += 'Year: {}\n'.format(release['year'])
    for track in release['tracklist']:
         r += '\t{position}: {title}\n'.format(**track)
    return r


def normalize_for_matching(text):
    """Normalize text for matching between datasets"""
    if not text:
        return ""
    # Convert to lowercase, remove extra spaces, and strip
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    # Remove common punctuation that might cause mismatches
    normalized = re.sub(r'[.,!?;:"\'()]', '', normalized)
    return normalized

def createAlbumDescriptionDataset():
    # Load reviews dataset
    print("Loading reviews dataset...")
    reviews_df = pd.read_csv('reviews.csv')
    
    # Create a dictionary for fast lookup of reviews by (artist, album)
    reviews_dict = {}
    for _, row in reviews_df.iterrows():
        try:
            artist_norm = normalize_for_matching(str(row['artist']))
            album_norm = normalize_for_matching(str(row['album']))
        except Exception as e:
            print(row)
        key = (artist_norm, album_norm)
        reviews_dict[key] = row['small_text']
    
    print(f"Loaded {len(reviews_dict)} reviews for matching")
    
    releaseIds = DiscogsDataset.getAllReleaseIds()
    seen_albums = set()  # Track unique artist-title combinations
    processed_count = 0
    duplicate_count = 0
    review_matches = 0
    
    with open('albumDescriptionDataset.jsonl', 'w') as desc_file, \
         open('albumReviewDataset.jsonl', 'w') as review_file:
        
        for idNum in tqdm(releaseIds, desc="Processing releases"):
            try:
                release = DiscogsDataset.getRelease(idNum)

                # Create unique identifier for this album
                artist_str = str(getArtistStr(release))
                title_str = str(release['title'])

                if not artist_str or not title_str:
                    continue

                # Normalize for comparison (lowercase, strip whitespace)
                album_key = (artist_str.lower().strip(), title_str.lower().strip())

                # Skip if we've already seen this album
                if album_key in seen_albums:
                    duplicate_count += 1
                    continue

                # Add to seen set
                seen_albums.add(album_key)

                releaseStr = getReleaseDescriptionStr(release)

                language = langdetect.detect(releaseStr)
                if language != 'en':
                    continue

                # Write album description dataset
                trainingStr = getReleaseDescriptionStr(release)
                trainingJson = {'note': '### '+trainingStr}
                desc_file.write(json.dumps(trainingJson) + '\n')
                processed_count += 1

                # Check for matching review
                artist_norm = normalize_for_matching(artist_str)
                title_norm = normalize_for_matching(title_str)
                review_key = (artist_norm, title_norm)

                if review_key in reviews_dict:
                    # Found a matching review, create training pair
                    review_text = reviews_dict[review_key]
                    review_training_json = {
                        'input': trainingStr.strip(),
                        'output': review_text.strip()
                    }
                    review_file.write(json.dumps(review_training_json) + '\n')
                    review_matches += 1

                # Print progress every 10000 processed albums
                if processed_count % 10000 == 0:
                    tqdm.write(f"Processed {processed_count} unique albums, skipped {duplicate_count} duplicates, found {review_matches} review matches")

            except Exception as e:
                print(f"{artist_str} - {release['title']}")
                traceback.print_exc()
    
    print(f"Final stats: {processed_count} unique albums processed, {duplicate_count} duplicates skipped, {review_matches} review matches found")
    print(f"Review dataset saved to albumReviewDataset.jsonl")


if __name__ == '__main__':
    createAlbumDescriptionDataset()
