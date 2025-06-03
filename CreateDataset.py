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

prompt = """{genreStr} album by {artistStr} titled '{title}' from {year}, tracks include {firstTracksStr}"""


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

def getImagePromptStr(release):
    trackNames = [track['title'] for track in release['tracklist']]
    #print(trackNames)
    firstTracksStr = ', '.join(trackNames[:3])
    return prompt.format(**release, artistStr=getArtistStr(release), genreStr=getGenreStr(release), firstTracksStr=firstTracksStr)

def getReleaseDescriptionStr(release):
    r = ''
    #r += 'Artist: {}\n'.format(getArtistStr(release))
    #r += 'Title: {}\n'.format(release['title'])
    r += '{} - {}\n'.format(getArtistStr(release), release['title'])
    r += 'Genre: {}\n'.format(getGenreStr(release))
    r += 'Year: {}\n'.format(release['year'])
    # for track in release['tracklist']:
    #     r += '\t{position}: {title}\n'.format(**track)
    return r


def removeImage(idNum):
    imPath = os.path.join(DiscogsDataset.IMAGE_DIR_PATH, str(idNum) + '.jpg')
    print("Removing "+imPath)
    os.remove(imPath)


def removeImagesWithoutMetadata():
    for imPath in glob.glob(os.path.join(DiscogsDataset.IMAGE_DIR_PATH, '*.jpg')):
        idNum = re.findall('[0-9]+', imPath)
        idNum = int(idNum[0])
        if not idNum in releaseIds:
            removeImage(idNum)


def createAlbumDescriptionDataset():
    with open('albumDescriptionDataset.jsonl', 'w') as f:
        for idNum in tqdm(releaseIds):
            try:
                release = DiscogsDataset.getRelease(idNum)
                releaseStr = getReleaseDescriptionStr(release)

                language = langdetect.detect(releaseStr)
                if language != 'en':
                    continue

                trainingStr = getReleaseDescriptionStr(release)
                trainingJson = {'note': '### '+trainingStr}
                f.write(json.dumps(trainingJson) + '\n')
            except Exception as e:
                print(e)
                print(idNum)


def createTextToImageDataset():
    random.seed("dongs")
    releaseIds = DiscogsDataset.getAllReleaseIds()
    print('{} entries'.format(len(releaseIds)))
    random.shuffle(releaseIds)

    lines_written = 0
    datasetPath = os.path.join(DiscogsDataset.IMAGE_DIR_PATH, 'dataset.csv')
    with open(datasetPath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
        writer.writerow(['filename', 'text'])

        for idNum in tqdm(releaseIds):
            image_filename = os.path.join(DiscogsDataset.IMAGE_DIR_PATH, str(idNum) + '.jpg')
            if not os.path.exists(image_filename):
                continue

            release = DiscogsDataset.getRelease(idNum)

            if not 'genres' in release or release['year'] == 0:
                #removeImage(idNum)
                continue

            try:
                release_text = getImagePromptStr(release)
                writer.writerow([image_filename, release_text])
                lines_written += 1
            except Exception as e:
                print(e)
                print(release)
                #removeImage(idNum)

    print(f'Wrote {lines_written} entries')


def resize_with_pad(image, target_size, fill_color=(0, 0, 0)):
    ratio = max(target_size[0] / image.size[0], target_size[1] / image.size[1])
    new_size = tuple([int(x * ratio) for x in image.size])
    image = image.resize(new_size, Image.BILINEAR)

    new_im = Image.new("RGB", target_size, fill_color)
    new_im.paste(image, ((target_size[0] - new_size[0]) // 2,
                         (target_size[1] - new_size[1]) // 2))
    return new_im


class ResizeWithPad:
    def __init__(self, size, fill_color=(0, 0, 0)):
        self.size = size
        self.fill_color = fill_color

    def __call__(self, image):
        return resize_with_pad(image, self.size, self.fill_color)


imageTransform = transforms.Compose([
            ResizeWithPad((224, 224), fill_color=(255,255,255)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class StreamingImageTextDataset(IterableDataset):
    def __init__(self, csv_file, tokenizer, max_length=77):
        self.data = load_dataset('csv',
            data_files=csv_file,
            split='train',
            quoting=csv.QUOTE_ALL,
            escapechar='\\',
            doublequote=True,
            encoding='utf-8'
        )
        self.data = self.data.shuffle()
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return self.data.num_rows

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            n_workers = 1
            worker_id = 0
        else:
            n_workers = worker_info.num_workers
            worker_id = worker_info.id

        for i, item in enumerate(self.data):
            if i % n_workers != worker_id:
                continue

            try:
                image = Image.open(item['filename']).convert('RGB')
                image = imageTransform(image)

                text_encoding = self.tokenizer(
                    item['text'],
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                yield {
                    'image': image,
                    'input_ids': text_encoding.input_ids.squeeze(),
                    'attention_mask': text_encoding.attention_mask.squeeze(),
                    'text': item['text']  # Keep original text for reference
                }
            except Exception as e:
                print(e)
                print(item['filename'])
                pass


def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    texts = [item['text'] for item in batch]  # Original texts
    return {'images': images, 'input_ids': input_ids, 'attention_mask': attention_mask}



def getDataset(tokenizer):
    return StreamingImageTextDataset(DiscogsDataset.DATASET_CSV_PATH, tokenizer)



if __name__ == '__main__':
    createTextToImageDataset()


def demonstrateDataset():
    from transformers import ViTModel, DistilBertModel, DistilBertTokenizer, ResNetModel
    text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    from CreateDataset import getDataset
    dataset = getDataset(tokenizer)

    from torch.utils.data import DataLoader, IterableDataset
    dataloader = DataLoader(dataset, batch_size=72, collate_fn=collate_fn, num_workers=8, prefetch_factor=2, pin_memory=False)

    for i, d in enumerate(dataloader):  print(i)

