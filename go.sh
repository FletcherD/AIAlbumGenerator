#!/bin/bash

. "/home/media/anaconda3/etc/profile.d/conda.sh"
conda activate discogs

cd /home/media/discogs-gpt2
python3 generate.py
