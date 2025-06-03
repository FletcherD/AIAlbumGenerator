#!/bin/bash

cd /media/media/Storage/Media/discogs-gpt2

source ./.venv/bin/activate

python3 generate.py | tee ./logs/generation_$(date +%s).txt
