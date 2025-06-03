MODEL = "774M"

from gpt_2_finetuning.encoder import get_encoder
enc = get_encoder(MODEL)
import tensorflow as tf

trainFile = "dataset.txt"

import os
import numpy as np

from gpt_2_finetuning.interactive_conditional_samples import interact_model
from gpt_2_finetuning.train import train

train(dataset_path=trainFile,
      model_name=MODEL,
      n_steps=10000,
      save_every=5000,
      sample_every=1000,
      mem_saving_gradients=True,
      print_loss_every=1000,
      max_checkpoints_to_keep=2)

