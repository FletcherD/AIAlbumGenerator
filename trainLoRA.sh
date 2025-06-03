export MODEL_NAME="runwayml/stable-diffusion-v1-5"

  accelerate launch diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --dataset_name="/media/media/Storage/Media/DiscogsImageDataset" \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=30000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir="album-cover-lora" \
  --checkpointing_steps=100 \
  --checkpoints_total_limit=5 \
  --validation_prompt="Disco, Funk, Synth-pop, Electro album by Daft Punk titled 'Random Access Memories' from 2013, tracks include Get Lucky, Doin' It Right, Lose Yourself To Dance" \
  --seed=42 \
  --report_to=wandb \
  --resume_from_checkpoint="latest"
