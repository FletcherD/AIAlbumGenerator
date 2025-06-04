#!/usr/bin/env python3
"""
Fine-tune GPT-2 on album review dataset using Hugging Face transformers
"""
import json
import torch
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    DataCollatorForLanguageModeling,
    TrainingArguments, 
    Trainer
)
import os

def load_jsonl_dataset(file_path):
    """Load JSONL dataset and format for GPT-2 training"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            # Format as: Album description -> Review
            # Using special tokens to separate input and output
            text = f"<|album|>{item['input']}<|review|>{item['output']}<|endoftext|>"
            data.append({"text": text})
    return data

def tokenize_function(examples, tokenizer, max_length=1024):
    """Tokenize the dataset"""
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding=False, 
        max_length=max_length,
        return_overflowing_tokens=False,
    )

def main():
    # Configuration
    MODEL_NAME = "gpt2"  # or "gpt2-medium", "gpt2-large", etc.
    DATASET_FILE = "albumReviewDataset.jsonl"
    OUTPUT_DIR = "./gpt2-review-finetuned"
    MAX_LENGTH = 1024
    
    # Check if dataset exists
    if not os.path.exists(DATASET_FILE):
        print(f"Dataset file {DATASET_FILE} not found!")
        print("Please run createAlbumDescriptionDataset.py first to generate the dataset.")
        return
    
    print("Loading tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    
    # Add special tokens
    special_tokens = {
        "additional_special_tokens": ["<|album|>", "<|review|>"]
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Set pad token to eos token
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading and tokenizing dataset...")
    raw_dataset = load_jsonl_dataset(DATASET_FILE)
    print(f"Loaded {len(raw_dataset)} training examples")
    
    # Create dataset
    dataset = Dataset.from_list(raw_dataset)
    
    # Tokenize
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    train_dataset = tokenized_dataset
    
    print(f"Training samples: {len(train_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 is causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=False,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        logging_steps=100,
        save_steps=500,
        save_strategy="steps",
        save_total_limit=5,  # Only keep 5 most recent checkpoints
        dataloader_drop_last=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to="wandb",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # Check for existing checkpoints and resume if available
    checkpoint_dir = None
    if os.path.exists(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            # Sort by checkpoint number to get the latest
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = checkpoints[-1]
            checkpoint_dir = os.path.join(OUTPUT_DIR, latest_checkpoint)
            print(f"Found checkpoint: {checkpoint_dir}")
    
    print("Starting training...")
    if checkpoint_dir:
        print(f"Resuming from checkpoint: {checkpoint_dir}")
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        print("Starting training from scratch")
        trainer.train()
    
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Training completed! Model saved to {OUTPUT_DIR}")
    
    # Print a sample generation
    print("\n" + "="*50)
    print("SAMPLE GENERATION:")
    print("="*50)
    
    # Test generation
    model.eval()
    device = next(model.parameters()).device
    sample_input = "<|album|>Artist: The Beatles\nTitle: Abbey Road\nGenre: Rock\nYear: 1969\n\t1: Come Together\n\t2: Something<|review|>"
    
    input_ids = tokenizer.encode(sample_input, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + 100,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    print(generated_text)
    print("="*50)

if __name__ == "__main__":
    main()
