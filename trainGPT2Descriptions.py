#!/usr/bin/env python3
"""
Fine-tune GPT-2 on album description dataset for unconditional generation
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
    """Load JSONL dataset for unconditional generation"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            # Use the text directly for unconditional generation
            data.append({"text": item['text']})
    return data

def tokenize_function(examples, tokenizer, max_length=512):
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
    DATASET_FILE = "albumDescriptionDataset.jsonl"
    OUTPUT_DIR = "./gpt2-descriptions-finetuned"
    MAX_LENGTH = 512  # Shorter than reviews since descriptions are more concise
    
    # Check if dataset exists
    if not os.path.exists(DATASET_FILE):
        print(f"Dataset file {DATASET_FILE} not found!")
        print("Please run createAlbumDescriptionDataset.py first to generate the dataset.")
        return
    
    print("Loading tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    
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
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Larger batch since descriptions are shorter
        gradient_accumulation_steps=2,
        warmup_steps=500,
        logging_steps=100,
        save_steps=1000,
        save_strategy="steps",
        dataloader_drop_last=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to=wandb
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Training completed! Model saved to {OUTPUT_DIR}")
    
    # Print a sample generation
    print("\n" + "="*50)
    print("SAMPLE GENERATION:")
    print("="*50)
    
    # Test unconditional generation
    model.eval()
    
    with torch.no_grad():
        # Start with just the beginning token for fully unconditional generation
        input_ids = tokenizer.encode("Artist:", return_tensors="pt")
        
        output = model.generate(
            input_ids,
            max_length=200,
            num_return_sequences=3,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    for i, generated in enumerate(output):
        generated_text = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"Generated Album {i+1}:")
        print(generated_text)
        print("-" * 30)
    
    print("="*50)

if __name__ == "__main__":
    main()
