#!/usr/bin/env python3
"""
Generate album reviews using fine-tuned GPT-2 model
"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
import os

def generate_review(model, tokenizer, album_description, max_length=200, temperature=0.8, num_samples=1):
    """Generate a review for the given album description"""
    
    # Format input with special tokens
    prompt = f"<|album|>{album_description.strip()}<|review|>"
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_length,
            num_return_sequences=num_samples,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    # Decode and extract review part
    reviews = []
    for output in outputs:
        full_text = tokenizer.decode(output, skip_special_tokens=True)
        
        # Extract just the review part (after <|review|>)
        if "<|review|>" in full_text:
            review = full_text.split("<|review|>", 1)[1].strip()
            reviews.append(review)
        else:
            reviews.append(full_text)
    
    return reviews

def main():
    parser = argparse.ArgumentParser(description="Generate album reviews using fine-tuned GPT-2")
    parser.add_argument("--model_path", default="./gpt2-review-finetuned", 
                       help="Path to fine-tuned model")
    parser.add_argument("--album_description", type=str,
                       help="Album description to generate review for")
    parser.add_argument("--max_length", type=int, default=200,
                       help="Maximum length of generated review")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Temperature for generation (higher = more creative)")
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of review samples to generate")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        print("Please train the model first using trainGPT2Review.py")
        return
    
    print("Loading model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    
    if args.interactive:
        print("Interactive mode - Enter album descriptions to generate reviews")
        print("Format: Artist: [name]\\nTitle: [title]\\nGenre: [genre]\\nYear: [year]\\n\\t1: [track1]\\n\\t2: [track2]...")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            print("\nEnter album description (or 'quit' to exit):")
            description = input().strip()
            
            if description.lower() == 'quit':
                break
                
            if not description:
                continue
            
            reviews = generate_review(
                model, tokenizer, description, 
                args.max_length, args.temperature, args.num_samples
            )
            
            print(f"\nGenerated Review{'s' if len(reviews) > 1 else ''}:")
            print("-" * 30)
            for i, review in enumerate(reviews, 1):
                if len(reviews) > 1:
                    print(f"Sample {i}:")
                print(review)
                if i < len(reviews):
                    print("-" * 30)
    
    else:
        if not args.album_description:
            print("Please provide --album_description or use --interactive mode")
            return
        
        reviews = generate_review(
            model, tokenizer, args.album_description,
            args.max_length, args.temperature, args.num_samples
        )
        
        print("Generated Review:")
        print("-" * 30)
        for review in reviews:
            print(review)

if __name__ == "__main__":
    main()