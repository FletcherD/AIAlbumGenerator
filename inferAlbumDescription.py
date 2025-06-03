import transformers
import argparse
import time

from generate import parseAlbumDescription, getAlbumImagePrompt

modelPath = './finetuned/'

def inferAlbumDescription(tokenizer, generator, temperature=1.0):
    result = generator(tokenizer.eos_token + '\nArtist:', num_return_sequences=1, temperature=temperature)
    return result[0]["generated_text"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer album descriptions using GPT-2')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for text generation (default: 1.0)')
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(modelPath)
    generator = transformers.pipeline('text-generation', model=modelPath, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id)
    transformers.set_seed(int(time.time()))

    for i in range(10):
        releaseStr = inferAlbumDescription(tokenizer, generator, temperature=args.temperature)
        print(releaseStr)
        try:
            parsedAlbum = parseAlbumDescription(releaseStr)
            imagePrompt = getAlbumImagePrompt(parsedAlbum)
            print(f"{imagePrompt}")
        except Exception as e:
            pass
