from PIL import Image
import os
from tqdm import tqdm

def pad_to_square(image_path, output_path, target_size=512):
    """Resize image to fit in square with white padding"""
    img = Image.open(image_path)

    # Calculate new size maintaining aspect ratio
    img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

    # Create white square canvas
    canvas = Image.new('RGB', (target_size, target_size), 'white')

    # Center the image on canvas
    x = (target_size - img.width) // 2
    y = (target_size - img.height) // 2
    canvas.paste(img, (x, y))

    canvas.save(output_path, 'JPEG', quality=95)

# Process your dataset
input_dir = "/media/media/Storage/Media/DiscogsImageDatasetRaw"
output_dir = "/media/media/Storage/Media/DiscogsImageDataset"
os.makedirs(output_dir, exist_ok=True)

for filename in tqdm(os.listdir(input_dir)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        if os.path.exists(output_path):
            continue
        try:
            pad_to_square(input_path, output_path)
        except:
            print(input_path)
            continue
