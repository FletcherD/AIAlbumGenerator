#!/usr/bin/env python3
"""
Script to remove corrupted images that can't be read by PIL.
Uses multiprocessing for concurrent processing with progress tracking.
"""

import argparse
import os
import sys
from multiprocessing import Pool, Manager
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def check_image(args):
    """Check if an image file can be read by PIL."""
    file_path, dry_run, progress_counter, lock = args
    
    try:
        with Image.open(file_path) as img:
            img.verify()
        
        with lock:
            progress_counter.value += 1
        return None
    except Exception:
        with lock:
            progress_counter.value += 1
        return file_path


def get_image_files(directory):
    """Get all image files in the directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    
    for file_path in Path(directory).rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    return image_files


def main():
    parser = argparse.ArgumentParser(description='Remove corrupted images that cannot be read by PIL')
    parser.add_argument('directory', help='Directory to scan for images')
    parser.add_argument('-p', '--processes', type=int, default=4,
                       help='Number of concurrent processes (default: 4)')
    parser.add_argument('-d', '--dry-run', action='store_true',
                       help='Show what would be removed without actually removing files')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a valid directory")
        sys.exit(1)
    
    print(f"Scanning for image files in '{args.directory}'...")
    image_files = get_image_files(args.directory)
    
    if not image_files:
        print("No image files found.")
        return
    
    print(f"Found {len(image_files)} image files")
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be removed")
    
    print(f"Checking images with {args.processes} processes...")
    
    manager = Manager()
    progress_counter = manager.Value('i', 0)
    lock = manager.Lock()
    
    # Prepare arguments for multiprocessing
    check_args = [(file_path, args.dry_run, progress_counter, lock) 
                  for file_path in image_files]
    
    corrupted_files = []
    
    with Pool(processes=args.processes) as pool:
        with tqdm(total=len(image_files), desc="Checking images") as pbar:
            # Use imap to get results as they complete
            for result in pool.imap(check_image, check_args):
                if result is not None:
                    corrupted_files.append(result)
                pbar.update(1)
    
    print(f"\nFound {len(corrupted_files)} corrupted images")
    
    if corrupted_files:
        if args.dry_run:
            print("\nFiles that would be removed:")
            for file_path in corrupted_files:
                print(f"  {file_path}")
            print(f"\nTotal: {len(corrupted_files)} files would be removed")
        else:
            print("\nRemoving corrupted files...")
            removed_count = 0
            for file_path in tqdm(corrupted_files, desc="Removing files"):
                try:
                    os.remove(file_path)
                    removed_count += 1
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")
            
            print(f"\nRemoved {removed_count} corrupted image files")
    else:
        print("No corrupted images found!")


if __name__ == "__main__":
    main()