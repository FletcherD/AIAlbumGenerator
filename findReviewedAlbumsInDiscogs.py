#!/usr/bin/env python3
"""
Script to find albums from reviews.csv in Discogs and add missing ones to the database.

This script:
1. Reads the reviews.csv dataset
2. For each album, searches Discogs by artist and title
3. Checks if each result is already in our database
4. If not, fetches the release data from Discogs and adds it to the database
"""

import csv
import time
import pandas as pd
from typing import List, Dict, Optional

import discogsApi
import DiscogsDataset

# Just for when it gets interrupted
START_REVIEW = 250

def load_reviews_data(csv_path: str = 'reviews.csv') -> List[Dict[str, str]]:
    """Load reviews data from CSV file."""
    reviews = []
    
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            reviews.append({
                'artist': row['artist'],
                'album': row['album'],
                'year_released': row['year_released'] if pd.notna(row['year_released']) else None
            })
    except Exception as e:
        print(f"Error loading reviews data: {e}")
        return []
    
    return reviews


def search_and_add_album(artist: str, album: str, year: Optional[str] = None) -> int:
    """
    Search for an album in Discogs and add missing releases to the database.
    
    Args:
        artist: Artist name
        album: Album title
        year: Release year (optional, for verification)
        
    Returns:
        Number of new releases added to the database
    """
    print(f"Searching for: {artist} - {album}")
    
    # Search Discogs for the album
    search_results = discogsApi.searchReleaseByArtistAndTitle(artist, album)
    
    if not search_results:
        print(f"  No results found for {artist} - {album}")
        return 0
    
    added_count = 0
    
    # Process each search result
    for result in search_results[:1]:
        if 'id' not in result:
            continue
            
        release_id = result['id']
        
        # Check if we already have this release in our database
        existing_release = DiscogsDataset.getRelease(release_id)
        
        if existing_release is not None:
            print(f"  Release {release_id} already in database")
            continue
        
        # Fetch full release data from Discogs
        print(f"  Fetching release {release_id} from Discogs...")
        try:
            release_data = discogsApi.getReleaseInfo(release_id)
            
            # Check if we got valid data
            if 'message' in release_data:
                print(f"  Error fetching release {release_id}: {release_data['message']}")
                continue
            
            # Add to database
            DiscogsDataset.addRelease(release_id, release_data)
            print(f"  Added release {release_id} to database: {release_data.get('title', 'Unknown Title')}")
            added_count += 1
            
        except Exception as e:
            print(f"  Error processing release {release_id}: {e}")
            continue
    
    return added_count


def main():
    """Main function to process all reviewed albums."""
    print("Loading reviews data...")
    reviews = load_reviews_data()
    
    if not reviews:
        print("No reviews data found. Exiting.")
        return
    
    print(f"Found {len(reviews)} albums in reviews dataset")
    
    total_added = 0
    processed = 0
    
    for review in reviews[START_REVIEW:]:
        artist = review['artist']
        album = review['album']
        year = review['year_released']
        
        try:
            added = search_and_add_album(artist, album, year)
            total_added += added
            processed += 1
            
            print(f"Progress: {processed}/{len(reviews)} albums processed, {total_added} releases added")
            
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"Error processing {artist} - {album}: {e}")
            continue
    
    print(f"\nCompleted! Processed {processed} albums, added {total_added} new releases to database.")


if __name__ == "__main__":
    main()
