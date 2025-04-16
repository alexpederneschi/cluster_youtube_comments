"""
Handles emojis and sampling
"""
import argparse
import sys
import logging
from functools import partial

import pandas as pd
from tqdm import tqdm
import emoji

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("preprocess")

DEAFULT_SAMPLE_SIZE = 0   # No sampling

def handle_emojis(text, mode='keep'):
    """Process emojis in text (keep/convert/remove)"""

    if mode == 'convert':    
        return emoji.demojize(text)
    elif mode == 'remove':
        return ''.join(c for c in text if c not in emoji.EMOJI_DATA)

def main():
    """
    Preprocesses YouTube comment JSON data with emoji handling and sampling.
    Defaults input to stdin and output to stdout.
    """
    parser = argparse.ArgumentParser(description='Process YouTube comments')
    parser.add_argument('--input', type=argparse.FileType('r'), default=sys.stdin, help='Input JSON file (default: stdin)')
    parser.add_argument('--output', type=argparse.FileType('w', encoding='utf-8'), default=sys.stdout, help='Output JSON file (default: stdout)')
    parser.add_argument('--emojis', choices=['keep', 'convert', 'remove'], default='keep', help='Emoji handling mode')
    parser.add_argument('--sample', type=int, default=DEAFULT_SAMPLE_SIZE, help='Sample size (0 = no sampling)')
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading input...")
    df = pd.read_json(args.input, lines=True)
    df = df.dropna(subset=["Comment"])

    # Process emojis
    if args.emojis != 'keep':
        logger.info(f"Processing emojis ({args.emojis})...")
        tqdm.pandas(desc="Progress")
        process_emojis = partial(handle_emojis, mode=args.emojis)
        df['Comment'] = df['Comment'].progress_apply(process_emojis)
        
    # Sampling
    if args.sample > 0:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
        logger.info(f"Sampled {len(df):,} comments")

    # Save
    df.to_json(args.output, orient='records', lines=True, force_ascii=False)
    logger.info(f"Saved {len(df):,} comments")

if __name__ == '__main__':
    main()