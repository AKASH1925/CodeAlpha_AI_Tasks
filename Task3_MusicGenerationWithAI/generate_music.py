"""
Main Music Generation Script
"""
import os
import sys
import argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.generate import MusicGenerator, load_generator
from src.utils import load_vocabulary, create_inverse_vocabulary, ensure_directories
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate Music with AI')
    
    parser.add_argument('--model_path', type=str, default='models/music_lstm_model.keras',
                       help='Path to trained model')
    parser.add_argument('--vocab_path', type=str, default='models/vocabulary.json',
                       help='Path to vocabulary file')
    parser.add_argument('--output_dir', type=str, default='generated',
                       help='Directory to save generated MIDI files')
    parser.add_argument('--length', type=int, default=500,
                       help='Number of notes to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Randomness factor (0.0-1.0)')
    parser.add_argument('--instrument', type=str, default='piano',
                       choices=['piano', 'guitar', 'violin'],
                       help='Instrument type')
    parser.add_argument('--num_files', type=int, default=3,
                       help='Number of files to generate')
    
    return parser.parse_args()
def main():
    """Main generation function"""
    args = parse_args()
    
    print("=" * 60)
    print("Music Generation with AI - Generation Script")
    print("=" * 60)
    
    ensure_directories()
    
    if not os.path.exists(args.model_path):
        print(f"\nError: Model file not found at {args.model_path}")
        print("Please train the model first using: python train.py")
        return
    
    print(f"\n1. Loading vocabulary from {args.vocab_path}...")
    vocabulary = load_vocabulary(args.vocab_path)
    inverse_vocabulary = create_inverse_vocabulary(vocabulary)
    
    print(f"\n2. Loading trained model...")
    generator = load_generator(args.model_path, vocabulary, inverse_vocabulary)
    
    print(f"\n3. Generating music...")
    print(f"   Notes to generate: {args.length}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Instrument: {args.instrument}")
    
    for i in range(args.num_files):
        output_path = os.path.join(args.output_dir, f'generated_music_{i+1}.mid')
        
        print(f"\n   Generating file {i+1}/{args.num_files}...")
        generator.generate_and_save(
            output_path=output_path,
            length=args.length,
            temperature=args.temperature,
            instrument_type=args.instrument
        )
    
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Generated {args.num_files} MIDI files in: {args.output_dir}/")
    print("\nYou can:")
    print("  - Play them with any MIDI player")
    print("  - Convert to audio using Audacity")
    
    print("\nTips:")
    print("  - Try different temperatures (0.5-1.0)")
    print("  - Lower temperature = more predictable")
    print("  - Higher temperature = more random")
if __name__ == '__main__':
    main()