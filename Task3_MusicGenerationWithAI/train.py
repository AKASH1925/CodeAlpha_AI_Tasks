"""
Main Training Script for Music Generation
"""
import os
import sys
import argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_preprocessor import MidiPreprocessor, prepare_midi_data
from src.model import MusicLSTM, create_model
from src.utils import (save_vocabulary, plot_training_history, 
                      analyze_notes, print_stats, ensure_directories,
                      calculate_sequence_length)
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Music Generation Model')
    
    parser.add_argument('--data_dir', type=str, default='data/midi',
                       help='Directory containing MIDI files')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save model and vocabulary')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--sequence_length', type=int, default=50,
                       help='Length of input sequences')
    parser.add_argument('--lstm_units', type=int, default=256,
                       help='Number of units in LSTM layers')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--simple', action='store_true',
                       help='Use simpler model architecture')
    parser.add_argument('--plot', action='store_true',
                       help='Plot training history')
    
    return parser.parse_args()
def main():
    """Main training function"""
    args = parse_args()
    
    print("=" * 60)
    print("Music Generation with AI - Training Script")
    print("=" * 60)
    
    ensure_directories()
    
    print(f"\n1. Loading MIDI files from {args.data_dir}...")
    
    try:
        X, y, vocabulary, inverse_vocabulary = prepare_midi_data(
            args.data_dir, 
            args.sequence_length
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure you have MIDI files in the data/midi/ directory")
        return
    
    preprocessor = MidiPreprocessor()
    preprocessor.load_midi_files(args.data_dir)
    stats = analyze_notes(preprocessor.notes)
    print_stats(stats)
    
    seq_length = calculate_sequence_length(len(y), args.sequence_length)
    
    print(f"\n2. Creating LSTM model...")
    vocabulary_size = len(vocabulary)
    
    music_model = create_model(
        vocabulary_size=vocabulary_size,
        sequence_length=args.sequence_length,
        simple=args.simple,
        lstm_units=args.lstm_units
    )
    
    print(f"\n3. Training model...")
    history = music_model.train(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2
    )
    
    print(f"\n4. Saving model and vocabulary...")
    
    model_path = os.path.join(args.output_dir, 'music_lstm_model.keras')
    vocab_path = os.path.join(args.output_dir, 'vocabulary.json')
    inv_vocab_path = os.path.join(args.output_dir, 'inverse_vocabulary.json')
    
    music_model.save_model(model_path)
    save_vocabulary(vocabulary, vocab_path)
    save_vocabulary(inverse_vocabulary, inv_vocab_path)
    
    if args.plot:
        plot_path = os.path.join(args.output_dir, 'training_history.png')
        plot_training_history(history, plot_path)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {model_path}")
    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Final training accuracy: {history['accuracy'][-1]:.4f}")
    print("\nNext steps:")
    print("  Run 'python generate_music.py' to generate new music!")
if __name__ == '__main__':
    main()