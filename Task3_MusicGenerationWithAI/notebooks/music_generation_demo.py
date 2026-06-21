"""
Music Generation Demo Script
Run this to quickly test the system with sample data
"""
import os
import sys
# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessor import MidiPreprocessor
from src.model import MusicLSTM
from src.generate import MusicGenerator
from src.utils import ensure_directories
def main():
    """Run a quick demo"""
    print("=" * 60)
    print("Music Generation Demo")
    print("=" * 60)
    
    # Ensure directories exist
    ensure_directories()
    
    # Preprocessor will create sample MIDI if none exist
    data_dir = 'data/midi'
    
    print("\n1. Loading MIDI data...")
    preprocessor = MidiPreprocessor()
    preprocessor.load_midi_files(data_dir)
    
    if len(preprocessor.notes) == 0:
        print("No notes found. Check your MIDI files.")
        return
    
    print(f"\nFound {len(preprocessor.notes)} notes")
    
    # Prepare sequences
    print("\n2. Preparing training sequences...")
    sequence_length = 50
    X, y = preprocessor.prepare_sequences(sequence_length)
    vocabulary = preprocessor.vocabulary
    inverse_vocabulary = preprocessor.inverse_vocabulary
    
    print(f"Training data shape: {X.shape}")
    
    # Build and train model (quick training for demo)
    print("\n3. Building and training model (quick demo - 10 epochs)...")
    model = MusicLSTM(len(vocabulary), sequence_length)
    model.build_simple_model(lstm_units=128)
    
    history = model.train(X, y, epochs=10, batch_size=32, validation_split=0.2)
    
    print(f"\nTraining accuracy: {history['accuracy'][-1]:.2%}")
    
    # Generate music
    print("\n4. Generating music...")
    generator = MusicGenerator(model, vocabulary, inverse_vocabulary)
    
    output_path = 'generated/demo_output.mid'
    generator.generate_and_save(
        output_path=output_path,
        length=200,
        temperature=0.8,
        instrument_type='piano'
    )
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print(f"Generated MIDI file: {output_path}")
    print("\nPlay the file with any MIDI player to hear the result!")
    print("\nTo train with your own MIDI files:")
    print("  1. Add MIDI files to data/midi/")
    print("  2. Run: python train.py --epochs 50")
    print("  3. Generate: python generate_music.py")
if __name__ == '__main__':
    main()