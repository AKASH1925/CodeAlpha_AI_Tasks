"""
MIDI Data Preprocessor
Handles parsing MIDI files and converting to training sequences
"""
import os
import numpy as np
from music21 import converter, instrument, note, chord, stream
from typing import List, Tuple, Dict
class MidiPreprocessor:
    """Preprocesses MIDI files into sequences for training"""
    
    def __init__(self):
        self.notes: List[str] = []
        self.vocabulary: Dict[str, int] = {}
        self.inverse_vocabulary: Dict[int, str] = {}
    
    def load_midi_files(self, directory: str) -> None:
        """Load and parse all MIDI files from directory"""
        print(f"Loading MIDI files from {directory}...")
        
        midi_files = [f for f in os.listdir(directory) if f.endswith(('.mid', '.midi'))]
        
        if not midi_files:
            print(f"No MIDI files found in {directory}")
            print("Creating sample MIDI file for demonstration...")
            self._create_sample_midi(directory)
            midi_files = [f for f in os.listdir(directory) if f.endswith(('.mid', '.midi'))]
        
        for file in midi_files:
            filepath = os.path.join(directory, file)
            print(f"Processing: {file}")
            
            try:
                midi = converter.parse(filepath)
                self._extract_notes(midi)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
        
        print(f"Total notes extracted: {len(self.notes)}")
        self._create_vocabulary()
    
    def _extract_notes(self, midi_stream: stream.Stream) -> None:
        """Extract notes and chords from a MIDI stream"""
        try:
            parts = instrument.partitionByInstrument(midi_stream)
        except:
            parts = None
        
        if parts:
            for element in parts.parts:
                self._parse_element(element)
        else:
            self._parse_element(midi_stream.flat)
    
    def _parse_element(self, element) -> None:
        """Parse notes, chords, and rests from a music21 element"""
        for el in element:
            if isinstance(el, note.Note):
                self.notes.append(str(el.pitch))
            elif isinstance(el, chord.Chord):
                self.notes.append('.'.join([str(n) for n in el.normalOrder]))
            elif isinstance(el, note.Rest):
                self.notes.append('rest')
    
    def _create_sample_midi(self, directory: str) -> None:
        """Create a sample MIDI file for demonstration"""
        sample_notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4',
                       'C5', 'B4', 'A4', 'G4', 'F4', 'E4', 'D4', 'C4']
        
        melody = stream.Stream()
        
        for note_name in sample_notes:
            if note_name != 'rest':
                n = note.Note(note_name)
                n.quarterLength = 0.5
                melody.append(n)
        
        output_path = os.path.join(directory, 'sample_melody.mid')
        melody.write('midi', fp=output_path)
        print(f"Created sample melody: {output_path}")
    
    def _create_vocabulary(self) -> None:
        """Create vocabulary mapping for notes"""
        unique_notes = sorted(set(self.notes))
        self.vocabulary = {note: idx for idx, note in enumerate(unique_notes)}
        self.inverse_vocabulary = {idx: note for note, idx in self.vocabulary.items()}
        
        print(f"Vocabulary size: {len(self.vocabulary)} unique notes")
    
    def prepare_sequences(self, sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare input/output sequences for training"""
        print(f"Preparing sequences with length {sequence_length}...")
        
        X = []
        y = []
        
        note_integers = [self.vocabulary[note] for note in self.notes]
        
        for i in range(len(note_integers) - sequence_length):
            input_seq = note_integers[i:i + sequence_length]
            output_seq = note_integers[i + sequence_length]
            
            X.append(input_seq)
            y.append(output_seq)
        
        X = np.array(X)
        y = np.array(y)
        
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        X = X / float(len(self.vocabulary))
        
        print(f"Training sequences: {X.shape[0]}")
        print(f"Sequence shape: {X.shape}")
        
        return X, y
    
    def get_vocabulary_size(self) -> int:
        return len(self.vocabulary)
    
    def get_vocabulary(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        return self.vocabulary, self.inverse_vocabulary
def prepare_midi_data(data_dir: str, sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """Convenience function to prepare MIDI data for training"""
    preprocessor = MidiPreprocessor()
    preprocessor.load_midi_files(data_dir)
    
    if len(preprocessor.notes) == 0:
        raise ValueError("No notes were extracted from MIDI files")
    
    X, y = preprocessor.prepare_sequences(sequence_length)
    vocab, inv_vocab = preprocessor.get_vocabulary()
    
    return X, y, vocab, inv_vocab