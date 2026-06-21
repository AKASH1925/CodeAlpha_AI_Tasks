"""
Music Generation Module
Generates new music sequences using trained LSTM model
"""
import numpy as np
import os
from typing import List, Optional
from music21 import stream, note, chord, instrument
from .model import MusicLSTM
class MusicGenerator:
    """Generates music using trained LSTM model"""
    
    def __init__(self, model: MusicLSTM, vocabulary: dict, inverse_vocabulary: dict):
        self.model = model
        self.vocabulary = vocabulary
        self.inverse_vocabulary = inverse_vocabulary
    
    def generate_sequence(self, length: int = 500, temperature: float = 1.0,
                         seed_sequence: Optional[np.ndarray] = None) -> List[str]:
        """Generate a sequence of notes"""
        print(f"Generating {length} notes with temperature {temperature}...")
        
        if seed_sequence is None:
            seed = [np.random.randint(0, len(self.vocabulary)) 
                   for _ in range(self.model.sequence_length)]
        else:
            seed = seed_sequence.tolist()
        
        generated = []
        current_sequence = list(seed)
        
        for i in range(length):
            input_seq = np.array(current_sequence[-self.model.sequence_length:])
            predicted_index = self.model.predict_next_note(input_seq, temperature)
            
            note_str = self.inverse_vocabulary[predicted_index]
            generated.append(note_str)
            
            current_sequence.append(predicted_index)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{length} notes...")
        
        print("Generation complete!")
        return generated
    
    def notes_to_midi(self, notes: List[str], output_path: str, 
                     instrument_type: str = 'piano') -> None:
        """Convert notes to MIDI file"""
        print(f"Converting {len(notes)} notes to MIDI...")
        
        output_stream = stream.Stream()
        
        if instrument_type == 'piano':
            output_stream.append(instrument.Piano())
        elif instrument_type == 'guitar':
            output_stream.append(instrument.Guitar())
        elif instrument_type == 'violin':
            output_stream.append(instrument.Violin())
        else:
            output_stream.append(instrument.Piano())
        
        for note_str in notes:
            try:
                if '.' in note_str and note_str != 'rest':
                    chord_notes = [int(n) for n in note_str.split('.')]
                    c = chord.Chord(chord_notes)
                    c.quarterLength = 0.5
                    output_stream.append(c)
                elif note_str == 'rest':
                    r = note.Rest()
                    r.quarterLength = 0.5
                    output_stream.append(r)
                else:
                    n = note.Note(note_str)
                    n.quarterLength = 0.5
                    output_stream.append(n)
            except Exception as e:
                print(f"  Skipping invalid note: {note_str} - {e}")
                continue
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        output_stream.write('midi', fp=output_path)
        print(f"MIDI file saved: {output_path}")
    
    def generate_and_save(self, output_path: str, length: int = 500,
                         temperature: float = 1.0, instrument_type: str = 'piano') -> None:
        """Generate music and save directly to MIDI file"""
        notes = self.generate_sequence(length, temperature)
        self.notes_to_midi(notes, output_path, instrument_type)
        
        print(f"\nGenerated MIDI file: {output_path}")
def load_generator(model_path: str, vocabulary: dict, 
                  inverse_vocabulary: dict) -> MusicGenerator:
    """Load a generator with saved model and vocabulary"""
    from .model import MusicLSTM
    
    model = MusicLSTM(len(vocabulary), sequence_length=50)
    model.load_model_from_file(model_path)
    
    return MusicGenerator(model, vocabulary, inverse_vocabulary)