"""
Utility Functions for Music Generation
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
def save_vocabulary(vocabulary: Dict[str, int], filepath: str) -> None:
    """Save vocabulary mapping to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(vocabulary, f, indent=2)
    print(f"Vocabulary saved to {filepath}")
def load_vocabulary(filepath: str) -> Dict[str, int]:
    """Load vocabulary mapping from JSON file"""
    with open(filepath, 'r') as f:
        vocabulary = json.load(f)
    print(f"Vocabulary loaded from {filepath}")
    return vocabulary
def create_inverse_vocabulary(vocabulary: Dict[str, int]) -> Dict[int, str]:
    """Create inverse vocabulary mapping"""
    return {idx: note for note, idx in vocabulary.items()}
def plot_training_history(history: dict, save_path: str = None) -> None:
    """Plot training history (loss and accuracy)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['loss'], label='Training Loss', color='blue')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['accuracy'], label='Training Accuracy', color='blue')
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training plot saved to {save_path}")
    
    plt.show()
def analyze_notes(notes: list) -> dict:
    """Analyze statistics of notes"""
    from collections import Counter
    
    stats = {
        'total_notes': len(notes),
        'unique_notes': len(set(notes)),
        'most_common': Counter(notes).most_common(10),
        'note_distribution': dict(Counter(notes))
    }
    return stats
def print_stats(stats: dict) -> None:
    """Print note statistics"""
    print("\n=== Note Statistics ===")
    print(f"Total notes: {stats['total_notes']}")
    print(f"Unique notes: {stats['unique_notes']}")
    print(f"\nTop 10 most common notes:")
    for note, count in stats['most_common']:
        print(f"  {note}: {count}")
def validate_midi_file(filepath: str) -> bool:
    """Validate that a MIDI file exists and is readable"""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False
    
    try:
        from music21 import converter
        midi = converter.parse(filepath)
        return True
    except Exception as e:
        print(f"Error parsing MIDI file: {e}")
        return False
def ensure_directories() -> None:
    """Ensure all required directories exist"""
    dirs = ['data/midi', 'models', 'generated']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Directory ready: {d}")
def calculate_sequence_length(data_size: int, recommended: int = 50) -> int:
    """Calculate appropriate sequence length based on data size"""
    if data_size < recommended * 10:
        adjusted = max(10, data_size // 10)
        print(f"Adjusted sequence length from {recommended} to {adjusted} for small dataset")
        return adjusted
    return recommended