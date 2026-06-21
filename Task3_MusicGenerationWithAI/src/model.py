"""
LSTM Neural Network Model for Music Generation
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
class MusicLSTM:
    """LSTM model for music generation"""
    
    def __init__(self, vocabulary_size: int, sequence_length: int = 50):
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length
        self.model = None
        self.history = None
    
    def build_model(self, lstm_units: int = 256, learning_rate: float = 0.001) -> Sequential:
        """Build the full LSTM model architecture"""
        print("Building LSTM model...")
        
        self.model = Sequential([
            LSTM(lstm_units, input_shape=(self.sequence_length, 1), return_sequences=True),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(lstm_units, return_sequences=True),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(lstm_units),
            Dropout(0.3),
            BatchNormalization(),
            
            Dense(128, activation='relu'),
            Dropout(0.2),
            
            Dense(self.vocabulary_size, activation='softmax')
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        self.model.summary()
        return self.model
    
    def build_simple_model(self, lstm_units: int = 128, learning_rate: float = 0.001) -> Sequential:
        """Build a simpler LSTM model for faster training"""
        print("Building simple LSTM model...")
        
        self.model = Sequential([
            LSTM(lstm_units, input_shape=(self.sequence_length, 1)),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            Dense(self.vocabulary_size, activation='softmax')
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        self.model.summary()
        return self.model
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, 
              batch_size: int = 64, validation_split: float = 0.2) -> dict:
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print(f"\nTraining model...")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def predict_next_note(self, sequence: np.ndarray, temperature: float = 1.0) -> int:
        """Predict the next note given a sequence"""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        sequence = sequence.reshape(1, self.sequence_length, 1)
        predictions = self.model.predict(sequence, verbose=0)[0]
        
        if temperature != 1.0:
            predictions = np.log(predictions + 1e-10) / temperature
            exp_preds = np.exp(predictions)
            predictions = exp_preds / np.sum(exp_preds)
        
        predicted_index = np.random.choice(len(predictions), p=predictions)
        return predicted_index
    
    def save_model(self, filepath: str) -> None:
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model_from_file(self, filepath: str) -> None:
        """Load model from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")
def create_model(vocabulary_size: int, sequence_length: int = 50, 
                simple: bool = False, lstm_units: int = 256) -> MusicLSTM:
    """Convenience function to create and build a model"""
    music_model = MusicLSTM(vocabulary_size, sequence_length)
    
    if simple:
        music_model.build_simple_model(lstm_units=lstm_units)
    else:
        music_model.build_model(lstm_units=lstm_units)
    
    return music_model