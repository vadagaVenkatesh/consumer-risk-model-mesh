"""LSTM Attention Network for Thin-File Cash Flow Modeling

Implements time-aware attention mechanism for PD estimation on borrowers
with limited credit history. Uses temporal cash flow patterns to predict
default probability within Basel IRB framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    raise ImportError("TensorFlow not installed")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LSTMConfig:
    """Configuration for LSTM Attention Network"""
    sequence_length: int = 12
    input_features: int = 15
    lstm_units: int = 128
    attention_units: int = 64
    dense_units: List[int] = None
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 100
    
    def __post_init__(self):
        if self.dense_units is None:
            self.dense_units = [64, 32]


class AttentionLayer(layers.Layer):
    """Bahdanau Attention Mechanism"""
    
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W_h = self.add_weight(
            name='attention_W_h',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W_s = self.add_weight(
            name='attention_W_s',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.v = self.add_weight(
            name='attention_v',
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, hidden_states):
        state_summary = tf.reduce_mean(hidden_states, axis=1, keepdims=True)
        state_summary = tf.tile(state_summary, [1, tf.shape(hidden_states)[1], 1])
        score = tf.nn.tanh(
            tf.matmul(hidden_states, self.W_h) + 
            tf.matmul(state_summary, self.W_s)
        )
        score = tf.matmul(score, self.v)
        attention_weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(hidden_states * attention_weights, axis=1)
        return context, tf.squeeze(attention_weights, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


class AttentionLSTM:
    """LSTM with Attention for Thin-File Borrower PD Estimation"""
    
    def __init__(self, config: LSTMConfig):
        self.config = config
        self.model = None
        self.history = None
        self._build_model()
        
    def _build_model(self):
        inputs = keras.Input(
            shape=(self.config.sequence_length, self.config.input_features),
            name='transaction_sequence'
        )
        lstm_out = layers.Bidirectional(
            layers.LSTM(
                self.config.lstm_units,
                return_sequences=True,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate,
                name='lstm_encoder'
            ),
            name='bidirectional_lstm'
        )(inputs)
        context, attention_weights = AttentionLayer(
            self.config.attention_units,
            name='attention'
        )(lstm_out)
        x = context
        for i, units in enumerate(self.config.dense_units):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.01),
                name=f'dense_{i}'
            )(x)
            x = layers.Dropout(self.config.dropout_rate)(x)
        pd_output = layers.Dense(1, activation='sigmoid', name='pd_probability')(x)
        self.model = keras.Model(inputs=inputs, outputs=[pd_output, attention_weights], name='AttentionLSTM_PD')
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss={'pd_probability': 'binary_crossentropy', 'attention': lambda y_true, y_pred: 0.0},
            metrics={'pd_probability': [keras.metrics.AUC(name='auc'), keras.metrics.BinaryAccuracy(name='accuracy')]}
        )
        logger.info(f"Built LSTM Attention model with {self.model.count_params():,} parameters")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict:
        validation_data = None
        if X_val is not None and y_val is not None:
            dummy_attention_val = np.zeros((len(y_val), self.config.sequence_length))
            validation_data = (X_val, {'pd_probability': y_val, 'attention': dummy_attention_val})
        dummy_attention_train = np.zeros((len(y_train), self.config.sequence_length))
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_pd_probability_auc' if validation_data else 'pd_probability_auc', patience=10, mode='max', restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss' if validation_data else 'loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]
        logger.info("Starting LSTM training...")
        self.history = self.model.fit(X_train, {'pd_probability': y_train, 'attention': dummy_attention_train}, batch_size=self.config.batch_size, epochs=self.config.epochs, validation_data=validation_data, callbacks=callbacks, verbose=1)
        return self.history.history
    
    def predict(self, X: np.ndarray, return_attention: bool = False) -> np.ndarray:
        pd_probs, attention_weights = self.model.predict(X, verbose=0)
        if return_attention:
            return pd_probs.flatten(), attention_weights
        return pd_probs.flatten()
    
    def save(self, filepath: str):
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, config: LSTMConfig):
        instance = cls(config)
        instance.model = keras.models.load_model(filepath, custom_objects={'AttentionLayer': AttentionLayer})
        logger.info(f"Model loaded from {filepath}")
        return instance
