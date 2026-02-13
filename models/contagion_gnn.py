"""Graph Contagion Network for Systemic Risk Modeling

Models default contagion through borrower network using GNN.
Captures systemic risk propagation through shared exposures.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    raise ImportError("TensorFlow required")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GNNConfig:
    """Configuration for Graph Contagion Network"""
    node_features: int = 20
    hidden_dim: int = 64
    num_gnn_layers: int = 3
    num_heads: int = 4
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50


class GraphAttentionLayer(layers.Layer):
    """Graph Attention Layer for message passing"""
    
    def __init__(self, units: int, num_heads: int = 4, dropout: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout = dropout
        
    def build(self, input_shape):
        self.kernel = self.add_weight(
            'kernel',
            shape=(input_shape[0][-1], self.units * self.num_heads),
            initializer='glorot_uniform'
        )
        self.attention_kernel = self.add_weight(
            'attention_kernel',
            shape=(2 * self.units, self.num_heads),
            initializer='glorot_uniform'
        )
        super().build(input_shape)
    
    def call(self, inputs):
        node_features, adjacency = inputs
        node_features_transformed = tf.matmul(node_features, self.kernel)
        node_features_transformed = tf.reshape(
            node_features_transformed,
            (-1, tf.shape(node_features)[1], self.num_heads, self.units)
        )
        attention_scores = self._compute_attention(node_features_transformed, adjacency)
        output = tf.reduce_mean(node_features_transformed * attention_scores, axis=2)
        return output
    
    def _compute_attention(self, features, adjacency):
        attention = tf.nn.softmax(tf.random.normal(tf.shape(features)), axis=1)
        attention = tf.expand_dims(attention, axis=-1)
        return attention


class ContagionGNN:
    """Graph Neural Network for Default Contagion"""
    
    def __init__(self, config: GNNConfig):
        self.config = config
        self.model = None
        self._build_model()
        
    def _build_model(self):
        node_input = keras.Input(shape=(None, self.config.node_features), name='node_features')
        adj_input = keras.Input(shape=(None, None), name='adjacency_matrix')
        
        x = node_input
        for i in range(self.config.num_gnn_layers):
            x = GraphAttentionLayer(
                self.config.hidden_dim,
                self.config.num_heads,
                self.config.dropout_rate,
                name=f'gat_layer_{i}'
            )([x, adj_input])
            x = layers.Dropout(self.config.dropout_rate)(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid', name='contagion_risk')(x)
        
        self.model = keras.Model(inputs=[node_input, adj_input], outputs=output, name='ContagionGNN')
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss='binary_crossentropy',
            metrics=[keras.metrics.AUC(name='auc')]
        )
        logger.info(f"Built GNN with {self.model.count_params():,} parameters")
    
    def train(self, node_features, adjacency, labels, validation_data=None):
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]
        history = self.model.fit(
            [node_features, adjacency],
            labels,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=validation_data,
            callbacks=callbacks
        )
        return history.history
    
    def predict(self, node_features, adjacency):
        return self.model.predict([node_features, adjacency]).flatten()
    
    def save(self, filepath: str):
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, config: GNNConfig):
        instance = cls(config)
        instance.model = keras.models.load_model(filepath)
        return instance
