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
    """Graph attention layer (Velickovic et al., 2018) with multi-head averaging.

    Attention coefficients are computed from the transformed node features and
    masked by the adjacency matrix, so a node only attends to its neighbours.
    Message passing is therefore genuinely a function of the graph: change an
    edge and the output changes.
    """

    def __init__(self, units: int, num_heads: int = 4, dropout: float = 0.3,
                 negative_slope: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout = dropout
        self.negative_slope = negative_slope

    def build(self, input_shape):
        feat_dim = input_shape[0][-1]
        self.kernel = self.add_weight(
            name='kernel',
            shape=(feat_dim, self.units * self.num_heads),
            initializer='glorot_uniform',
            trainable=True,
        )
        # Split form of the GAT attention vector a: one half scores the source
        # node, the other the destination. Keeping them separate lets the
        # pairwise scores be built by broadcasting rather than by materialising
        # an N x N concatenation.
        self.att_src = self.add_weight(
            name='att_src', shape=(self.units, self.num_heads),
            initializer='glorot_uniform', trainable=True,
        )
        self.att_dst = self.add_weight(
            name='att_dst', shape=(self.units, self.num_heads),
            initializer='glorot_uniform', trainable=True,
        )
        self.drop = layers.Dropout(self.dropout)
        super().build(input_shape)

    def call(self, inputs, training=None):
        node_features, adjacency = inputs                      # (B,N,F), (B,N,N)

        wh = tf.matmul(node_features, self.kernel)             # (B,N,H*U)
        wh = tf.reshape(wh, (-1, tf.shape(node_features)[1],
                             self.num_heads, self.units))      # (B,N,H,U)

        e_src = tf.einsum('bnhu,uh->bnh', wh, self.att_src)    # (B,N,H)
        e_dst = tf.einsum('bnhu,uh->bnh', wh, self.att_dst)    # (B,N,H)
        e = e_src[:, :, tf.newaxis, :] + e_dst[:, tf.newaxis, :, :]   # (B,N,N,H)
        e = tf.nn.leaky_relu(e, alpha=self.negative_slope)

        # Mask non-edges before the softmax so they receive exactly zero weight.
        mask = tf.expand_dims(adjacency, axis=-1)              # (B,N,N,1)
        e = tf.where(mask > 0, e, tf.fill(tf.shape(e), tf.constant(-1e9, e.dtype)))

        alpha = tf.nn.softmax(e, axis=2)                       # over neighbours j
        alpha = self.drop(alpha, training=training)

        out = tf.einsum('bijh,bjhu->bihu', alpha, wh)          # (B,N,H,U)
        return tf.reduce_mean(out, axis=2)                     # average heads -> (B,N,U)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.units,)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(units=self.units, num_heads=self.num_heads,
                   dropout=self.dropout, negative_slope=self.negative_slope)
        return cfg


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
