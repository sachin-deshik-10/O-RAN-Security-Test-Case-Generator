"""
Deep Learning Models for O-RAN Security Analysis
Author: N. Sachin Deshik
GitHub: sachin-deshik-10
Email: nsachindeshik.ec21@rvce.edu.in
LinkedIn: https://www.linkedin.com/in/sachin-deshik-nayakula-62b93b362
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ORANTransformerModel(nn.Module):
    """
    Transformer-based model for O-RAN network analysis
    Processes sequential network data and identifies patterns
    """
    
    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 6, num_classes: int = 5):
        super(ORANTransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        
        # Transformer expects (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=0)
        
        # Classification
        output = self.classifier(x)
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ORANGANModel:
    """
    Generative Adversarial Network for O-RAN data augmentation
    Generates synthetic O-RAN network data for training
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 100):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
    def _build_generator(self):
        """Build generator network"""
        model = models.Sequential([
            layers.Dense(128, input_dim=self.latent_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            
            layers.Dense(256),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            
            layers.Dense(512),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            
            layers.Dense(self.input_dim, activation='tanh')
        ])
        
        return model
    
    def _build_discriminator(self):
        """Build discriminator network"""
        model = models.Sequential([
            layers.Dense(512, input_dim=self.input_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            layers.Dense(256),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            layers.Dense(128),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def compile_models(self):
        """Compile GAN models"""
        optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        
        # Compile discriminator
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        # Compile combined model
        self.discriminator.trainable = False
        
        gan_input = layers.Input(shape=(self.latent_dim,))
        generated_data = self.generator(gan_input)
        gan_output = self.discriminator(generated_data)
        
        self.combined = models.Model(gan_input, gan_output)
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=optimizer
        )
        
    def train(self, X_train: np.ndarray, epochs: int = 1000, batch_size: int = 32):
        """Train GAN model"""
        # Normalize data
        X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
        
        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_data = X_train[idx]
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_data = self.generator.predict(noise)
            
            d_loss_real = self.discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
            d_loss_fake = self.discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))
            
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}, D loss: {(d_loss_real[0] + d_loss_fake[0])/2:.4f}, G loss: {g_loss:.4f}")
    
    def generate_samples(self, n_samples: int) -> np.ndarray:
        """Generate synthetic O-RAN data samples"""
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        return self.generator.predict(noise)

class ORANRNNModel:
    """
    Recurrent Neural Network for O-RAN time series prediction
    Predicts future network states and performance metrics
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, output_dim: int = 1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.model = self._build_model()
        
    def _build_model(self):
        """Build RNN model"""
        model = models.Sequential([
            layers.LSTM(self.hidden_dim, return_sequences=True, 
                       input_shape=(None, self.input_dim)),
            layers.Dropout(0.2),
            
            layers.LSTM(self.hidden_dim, return_sequences=True),
            layers.Dropout(0.2),
            
            layers.LSTM(self.hidden_dim, return_sequences=False),
            layers.Dropout(0.2),
            
            layers.Dense(self.hidden_dim // 2, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(self.output_dim)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_sequences(self, data: np.ndarray, sequence_length: int = 10, 
                         prediction_steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training"""
        X, y = [], []
        
        for i in range(len(data) - sequence_length - prediction_steps + 1):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length:i+sequence_length+prediction_steps])
            
        return np.array(X), np.array(y)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray, epochs: int = 100):
        """Train RNN model"""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5)
            ]
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)

class ORANAutoEncoder:
    """
    Variational Autoencoder for O-RAN data compression and anomaly detection
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 64):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.vae = self._build_vae()
        
    def _build_encoder(self):
        """Build encoder network"""
        inputs = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        z_mean = layers.Dense(self.latent_dim)(x)
        z_log_var = layers.Dense(self.latent_dim)(x)
        
        return models.Model(inputs, [z_mean, z_log_var])
    
    def _build_decoder(self):
        """Build decoder network"""
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(128, activation='relu')(latent_inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.input_dim, activation='sigmoid')(x)
        
        return models.Model(latent_inputs, outputs)
    
    def _sampling(self, args):
        """Sampling layer for VAE"""
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def _build_vae(self):
        """Build complete VAE"""
        inputs = layers.Input(shape=(self.input_dim,))
        z_mean, z_log_var = self.encoder(inputs)
        z = layers.Lambda(self._sampling)([z_mean, z_log_var])
        outputs = self.decoder(z)
        
        vae = models.Model(inputs, outputs)
        
        # VAE loss
        reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
        reconstruction_loss *= self.input_dim
        
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        
        vae.compile(optimizer='adam')
        
        return vae
    
    def train(self, X_train: np.ndarray, X_val: np.ndarray, epochs: int = 100):
        """Train VAE"""
        history = self.vae.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )
        
        return history
    
    def detect_anomalies(self, X: np.ndarray, threshold_percentile: float = 95) -> np.ndarray:
        """Detect anomalies using reconstruction error"""
        reconstructed = self.vae.predict(X)
        mse = np.mean(np.square(X - reconstructed), axis=1)
        
        threshold = np.percentile(mse, threshold_percentile)
        anomalies = mse > threshold
        
        return anomalies.astype(int)

class ORANGraphNeuralNetwork:
    """
    Graph Neural Network for O-RAN network topology analysis
    Analyzes relationships between network components
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 5):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
    def build_model(self, num_nodes: int, num_edges: int):
        """Build GNN model using TensorFlow"""
        # Node features
        node_features = layers.Input(shape=(num_nodes, self.input_dim))
        
        # Adjacency matrix
        adjacency = layers.Input(shape=(num_nodes, num_nodes))
        
        # Graph convolutional layers
        x = self._graph_conv_layer(node_features, adjacency, self.hidden_dim)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.2)(x)
        
        x = self._graph_conv_layer(x, adjacency, self.hidden_dim)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.2)(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Classification
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model([node_features, adjacency], outputs)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _graph_conv_layer(self, node_features, adjacency, output_dim):
        """Graph convolution layer"""
        # Normalize adjacency matrix
        degree = tf.reduce_sum(adjacency, axis=-1, keepdims=True)
        degree = tf.maximum(degree, 1.0)  # Avoid division by zero
        normalized_adj = adjacency / degree
        
        # Graph convolution: A * X * W
        x = layers.Dense(output_dim)(node_features)
        x = tf.matmul(normalized_adj, x)
        
        return x

class ORANEnsembleModel:
    """
    Ensemble model combining multiple deep learning approaches
    Provides robust predictions through model averaging
    """
    
    def __init__(self, input_dim: int, model_types: List[str] = None):
        self.input_dim = input_dim
        self.model_types = model_types or ['cnn', 'rnn', 'transformer']
        self.models = {}
        self.weights = {}
        
    def build_cnn_model(self, sequence_length: int):
        """Build CNN model for 1D sequence data"""
        model = models.Sequential([
            layers.Reshape((sequence_length, self.input_dim, 1)),
            
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(5, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_ensemble(self, sequence_length: int = 10):
        """Build ensemble of models"""
        if 'cnn' in self.model_types:
            self.models['cnn'] = self.build_cnn_model(sequence_length)
            
        if 'rnn' in self.model_types:
            self.models['rnn'] = ORANRNNModel(
                input_dim=self.input_dim,
                output_dim=5
            ).model
            
        # Initialize equal weights
        self.weights = {model_type: 1.0 / len(self.models) 
                       for model_type in self.models}
        
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray, epochs: int = 100):
        """Train ensemble models"""
        histories = {}
        
        for model_type, model in self.models.items():
            logger.info(f"Training {model_type} model...")
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                ]
            )
            
            histories[model_type] = history
            
        return histories
    
    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        predictions = []
        
        for model_type, model in self.models.items():
            pred = model.predict(X)
            weighted_pred = pred * self.weights[model_type]
            predictions.append(weighted_pred)
            
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def update_weights(self, validation_scores: Dict[str, float]):
        """Update model weights based on validation performance"""
        total_score = sum(validation_scores.values())
        
        for model_type in self.models:
            if model_type in validation_scores:
                self.weights[model_type] = validation_scores[model_type] / total_score
                
        logger.info(f"Updated ensemble weights: {self.weights}")

class ORANModelManager:
    """
    Centralized manager for all O-RAN deep learning models
    Provides unified interface for model training and inference
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.model_performances = {}
        
    def initialize_models(self):
        """Initialize all models"""
        input_dim = self.config.get('input_dim', 50)
        
        # Initialize different model types
        self.models['transformer'] = ORANTransformerModel(input_dim)
        self.models['gan'] = ORANGANModel(input_dim)
        self.models['rnn'] = ORANRNNModel(input_dim)
        self.models['autoencoder'] = ORANAutoEncoder(input_dim)
        self.models['ensemble'] = ORANEnsembleModel(input_dim)
        
        logger.info("All O-RAN models initialized")
        
    def train_all_models(self, train_data: Dict[str, np.ndarray], 
                        val_data: Dict[str, np.ndarray]):
        """Train all models"""
        training_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name} model...")
            
            try:
                if model_name == 'transformer':
                    # Convert to PyTorch tensors and train
                    pass
                    
                elif model_name == 'gan':
                    model.compile_models()
                    model.train(train_data['X'])
                    
                elif model_name == 'rnn':
                    X_train, y_train = model.prepare_sequences(train_data['X'])
                    X_val, y_val = model.prepare_sequences(val_data['X'])
                    history = model.train(X_train, y_train, X_val, y_val)
                    training_results[model_name] = history
                    
                elif model_name == 'autoencoder':
                    history = model.train(train_data['X'], val_data['X'])
                    training_results[model_name] = history
                    
                elif model_name == 'ensemble':
                    model.build_ensemble()
                    histories = model.train_ensemble(
                        train_data['X'], train_data['y'],
                        val_data['X'], val_data['y']
                    )
                    training_results[model_name] = histories
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                
        return training_results
    
    def evaluate_models(self, test_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate all models"""
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'autoencoder':
                    anomalies = model.detect_anomalies(test_data['X'])
                    evaluation_results[model_name] = np.mean(anomalies)
                    
                elif model_name == 'ensemble':
                    predictions = model.predict_ensemble(test_data['X'])
                    # Calculate accuracy or other metrics
                    
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                
        return evaluation_results
    
    def get_model_recommendations(self) -> Dict[str, str]:
        """Get recommendations for model usage"""
        recommendations = {
            'transformer': 'Best for sequential pattern recognition',
            'gan': 'Use for data augmentation and synthetic data generation',
            'rnn': 'Ideal for time series prediction and forecasting',
            'autoencoder': 'Excellent for anomaly detection and compression',
            'ensemble': 'Provides most robust predictions through averaging'
        }
        
        return recommendations
