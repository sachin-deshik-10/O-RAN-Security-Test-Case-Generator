"""
Advanced ML/DL Models for O-RAN Security Analysis
Author: N. Sachin Deshik
GitHub: sachin-deshik-10
Email: nsachindeshik.ec21@rvce.edu.in
LinkedIn: https://www.linkedin.com/in/sachin-deshik-nayakula-62b93b362
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
import logging
import joblib
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ORANAnomalyDetector:
    """
    Advanced anomaly detection for O-RAN networks using ML/DL techniques
    Combines multiple approaches: Isolation Forest, LSTM, and Transformer models
    """
    
    def __init__(self, model_type: str = "hybrid"):
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series data for anomaly detection"""
        # Feature engineering for O-RAN metrics
        features = []
        
        # Network performance features
        if 'latency' in data.columns:
            features.extend(['latency', 'throughput', 'packet_loss'])
        
        # Security features  
        if 'vulnerability_score' in data.columns:
            features.extend(['vulnerability_score', 'threat_level'])
            
        # Resource utilization features
        if 'cpu_usage' in data.columns:
            features.extend(['cpu_usage', 'memory_usage', 'network_usage'])
            
        X = data[features].values
        
        # Create time-based features
        timestamps = pd.to_datetime(data['timestamp'])
        time_features = np.column_stack([
            timestamps.dt.hour,
            timestamps.dt.day_of_week,
            timestamps.dt.day_of_month
        ])
        
        X = np.column_stack([X, time_features])
        
        return X, timestamps.values
    
    def train_isolation_forest(self, X: np.ndarray):
        """Train Isolation Forest for anomaly detection"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        isolation_forest.fit(X_scaled)
        
        self.models['isolation_forest'] = isolation_forest
        self.scalers['isolation_forest'] = scaler
        
    def train_lstm_autoencoder(self, X: np.ndarray, sequence_length: int = 10):
        """Train LSTM Autoencoder for temporal anomaly detection"""
        # Prepare sequences
        X_seq = []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i+sequence_length])
        X_seq = np.array(X_seq)
        
        # Build LSTM Autoencoder
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, X.shape[1])),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.RepeatVector(sequence_length),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(X.shape[1]))
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train the model
        history = model.fit(
            X_seq, X_seq,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        self.models['lstm_autoencoder'] = model
        
    def detect_anomalies(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Detect anomalies using trained models"""
        results = {}
        
        # Isolation Forest predictions
        if 'isolation_forest' in self.models:
            X_scaled = self.scalers['isolation_forest'].transform(X)
            iso_pred = self.models['isolation_forest'].predict(X_scaled)
            results['isolation_forest'] = (iso_pred == -1).astype(int)
            
        # LSTM Autoencoder predictions
        if 'lstm_autoencoder' in self.models:
            sequence_length = 10
            X_seq = []
            for i in range(len(X) - sequence_length):
                X_seq.append(X[i:i+sequence_length])
            X_seq = np.array(X_seq)
            
            reconstructed = self.models['lstm_autoencoder'].predict(X_seq)
            mse = np.mean(np.square(X_seq - reconstructed), axis=(1, 2))
            threshold = np.percentile(mse, 95)
            
            # Pad with zeros to match original length
            lstm_pred = np.zeros(len(X))
            lstm_pred[sequence_length:] = (mse > threshold).astype(int)
            results['lstm_autoencoder'] = lstm_pred
            
        return results

class ORANThreatPredictor:
    """
    ML-based threat prediction for O-RAN networks
    Uses ensemble methods and deep learning for threat classification
    """
    
    def __init__(self):
        self.threat_classifier = None
        self.feature_scaler = None
        self.threat_embeddings = None
        self.is_trained = False
        
    def extract_threat_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for threat prediction"""
        features = []
        
        # Network behavior features
        if 'packet_rate' in data.columns:
            features.extend([
                'packet_rate', 'connection_count', 'bandwidth_usage',
                'protocol_distribution', 'port_scan_attempts'
            ])
            
        # Security event features
        if 'failed_logins' in data.columns:
            features.extend([
                'failed_logins', 'privilege_escalations', 'unusual_traffic',
                'malware_signatures', 'vulnerability_exploits'
            ])
            
        # System resource features
        if 'cpu_spikes' in data.columns:
            features.extend([
                'cpu_spikes', 'memory_anomalies', 'disk_activity',
                'network_congestion', 'service_failures'
            ])
            
        return data[features].values
    
    def train_threat_classifier(self, X: np.ndarray, y: np.ndarray):
        """Train ensemble threat classifier"""
        # Scale features
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Train Random Forest classifier
        self.threat_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.threat_classifier.fit(X_scaled, y)
        self.is_trained = True
        
    def predict_threats(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict threats and their probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
            
        X_scaled = self.feature_scaler.transform(X)
        predictions = self.threat_classifier.predict(X_scaled)
        probabilities = self.threat_classifier.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for threat prediction"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
            
        feature_names = [f'feature_{i}' for i in range(len(self.threat_classifier.feature_importances_))]
        importance_dict = dict(zip(feature_names, self.threat_classifier.feature_importances_))
        
        return importance_dict

class ORANPerformanceOptimizer:
    """
    AI-driven performance optimization for O-RAN networks
    Uses reinforcement learning and optimization algorithms
    """
    
    def __init__(self):
        self.optimization_model = None
        self.performance_predictor = None
        self.is_trained = False
        
    def train_performance_predictor(self, X: np.ndarray, y: np.ndarray):
        """Train performance prediction model"""
        from sklearn.ensemble import GradientBoostingRegressor
        
        self.performance_predictor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        self.performance_predictor.fit(X, y)
        self.is_trained = True
        
    def optimize_configuration(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize O-RAN configuration using AI"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
            
        # Convert config to feature vector
        config_vector = self._config_to_vector(current_config)
        
        # Predict performance
        predicted_performance = self.performance_predictor.predict([config_vector])[0]
        
        # Optimization suggestions
        optimized_config = self._suggest_optimizations(current_config, predicted_performance)
        
        return optimized_config
    
    def _config_to_vector(self, config: Dict[str, Any]) -> np.ndarray:
        """Convert configuration dictionary to feature vector"""
        # Implementation depends on specific O-RAN configuration parameters
        features = []
        
        # Network parameters
        features.extend([
            config.get('bandwidth_allocation', 100),
            config.get('latency_threshold', 10),
            config.get('throughput_target', 1000)
        ])
        
        # Resource allocation
        features.extend([
            config.get('cpu_allocation', 50),
            config.get('memory_allocation', 60),
            config.get('storage_allocation', 70)
        ])
        
        return np.array(features)
    
    def _suggest_optimizations(self, config: Dict[str, Any], performance: float) -> Dict[str, Any]:
        """Suggest configuration optimizations"""
        optimized_config = config.copy()
        
        # AI-driven optimization logic
        if performance < 0.7:  # Performance threshold
            optimized_config['bandwidth_allocation'] = min(config.get('bandwidth_allocation', 100) * 1.2, 200)
            optimized_config['cpu_allocation'] = min(config.get('cpu_allocation', 50) * 1.1, 100)
            
        return optimized_config

class ORANSecurityScorer:
    """
    Advanced security scoring system for O-RAN networks
    Combines multiple ML techniques for comprehensive security assessment
    """
    
    def __init__(self):
        self.scoring_models = {}
        self.weights = {
            'vulnerability_score': 0.3,
            'threat_level': 0.25,
            'compliance_score': 0.2,
            'anomaly_score': 0.15,
            'configuration_score': 0.1
        }
        
    def calculate_security_score(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Calculate comprehensive security score"""
        component_scores = {}
        
        # Vulnerability assessment
        component_scores['vulnerability_score'] = self._assess_vulnerabilities(metrics)
        
        # Threat level assessment
        component_scores['threat_level'] = self._assess_threats(metrics)
        
        # Compliance assessment
        component_scores['compliance_score'] = self._assess_compliance(metrics)
        
        # Anomaly assessment
        component_scores['anomaly_score'] = self._assess_anomalies(metrics)
        
        # Configuration assessment
        component_scores['configuration_score'] = self._assess_configuration(metrics)
        
        # Calculate weighted overall score
        overall_score = sum(
            component_scores[component] * self.weights[component]
            for component in component_scores
        )
        
        return {
            'overall_score': overall_score,
            'component_scores': component_scores,
            'risk_level': self._determine_risk_level(overall_score),
            'recommendations': self._generate_recommendations(component_scores)
        }
    
    def _assess_vulnerabilities(self, metrics: Dict[str, float]) -> float:
        """Assess vulnerability score"""
        vuln_count = metrics.get('vulnerability_count', 0)
        severity_score = metrics.get('avg_severity_score', 0)
        
        # Normalize and invert (lower vulnerabilities = higher score)
        score = max(0, 1 - (vuln_count * 0.1 + severity_score * 0.1))
        return min(1.0, score)
    
    def _assess_threats(self, metrics: Dict[str, float]) -> float:
        """Assess threat level score"""
        threat_count = metrics.get('active_threats', 0)
        threat_severity = metrics.get('max_threat_severity', 0)
        
        score = max(0, 1 - (threat_count * 0.2 + threat_severity * 0.15))
        return min(1.0, score)
    
    def _assess_compliance(self, metrics: Dict[str, float]) -> float:
        """Assess compliance score"""
        compliance_rate = metrics.get('compliance_percentage', 0) / 100
        return compliance_rate
    
    def _assess_anomalies(self, metrics: Dict[str, float]) -> float:
        """Assess anomaly score"""
        anomaly_count = metrics.get('anomaly_count', 0)
        anomaly_severity = metrics.get('avg_anomaly_severity', 0)
        
        score = max(0, 1 - (anomaly_count * 0.1 + anomaly_severity * 0.1))
        return min(1.0, score)
    
    def _assess_configuration(self, metrics: Dict[str, float]) -> float:
        """Assess configuration security score"""
        config_score = metrics.get('configuration_security_score', 0.5)
        return config_score
    
    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level based on score"""
        if score >= 0.8:
            return "LOW"
        elif score >= 0.6:
            return "MEDIUM"
        elif score >= 0.4:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _generate_recommendations(self, component_scores: Dict[str, float]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        for component, score in component_scores.items():
            if score < 0.6:
                recommendations.append(f"Improve {component.replace('_', ' ')}")
                
        return recommendations

class ORANMLPipeline:
    """
    Complete ML pipeline for O-RAN security and performance analysis
    Integrates all ML models and provides unified interface
    """
    
    def __init__(self):
        self.anomaly_detector = ORANAnomalyDetector()
        self.threat_predictor = ORANThreatPredictor()
        self.performance_optimizer = ORANPerformanceOptimizer()
        self.security_scorer = ORANSecurityScorer()
        self.pipeline_trained = False
        
    def train_pipeline(self, data: pd.DataFrame):
        """Train the complete ML pipeline"""
        logger.info("Training O-RAN ML pipeline...")
        
        # Prepare data for different models
        X_anomaly, timestamps = self.anomaly_detector.prepare_data(data)
        X_threat = self.threat_predictor.extract_threat_features(data)
        
        # Train anomaly detection
        self.anomaly_detector.train_isolation_forest(X_anomaly)
        self.anomaly_detector.train_lstm_autoencoder(X_anomaly)
        
        # Train threat prediction (assuming we have threat labels)
        if 'threat_label' in data.columns:
            y_threat = data['threat_label'].values
            self.threat_predictor.train_threat_classifier(X_threat, y_threat)
            
        # Train performance optimization
        if 'performance_score' in data.columns:
            y_performance = data['performance_score'].values
            self.performance_optimizer.train_performance_predictor(X_anomaly, y_performance)
            
        self.pipeline_trained = True
        logger.info("ML pipeline training completed")
        
    def analyze_oran_network(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive O-RAN network analysis"""
        if not self.pipeline_trained:
            raise ValueError("Pipeline not trained yet")
            
        results = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'comprehensive_oran_analysis'
        }
        
        # Anomaly detection
        X_anomaly, timestamps = self.anomaly_detector.prepare_data(data)
        anomalies = self.anomaly_detector.detect_anomalies(X_anomaly)
        results['anomalies'] = anomalies
        
        # Threat prediction
        X_threat = self.threat_predictor.extract_threat_features(data)
        if self.threat_predictor.is_trained:
            threat_predictions, threat_probabilities = self.threat_predictor.predict_threats(X_threat)
            results['threats'] = {
                'predictions': threat_predictions.tolist(),
                'probabilities': threat_probabilities.tolist()
            }
        
        # Security scoring
        security_metrics = self._extract_security_metrics(data)
        security_score = self.security_scorer.calculate_security_score(security_metrics)
        results['security_assessment'] = security_score
        
        # Performance optimization
        if self.performance_optimizer.is_trained:
            current_config = self._extract_configuration(data)
            optimized_config = self.performance_optimizer.optimize_configuration(current_config)
            results['optimization_suggestions'] = optimized_config
            
        return results
    
    def _extract_security_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract security metrics from data"""
        metrics = {}
        
        # Calculate various security metrics
        if 'vulnerability_count' in data.columns:
            metrics['vulnerability_count'] = data['vulnerability_count'].mean()
            
        if 'threat_level' in data.columns:
            metrics['active_threats'] = data['threat_level'].sum()
            
        if 'compliance_score' in data.columns:
            metrics['compliance_percentage'] = data['compliance_score'].mean() * 100
            
        return metrics
    
    def _extract_configuration(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract configuration parameters from data"""
        config = {}
        
        # Extract relevant configuration parameters
        if 'bandwidth' in data.columns:
            config['bandwidth_allocation'] = data['bandwidth'].mean()
            
        if 'cpu_usage' in data.columns:
            config['cpu_allocation'] = data['cpu_usage'].mean()
            
        if 'memory_usage' in data.columns:
            config['memory_allocation'] = data['memory_usage'].mean()
            
        return config
    
    def save_models(self, model_dir: str):
        """Save trained models"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save scikit-learn models
        if self.threat_predictor.is_trained:
            joblib.dump(self.threat_predictor.threat_classifier, 
                       os.path.join(model_dir, 'threat_classifier.pkl'))
            joblib.dump(self.threat_predictor.feature_scaler, 
                       os.path.join(model_dir, 'threat_scaler.pkl'))
        
        # Save TensorFlow models
        if 'lstm_autoencoder' in self.anomaly_detector.models:
            self.anomaly_detector.models['lstm_autoencoder'].save(
                os.path.join(model_dir, 'lstm_autoencoder.h5'))
                
        logger.info(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir: str):
        """Load pre-trained models"""
        import os
        
        # Load scikit-learn models
        threat_classifier_path = os.path.join(model_dir, 'threat_classifier.pkl')
        if os.path.exists(threat_classifier_path):
            self.threat_predictor.threat_classifier = joblib.load(threat_classifier_path)
            self.threat_predictor.feature_scaler = joblib.load(
                os.path.join(model_dir, 'threat_scaler.pkl'))
            self.threat_predictor.is_trained = True
        
        # Load TensorFlow models
        lstm_path = os.path.join(model_dir, 'lstm_autoencoder.h5')
        if os.path.exists(lstm_path):
            self.anomaly_detector.models['lstm_autoencoder'] = tf.keras.models.load_model(lstm_path)
            
        logger.info(f"Models loaded from {model_dir}")
