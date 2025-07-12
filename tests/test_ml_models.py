"""
Comprehensive test suite for O-RAN ML models
Author: N. Sachin Deshik
GitHub: sachin-deshik-10
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import the models to test
from ml_models.oran_ml_models import (
    ORANAnomalyDetector,
    ORANThreatPredictor,
    ORANPerformanceOptimizer,
    ORANSecurityScorer,
    ORANMLPipeline
)

class TestORANAnomalyDetector:
    """Test suite for ORANAnomalyDetector"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.detector = ORANAnomalyDetector()
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'latency': np.random.normal(10, 2, 100),
            'throughput': np.random.normal(1000, 100, 100),
            'packet_loss': np.random.uniform(0, 5, 100),
            'cpu_usage': np.random.normal(50, 10, 100),
            'memory_usage': np.random.normal(60, 15, 100),
            'network_usage': np.random.normal(70, 20, 100)
        })
    
    def test_init(self):
        """Test initialization"""
        assert self.detector.model_type == "hybrid"
        assert self.detector.models == {}
        assert self.detector.scalers == {}
        assert not self.detector.is_trained
    
    def test_prepare_data(self):
        """Test data preparation"""
        X, timestamps = self.detector.prepare_data(self.sample_data)
        
        assert X.shape[0] == len(self.sample_data)
        assert X.shape[1] == 9  # 6 features + 3 time features
        assert timestamps.shape[0] == len(self.sample_data)
        assert X.dtype == np.float64
    
    def test_prepare_data_missing_columns(self):
        """Test data preparation with missing columns"""
        minimal_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'latency': np.random.normal(10, 2, 10),
            'throughput': np.random.normal(1000, 100, 10)
        })
        
        X, timestamps = self.detector.prepare_data(minimal_data)
        assert X.shape[0] == 10
        assert X.shape[1] == 5  # 2 features + 3 time features
    
    def test_train_isolation_forest(self):
        """Test Isolation Forest training"""
        X, _ = self.detector.prepare_data(self.sample_data)
        self.detector.train_isolation_forest(X)
        
        assert 'isolation_forest' in self.detector.models
        assert 'isolation_forest' in self.detector.scalers
        assert hasattr(self.detector.models['isolation_forest'], 'predict')
    
    @patch('tensorflow.keras.Sequential')
    def test_train_lstm_autoencoder(self, mock_sequential):
        """Test LSTM Autoencoder training"""
        # Mock TensorFlow model
        mock_model = Mock()
        mock_model.fit.return_value = Mock()
        mock_sequential.return_value = mock_model
        
        X, _ = self.detector.prepare_data(self.sample_data)
        self.detector.train_lstm_autoencoder(X)
        
        assert 'lstm_autoencoder' in self.detector.models
        mock_model.fit.assert_called_once()
    
    def test_detect_anomalies_without_training(self):
        """Test anomaly detection without training"""
        X, _ = self.detector.prepare_data(self.sample_data)
        results = self.detector.detect_anomalies(X)
        
        assert isinstance(results, dict)
        assert len(results) == 0  # No models trained
    
    def test_detect_anomalies_with_isolation_forest(self):
        """Test anomaly detection with trained Isolation Forest"""
        X, _ = self.detector.prepare_data(self.sample_data)
        self.detector.train_isolation_forest(X)
        
        results = self.detector.detect_anomalies(X)
        
        assert 'isolation_forest' in results
        assert len(results['isolation_forest']) == len(X)
        assert all(val in [0, 1] for val in results['isolation_forest'])

class TestORANThreatPredictor:
    """Test suite for ORANThreatPredictor"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.predictor = ORANThreatPredictor()
        self.sample_data = pd.DataFrame({
            'packet_rate': np.random.normal(1000, 100, 100),
            'connection_count': np.random.poisson(50, 100),
            'bandwidth_usage': np.random.normal(70, 10, 100),
            'failed_logins': np.random.poisson(5, 100),
            'privilege_escalations': np.random.poisson(1, 100),
            'cpu_spikes': np.random.poisson(3, 100),
            'memory_anomalies': np.random.poisson(2, 100)
        })
        
        # Create mock labels
        self.sample_labels = np.random.randint(0, 2, 100)
    
    def test_init(self):
        """Test initialization"""
        assert self.predictor.threat_classifier is None
        assert self.predictor.feature_scaler is None
        assert not self.predictor.is_trained
    
    def test_extract_threat_features(self):
        """Test feature extraction"""
        features = self.predictor.extract_threat_features(self.sample_data)
        
        assert features.shape[0] == len(self.sample_data)
        assert features.shape[1] == 7  # Number of features
        assert features.dtype == np.float64
    
    def test_train_threat_classifier(self):
        """Test threat classifier training"""
        X = self.predictor.extract_threat_features(self.sample_data)
        self.predictor.train_threat_classifier(X, self.sample_labels)
        
        assert self.predictor.is_trained
        assert self.predictor.threat_classifier is not None
        assert self.predictor.feature_scaler is not None
    
    def test_predict_threats_without_training(self):
        """Test threat prediction without training"""
        X = self.predictor.extract_threat_features(self.sample_data)
        
        with pytest.raises(ValueError, match="Model not trained yet"):
            self.predictor.predict_threats(X)
    
    def test_predict_threats_with_training(self):
        """Test threat prediction with trained model"""
        X = self.predictor.extract_threat_features(self.sample_data)
        self.predictor.train_threat_classifier(X, self.sample_labels)
        
        predictions, probabilities = self.predictor.predict_threats(X)
        
        assert len(predictions) == len(X)
        assert probabilities.shape[0] == len(X)
        assert probabilities.shape[1] == 2  # Binary classification
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_get_feature_importance(self):
        """Test feature importance extraction"""
        X = self.predictor.extract_threat_features(self.sample_data)
        self.predictor.train_threat_classifier(X, self.sample_labels)
        
        importance = self.predictor.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == X.shape[1]
        assert all(isinstance(val, float) for val in importance.values())

class TestORANPerformanceOptimizer:
    """Test suite for ORANPerformanceOptimizer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = ORANPerformanceOptimizer()
        self.sample_config = {
            'bandwidth_allocation': 100,
            'latency_threshold': 10,
            'throughput_target': 1000,
            'cpu_allocation': 50,
            'memory_allocation': 60,
            'storage_allocation': 70
        }
        
        # Create sample training data
        self.sample_X = np.random.rand(100, 6)
        self.sample_y = np.random.rand(100)
    
    def test_init(self):
        """Test initialization"""
        assert self.optimizer.optimization_model is None
        assert self.optimizer.performance_predictor is None
        assert not self.optimizer.is_trained
    
    def test_train_performance_predictor(self):
        """Test performance predictor training"""
        self.optimizer.train_performance_predictor(self.sample_X, self.sample_y)
        
        assert self.optimizer.is_trained
        assert self.optimizer.performance_predictor is not None
    
    def test_optimize_configuration_without_training(self):
        """Test configuration optimization without training"""
        with pytest.raises(ValueError, match="Model not trained yet"):
            self.optimizer.optimize_configuration(self.sample_config)
    
    def test_optimize_configuration_with_training(self):
        """Test configuration optimization with trained model"""
        self.optimizer.train_performance_predictor(self.sample_X, self.sample_y)
        
        optimized_config = self.optimizer.optimize_configuration(self.sample_config)
        
        assert isinstance(optimized_config, dict)
        assert len(optimized_config) >= len(self.sample_config)
        assert all(key in optimized_config for key in self.sample_config.keys())
    
    def test_config_to_vector(self):
        """Test configuration to vector conversion"""
        vector = self.optimizer._config_to_vector(self.sample_config)
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 6  # Number of configuration parameters
    
    def test_suggest_optimizations(self):
        """Test optimization suggestions"""
        performance_score = 0.5  # Below threshold
        optimized_config = self.optimizer._suggest_optimizations(
            self.sample_config, performance_score
        )
        
        assert isinstance(optimized_config, dict)
        assert optimized_config['bandwidth_allocation'] > self.sample_config['bandwidth_allocation']
        assert optimized_config['cpu_allocation'] > self.sample_config['cpu_allocation']

class TestORANSecurityScorer:
    """Test suite for ORANSecurityScorer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.scorer = ORANSecurityScorer()
        self.sample_metrics = {
            'vulnerability_count': 5,
            'avg_severity_score': 3.2,
            'active_threats': 2,
            'max_threat_severity': 7.5,
            'compliance_percentage': 85,
            'anomaly_count': 3,
            'avg_anomaly_severity': 4.1,
            'configuration_security_score': 0.7
        }
    
    def test_init(self):
        """Test initialization"""
        assert isinstance(self.scorer.weights, dict)
        assert len(self.scorer.weights) == 5
        assert abs(sum(self.scorer.weights.values()) - 1.0) < 0.001
    
    def test_calculate_security_score(self):
        """Test security score calculation"""
        result = self.scorer.calculate_security_score(self.sample_metrics)
        
        assert isinstance(result, dict)
        assert 'overall_score' in result
        assert 'component_scores' in result
        assert 'risk_level' in result
        assert 'recommendations' in result
        
        assert 0 <= result['overall_score'] <= 1
        assert result['risk_level'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        assert isinstance(result['recommendations'], list)
    
    def test_assess_vulnerabilities(self):
        """Test vulnerability assessment"""
        score = self.scorer._assess_vulnerabilities(self.sample_metrics)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_assess_threats(self):
        """Test threat assessment"""
        score = self.scorer._assess_threats(self.sample_metrics)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_assess_compliance(self):
        """Test compliance assessment"""
        score = self.scorer._assess_compliance(self.sample_metrics)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert score == 0.85  # 85% compliance
    
    def test_determine_risk_level(self):
        """Test risk level determination"""
        assert self.scorer._determine_risk_level(0.9) == "LOW"
        assert self.scorer._determine_risk_level(0.7) == "MEDIUM"
        assert self.scorer._determine_risk_level(0.5) == "HIGH"
        assert self.scorer._determine_risk_level(0.3) == "CRITICAL"
    
    def test_generate_recommendations(self):
        """Test recommendation generation"""
        component_scores = {
            'vulnerability_score': 0.8,
            'threat_level': 0.5,  # Below threshold
            'compliance_score': 0.9,
            'anomaly_score': 0.4,  # Below threshold
            'configuration_score': 0.7
        }
        
        recommendations = self.scorer._generate_recommendations(component_scores)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) == 2  # Two scores below threshold
        assert any('threat level' in rec for rec in recommendations)
        assert any('anomaly score' in rec for rec in recommendations)

class TestORANMLPipeline:
    """Test suite for ORANMLPipeline"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.pipeline = ORANMLPipeline()
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'latency': np.random.normal(10, 2, 100),
            'throughput': np.random.normal(1000, 100, 100),
            'packet_loss': np.random.uniform(0, 5, 100),
            'cpu_usage': np.random.normal(50, 10, 100),
            'memory_usage': np.random.normal(60, 15, 100),
            'network_usage': np.random.normal(70, 20, 100),
            'packet_rate': np.random.normal(1000, 100, 100),
            'connection_count': np.random.poisson(50, 100),
            'bandwidth_usage': np.random.normal(70, 10, 100),
            'failed_logins': np.random.poisson(5, 100),
            'cpu_spikes': np.random.poisson(3, 100),
            'vulnerability_count': np.random.poisson(2, 100),
            'compliance_score': np.random.uniform(0.7, 1.0, 100)
        })
    
    def test_init(self):
        """Test initialization"""
        assert isinstance(self.pipeline.anomaly_detector, ORANAnomalyDetector)
        assert isinstance(self.pipeline.threat_predictor, ORANThreatPredictor)
        assert isinstance(self.pipeline.performance_optimizer, ORANPerformanceOptimizer)
        assert isinstance(self.pipeline.security_scorer, ORANSecurityScorer)
        assert not self.pipeline.pipeline_trained
    
    def test_train_pipeline(self):
        """Test pipeline training"""
        self.pipeline.train_pipeline(self.sample_data)
        
        assert self.pipeline.pipeline_trained
        assert 'isolation_forest' in self.pipeline.anomaly_detector.models
    
    def test_analyze_oran_network_without_training(self):
        """Test network analysis without training"""
        with pytest.raises(ValueError, match="Pipeline not trained yet"):
            self.pipeline.analyze_oran_network(self.sample_data)
    
    def test_analyze_oran_network_with_training(self):
        """Test network analysis with trained pipeline"""
        self.pipeline.train_pipeline(self.sample_data)
        
        results = self.pipeline.analyze_oran_network(self.sample_data)
        
        assert isinstance(results, dict)
        assert 'timestamp' in results
        assert 'analysis_type' in results
        assert 'anomalies' in results
        assert 'security_assessment' in results
    
    def test_extract_security_metrics(self):
        """Test security metrics extraction"""
        metrics = self.pipeline._extract_security_metrics(self.sample_data)
        
        assert isinstance(metrics, dict)
        assert 'vulnerability_count' in metrics
        assert 'compliance_percentage' in metrics
    
    def test_extract_configuration(self):
        """Test configuration extraction"""
        config = self.pipeline._extract_configuration(self.sample_data)
        
        assert isinstance(config, dict)
        assert 'cpu_allocation' in config
        assert 'memory_allocation' in config
    
    def test_save_and_load_models(self):
        """Test model saving and loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Train pipeline first
            self.pipeline.train_pipeline(self.sample_data)
            
            # Save models
            self.pipeline.save_models(temp_dir)
            
            # Check if files were created
            assert len(os.listdir(temp_dir)) > 0
            
            # Create new pipeline and load models
            new_pipeline = ORANMLPipeline()
            new_pipeline.load_models(temp_dir)
            
            # Verify models are loaded
            assert new_pipeline.threat_predictor.is_trained

class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    def setup_method(self):
        """Set up integration test fixtures"""
        self.pipeline = ORANMLPipeline()
        
        # Create comprehensive test data
        np.random.seed(42)  # For reproducibility
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='1min'),
            'latency': np.concatenate([
                np.random.normal(10, 2, 180),  # Normal traffic
                np.random.normal(50, 10, 20)   # Anomalous traffic
            ]),
            'throughput': np.concatenate([
                np.random.normal(1000, 100, 180),
                np.random.normal(200, 50, 20)
            ]),
            'packet_loss': np.concatenate([
                np.random.uniform(0, 2, 180),
                np.random.uniform(10, 20, 20)
            ]),
            'cpu_usage': np.random.normal(50, 10, 200),
            'memory_usage': np.random.normal(60, 15, 200),
            'network_usage': np.random.normal(70, 20, 200),
            'packet_rate': np.random.normal(1000, 100, 200),
            'connection_count': np.random.poisson(50, 200),
            'bandwidth_usage': np.random.normal(70, 10, 200),
            'failed_logins': np.random.poisson(5, 200),
            'cpu_spikes': np.random.poisson(3, 200),
            'vulnerability_count': np.random.poisson(2, 200),
            'compliance_score': np.random.uniform(0.7, 1.0, 200)
        })
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        # Train the pipeline
        self.pipeline.train_pipeline(self.test_data)
        
        # Analyze the network
        results = self.pipeline.analyze_oran_network(self.test_data)
        
        # Verify results structure
        assert isinstance(results, dict)
        assert 'timestamp' in results
        assert 'anomalies' in results
        assert 'security_assessment' in results
        
        # Verify anomaly detection found the injected anomalies
        anomalies = results['anomalies']['isolation_forest']
        anomaly_rate = np.mean(anomalies)
        assert anomaly_rate > 0.05  # Should detect some anomalies
    
    def test_performance_under_load(self):
        """Test pipeline performance with large dataset"""
        # Create larger dataset
        large_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10000, freq='1min'),
            'latency': np.random.normal(10, 2, 10000),
            'throughput': np.random.normal(1000, 100, 10000),
            'packet_loss': np.random.uniform(0, 5, 10000),
            'cpu_usage': np.random.normal(50, 10, 10000),
            'memory_usage': np.random.normal(60, 15, 10000),
            'network_usage': np.random.normal(70, 20, 10000),
            'packet_rate': np.random.normal(1000, 100, 10000),
            'connection_count': np.random.poisson(50, 10000),
            'bandwidth_usage': np.random.normal(70, 10, 10000),
            'failed_logins': np.random.poisson(5, 10000),
            'cpu_spikes': np.random.poisson(3, 10000),
            'vulnerability_count': np.random.poisson(2, 10000),
            'compliance_score': np.random.uniform(0.7, 1.0, 10000)
        })
        
        import time
        start_time = time.time()
        
        # Train with subset
        self.pipeline.train_pipeline(large_data.sample(1000))
        
        # Analyze full dataset
        results = self.pipeline.analyze_oran_network(large_data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 60  # 60 seconds threshold
        assert isinstance(results, dict)
    
    def test_data_quality_handling(self):
        """Test handling of poor quality data"""
        # Create data with missing values and outliers
        poor_data = self.test_data.copy()
        
        # Introduce NaN values
        poor_data.loc[10:20, 'latency'] = np.nan
        poor_data.loc[30:40, 'throughput'] = np.nan
        
        # Introduce extreme outliers
        poor_data.loc[50:60, 'latency'] = 10000  # Extreme latency
        poor_data.loc[70:80, 'cpu_usage'] = 200  # Impossible CPU usage
        
        # Pipeline should handle this gracefully
        try:
            self.pipeline.train_pipeline(poor_data.dropna())
            results = self.pipeline.analyze_oran_network(poor_data.dropna())
            assert isinstance(results, dict)
        except Exception as e:
            pytest.fail(f"Pipeline failed to handle poor quality data: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
