#!/usr/bin/env python3
"""
Model and API Testing Script
Author: N. Sachin Deshik
GitHub: sachin-deshik-10
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import requests
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class TestMetrics:
    """Test metrics data class"""
    name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time: float
    memory_usage: float
    status: str


class ModelAPITester:
    """ML Model and API Testing Framework"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.test_results: List[TestMetrics] = []
        self.api_base_url = "http://localhost:8000"
        self.test_data = None
        
    def generate_test_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic test data"""
        print("📊 Generating test data...")
        
        np.random.seed(42)
        
        # Generate realistic O-RAN network metrics
        timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='1min')
        
        test_data = pd.DataFrame({
            'timestamp': timestamps,
            'latency': np.random.normal(10, 2, n_samples),
            'throughput': np.random.normal(1000, 100, n_samples),
            'packet_loss': np.random.uniform(0, 5, n_samples),
            'cpu_usage': np.random.normal(50, 10, n_samples),
            'memory_usage': np.random.normal(60, 15, n_samples),
            'network_usage': np.random.normal(70, 20, n_samples),
            'security_events': np.random.poisson(3, n_samples),
            'vulnerability_score': np.random.uniform(1, 10, n_samples),
            'threat_level': np.random.uniform(0, 1, n_samples),
            'compliance_score': np.random.uniform(0.7, 1.0, n_samples),
            'rrc_connections': np.random.poisson(100, n_samples),
            'handover_success_rate': np.random.uniform(0.9, 0.99, n_samples),
            'bearer_setup_time': np.random.normal(150, 30, n_samples),
            'e2_message_count': np.random.poisson(50, n_samples),
            'xapp_response_time': np.random.normal(25, 5, n_samples),
            'ric_load': np.random.uniform(0.2, 0.8, n_samples),
            'interference_level': np.random.uniform(0, 1, n_samples),
            'resource_utilization': np.random.uniform(0.3, 0.9, n_samples),
            'qos_satisfaction': np.random.uniform(0.8, 1.0, n_samples)
        })
        
        # Add some anomalies
        anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        test_data.loc[anomaly_indices, 'latency'] *= 5
        test_data.loc[anomaly_indices, 'packet_loss'] *= 10
        test_data.loc[anomaly_indices, 'threat_level'] = 0.9
        
        # Add labels for testing
        test_data['is_anomaly'] = False
        test_data.loc[anomaly_indices, 'is_anomaly'] = True
        
        test_data['threat_category'] = np.random.choice(['normal', 'suspicious', 'malicious'], 
                                                      size=n_samples, 
                                                      p=[0.7, 0.2, 0.1])
        
        self.test_data = test_data
        print(f"✅ Generated {n_samples} test samples")
        return test_data
    
    def test_anomaly_detector(self) -> TestMetrics:
        """Test anomaly detection model"""
        print("🔍 Testing Anomaly Detection Model...")
        
        try:
            # Import and test the model
            sys.path.append(str(self.project_root))
            from ml_models.oran_ml_models import ORANAnomalyDetector
            
            start_time = time.time()
            
            # Initialize model
            detector = ORANAnomalyDetector()
            
            # Prepare features
            features = self.test_data[['latency', 'throughput', 'packet_loss', 'cpu_usage', 
                                    'memory_usage', 'security_events', 'vulnerability_score']]
            
            # Train model
            detector.train(features)
            
            # Test predictions
            predictions = detector.predict(features)
            probabilities = detector.predict_proba(features)
            
            # Calculate metrics
            true_labels = self.test_data['is_anomaly'].values
            
            # Binary classification metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, average='weighted')
            recall = recall_score(true_labels, predictions, average='weighted')
            f1 = f1_score(true_labels, predictions, average='weighted')
            
            inference_time = (time.time() - start_time) / len(features) * 1000  # ms per sample
            
            # Memory usage (approximate)
            memory_usage = sys.getsizeof(detector) / 1024 / 1024  # MB
            
            status = "PASSED" if accuracy > 0.8 else "FAILED"
            
            return TestMetrics(
                name="Anomaly Detection",
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                inference_time=inference_time,
                memory_usage=memory_usage,
                status=status
            )
            
        except Exception as e:
            print(f"❌ Error testing anomaly detector: {e}")
            return TestMetrics(
                name="Anomaly Detection",
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                inference_time=0.0,
                memory_usage=0.0,
                status="FAILED"
            )
    
    def test_threat_predictor(self) -> TestMetrics:
        """Test threat prediction model"""
        print("🎯 Testing Threat Prediction Model...")
        
        try:
            from ml_models.oran_ml_models import ORANThreatPredictor
            
            start_time = time.time()
            
            # Initialize model
            predictor = ORANThreatPredictor()
            
            # Prepare features and labels
            features = self.test_data[['latency', 'throughput', 'packet_loss', 'cpu_usage', 
                                    'memory_usage', 'security_events', 'vulnerability_score',
                                    'threat_level']]
            labels = self.test_data['threat_category']
            
            # Train model
            predictor.train(features, labels)
            
            # Test predictions
            predictions = predictor.predict(features)
            probabilities = predictor.predict_proba(features)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, average='weighted')
            recall = recall_score(labels, predictions, average='weighted')
            f1 = f1_score(labels, predictions, average='weighted')
            
            inference_time = (time.time() - start_time) / len(features) * 1000  # ms per sample
            memory_usage = sys.getsizeof(predictor) / 1024 / 1024  # MB
            
            status = "PASSED" if accuracy > 0.7 else "FAILED"
            
            return TestMetrics(
                name="Threat Prediction",
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                inference_time=inference_time,
                memory_usage=memory_usage,
                status=status
            )
            
        except Exception as e:
            print(f"❌ Error testing threat predictor: {e}")
            return TestMetrics(
                name="Threat Prediction",
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                inference_time=0.0,
                memory_usage=0.0,
                status="FAILED"
            )
    
    def test_performance_optimizer(self) -> TestMetrics:
        """Test performance optimization model"""
        print("⚡ Testing Performance Optimization Model...")
        
        try:
            from ml_models.oran_ml_models import ORANPerformanceOptimizer
            
            start_time = time.time()
            
            # Initialize model
            optimizer = ORANPerformanceOptimizer()
            
            # Prepare features and targets
            features = self.test_data[['latency', 'throughput', 'packet_loss', 'cpu_usage', 
                                    'memory_usage', 'resource_utilization', 'qos_satisfaction']]
            targets = self.test_data['qos_satisfaction']
            
            # Train model
            optimizer.train(features, targets)
            
            # Test predictions
            predictions = optimizer.predict(features)
            optimizations = optimizer.optimize(features.iloc[:10])
            
            # Calculate regression metrics
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            
            mse = mean_squared_error(targets, predictions)
            r2 = r2_score(targets, predictions)
            mae = mean_absolute_error(targets, predictions)
            
            # Convert to classification-like metrics for consistency
            accuracy = r2  # R² as accuracy measure
            precision = 1 - mae  # Inverse of MAE
            recall = 1 - np.sqrt(mse)  # Inverse of RMSE
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            inference_time = (time.time() - start_time) / len(features) * 1000  # ms per sample
            memory_usage = sys.getsizeof(optimizer) / 1024 / 1024  # MB
            
            status = "PASSED" if accuracy > 0.6 else "FAILED"
            
            return TestMetrics(
                name="Performance Optimization",
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                inference_time=inference_time,
                memory_usage=memory_usage,
                status=status
            )
            
        except Exception as e:
            print(f"❌ Error testing performance optimizer: {e}")
            return TestMetrics(
                name="Performance Optimization",
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                inference_time=0.0,
                memory_usage=0.0,
                status="FAILED"
            )
    
    def test_security_scorer(self) -> TestMetrics:
        """Test security scoring model"""
        print("🔒 Testing Security Scoring Model...")
        
        try:
            from ml_models.oran_ml_models import ORANSecurityScorer
            
            start_time = time.time()
            
            # Initialize model
            scorer = ORANSecurityScorer()
            
            # Prepare features
            features = self.test_data[['security_events', 'vulnerability_score', 'threat_level',
                                    'compliance_score', 'packet_loss', 'latency']]
            
            # Train model
            scorer.train(features)
            
            # Test predictions
            security_scores = scorer.predict(features)
            risk_levels = scorer.assess_risk(features)
            
            # Calculate metrics (using security score consistency)
            # For security scoring, we evaluate consistency and reasonable ranges
            score_variance = np.var(security_scores)
            score_mean = np.mean(security_scores)
            
            # Metrics based on score quality
            accuracy = 1.0 - min(score_variance / 100, 1.0)  # Lower variance = higher accuracy
            precision = min(score_mean / 10, 1.0)  # Reasonable score range
            recall = len([s for s in security_scores if 0 <= s <= 10]) / len(security_scores)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            inference_time = (time.time() - start_time) / len(features) * 1000  # ms per sample
            memory_usage = sys.getsizeof(scorer) / 1024 / 1024  # MB
            
            status = "PASSED" if accuracy > 0.7 else "FAILED"
            
            return TestMetrics(
                name="Security Scoring",
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                inference_time=inference_time,
                memory_usage=memory_usage,
                status=status
            )
            
        except Exception as e:
            print(f"❌ Error testing security scorer: {e}")
            return TestMetrics(
                name="Security Scoring",
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                inference_time=0.0,
                memory_usage=0.0,
                status="FAILED"
            )
    
    def test_ml_pipeline(self) -> TestMetrics:
        """Test complete ML pipeline"""
        print("🚀 Testing Complete ML Pipeline...")
        
        try:
            from ml_models.oran_ml_models import ORANMLPipeline
            
            start_time = time.time()
            
            # Initialize pipeline
            pipeline = ORANMLPipeline()
            
            # Prepare data
            features = self.test_data[['latency', 'throughput', 'packet_loss', 'cpu_usage', 
                                    'memory_usage', 'security_events', 'vulnerability_score',
                                    'threat_level', 'compliance_score']]
            
            # Train pipeline
            pipeline.train(features)
            
            # Test comprehensive analysis
            analysis_results = pipeline.analyze(features.iloc[:100])
            
            # Evaluate pipeline performance
            pipeline_accuracy = 1.0 if 'anomaly_predictions' in analysis_results else 0.0
            pipeline_precision = 1.0 if 'threat_predictions' in analysis_results else 0.0
            pipeline_recall = 1.0 if 'security_scores' in analysis_results else 0.0
            pipeline_f1 = 1.0 if 'optimization_suggestions' in analysis_results else 0.0
            
            inference_time = (time.time() - start_time) / 100 * 1000  # ms per sample
            memory_usage = sys.getsizeof(pipeline) / 1024 / 1024  # MB
            
            status = "PASSED" if all([pipeline_accuracy, pipeline_precision, pipeline_recall, pipeline_f1]) else "FAILED"
            
            return TestMetrics(
                name="ML Pipeline",
                accuracy=pipeline_accuracy,
                precision=pipeline_precision,
                recall=pipeline_recall,
                f1_score=pipeline_f1,
                inference_time=inference_time,
                memory_usage=memory_usage,
                status=status
            )
            
        except Exception as e:
            print(f"❌ Error testing ML pipeline: {e}")
            return TestMetrics(
                name="ML Pipeline",
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                inference_time=0.0,
                memory_usage=0.0,
                status="FAILED"
            )
    
    def test_api_health(self) -> bool:
        """Test API health endpoint"""
        print("🏥 Testing API health...")
        
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ API health check failed: {e}")
            return False
    
    def test_api_analyze_endpoint(self) -> TestMetrics:
        """Test API analyze endpoint"""
        print("🔌 Testing API analyze endpoint...")
        
        try:
            # Prepare test data
            test_sample = self.test_data.iloc[0].to_dict()
            
            # Remove non-serializable fields
            test_sample = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                         for k, v in test_sample.items() 
                         if k not in ['timestamp', 'is_anomaly', 'threat_category']}
            
            start_time = time.time()
            
            response = requests.post(
                f"{self.api_base_url}/api/v1/analyze",
                json={"data": test_sample},
                timeout=30
            )
            
            inference_time = (time.time() - start_time) * 1000  # ms
            
            if response.status_code == 200:
                result = response.json()
                
                # Check if response contains expected fields
                expected_fields = ['anomaly_score', 'threat_level', 'security_score']
                accuracy = sum(1 for field in expected_fields if field in result) / len(expected_fields)
                precision = 1.0 if 'confidence' in result else 0.8
                recall = 1.0 if 'recommendations' in result else 0.8
                f1 = 2 * (precision * recall) / (precision + recall)
                
                status = "PASSED" if accuracy > 0.5 else "FAILED"
                
                return TestMetrics(
                    name="API Analyze",
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    inference_time=inference_time,
                    memory_usage=0.0,  # Not applicable for API
                    status=status
                )
            else:
                return TestMetrics(
                    name="API Analyze",
                    accuracy=0.0,
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    inference_time=inference_time,
                    memory_usage=0.0,
                    status="FAILED"
                )
                
        except Exception as e:
            print(f"❌ Error testing API analyze endpoint: {e}")
            return TestMetrics(
                name="API Analyze",
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                inference_time=0.0,
                memory_usage=0.0,
                status="FAILED"
            )
    
    def run_all_tests(self) -> List[TestMetrics]:
        """Run all model and API tests"""
        print("🚀 Running Model and API Tests")
        print("=" * 60)
        
        # Generate test data
        self.generate_test_data()
        
        # Test models
        model_tests = [
            self.test_anomaly_detector,
            self.test_threat_predictor,
            self.test_performance_optimizer,
            self.test_security_scorer,
            self.test_ml_pipeline
        ]
        
        # Test API if available
        api_tests = []
        if self.test_api_health():
            print("✅ API is available")
            api_tests = [
                self.test_api_analyze_endpoint
            ]
        else:
            print("⚠️  API is not available, skipping API tests")
        
        # Run all tests
        all_tests = model_tests + api_tests
        results = []
        
        for test_func in all_tests:
            result = test_func()
            results.append(result)
            self.test_results.append(result)
            
            # Print result
            status_emoji = {
                "PASSED": "✅",
                "FAILED": "❌",
                "WARNING": "⚠️",
                "SKIPPED": "⏭️"
            }
            
            print(f"{status_emoji.get(result.status, '❓')} {result.name}: {result.status}")
            print(f"   Accuracy: {result.accuracy:.3f}")
            print(f"   Precision: {result.precision:.3f}")
            print(f"   Recall: {result.recall:.3f}")
            print(f"   F1-Score: {result.f1_score:.3f}")
            print(f"   Inference Time: {result.inference_time:.2f}ms")
            if result.memory_usage > 0:
                print(f"   Memory Usage: {result.memory_usage:.2f}MB")
            print()
        
        return results
    
    def generate_performance_report(self):
        """Generate performance visualization report"""
        print("📊 Generating performance report...")
        
        # Create output directory
        output_dir = self.project_root / "output" / "performance_reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance metrics plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        models = [r.name for r in self.test_results]
        accuracies = [r.accuracy for r in self.test_results]
        
        axes[0, 0].bar(models, accuracies)
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Inference time comparison
        inference_times = [r.inference_time for r in self.test_results]
        axes[0, 1].bar(models, inference_times)
        axes[0, 1].set_title('Inference Time Comparison')
        axes[0, 1].set_ylabel('Time (ms)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1-Score comparison
        f1_scores = [r.f1_score for r in self.test_results]
        axes[1, 0].bar(models, f1_scores)
        axes[1, 0].set_title('F1-Score Comparison')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Memory usage comparison
        memory_usages = [r.memory_usage for r in self.test_results if r.memory_usage > 0]
        memory_models = [r.name for r in self.test_results if r.memory_usage > 0]
        
        if memory_usages:
            axes[1, 1].bar(memory_models, memory_usages)
            axes[1, 1].set_title('Memory Usage Comparison')
            axes[1, 1].set_ylabel('Memory (MB)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate detailed report
        report_data = {
            "test_summary": {
                "total_tests": len(self.test_results),
                "passed": len([r for r in self.test_results if r.status == "PASSED"]),
                "failed": len([r for r in self.test_results if r.status == "FAILED"]),
                "average_accuracy": np.mean([r.accuracy for r in self.test_results]),
                "average_inference_time": np.mean([r.inference_time for r in self.test_results]),
                "total_memory_usage": sum([r.memory_usage for r in self.test_results])
            },
            "individual_results": [
                {
                    "name": r.name,
                    "accuracy": r.accuracy,
                    "precision": r.precision,
                    "recall": r.recall,
                    "f1_score": r.f1_score,
                    "inference_time": r.inference_time,
                    "memory_usage": r.memory_usage,
                    "status": r.status
                }
                for r in self.test_results
            ]
        }
        
        # Save JSON report
        with open(output_dir / "performance_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"✅ Performance report saved to {output_dir}")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("🎯 MODEL & API TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == "PASSED"])
        failed_tests = len([r for r in self.test_results if r.status == "FAILED"])
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📊 Success Rate: {(passed_tests / total_tests * 100):.1f}%")
        
        if self.test_results:
            avg_accuracy = np.mean([r.accuracy for r in self.test_results])
            avg_inference_time = np.mean([r.inference_time for r in self.test_results])
            total_memory = sum([r.memory_usage for r in self.test_results])
            
            print(f"📈 Average Accuracy: {avg_accuracy:.3f}")
            print(f"⚡ Average Inference Time: {avg_inference_time:.2f}ms")
            print(f"💾 Total Memory Usage: {total_memory:.2f}MB")
        
        # Performance targets
        print("\n🎯 Performance Targets:")
        print("• Accuracy: >90% ✅" if avg_accuracy > 0.9 else "• Accuracy: >90% ❌")
        print("• Inference Time: <100ms ✅" if avg_inference_time < 100 else "• Inference Time: <100ms ❌")
        print("• Memory Usage: <2GB ✅" if total_memory < 2048 else "• Memory Usage: <2GB ❌")
        
        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if result.status == "FAILED":
                    print(f"  • {result.name}")
        
        print("\n" + "=" * 60)


def main():
    """Main function"""
    print("🚀 O-RAN Security Test Case Generator")
    print("Model and API Testing Script")
    print("Author: N. Sachin Deshik")
    print("=" * 60)
    
    # Initialize tester
    tester = ModelAPITester()
    
    # Run tests
    results = tester.run_all_tests()
    
    # Generate performance report
    tester.generate_performance_report()
    
    # Print summary
    tester.print_summary()
    
    # Exit with appropriate code
    failed_count = len([r for r in results if r.status == "FAILED"])
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
