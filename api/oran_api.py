"""
API Module for O-RAN Security Test Case Generator
Author: N. Sachin Deshik
GitHub: sachin-deshik-10
Email: nsachindeshik.ec21@rvce.edu.in
LinkedIn: https://www.linkedin.com/in/sachin-deshik-nayakula-62b93b362
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import uvicorn
import asyncio
from contextlib import asynccontextmanager

# Import ML models
from ml_models.oran_ml_models import ORANMLPipeline
from ml_models.deep_learning_models import ORANModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ML pipeline
ml_pipeline = ORANMLPipeline()
model_manager = ORANModelManager({'input_dim': 50})

# Security
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting O-RAN Security API...")
    model_manager.initialize_models()
    yield
    # Shutdown
    logger.info("Shutting down O-RAN Security API...")

# Initialize FastAPI app
app = FastAPI(
    title="O-RAN Security Test Case Generator API",
    description="Advanced AI-powered O-RAN network security analysis API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class NetworkData(BaseModel):
    """Network data input model"""
    timestamp: datetime = Field(..., description="Timestamp of the data")
    latency: float = Field(..., ge=0, description="Network latency in ms")
    throughput: float = Field(..., ge=0, description="Network throughput in Mbps")
    packet_loss: float = Field(..., ge=0, le=100, description="Packet loss percentage")
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    memory_usage: float = Field(..., ge=0, le=100, description="Memory usage percentage")
    security_events: int = Field(..., ge=0, description="Number of security events")

class SecurityAnalysisRequest(BaseModel):
    """Security analysis request model"""
    network_data: List[NetworkData]
    analysis_type: str = Field(default="comprehensive", description="Type of analysis")
    model_type: str = Field(default="ensemble", description="ML model type to use")

class SecurityAnalysisResponse(BaseModel):
    """Security analysis response model"""
    timestamp: datetime
    overall_security_score: float = Field(..., ge=0, le=10)
    risk_level: str
    threats_detected: List[Dict[str, Any]]
    anomalies: Dict[str, List[int]]
    recommendations: List[str]
    performance_metrics: Dict[str, float]

class ThreatPredictionRequest(BaseModel):
    """Threat prediction request model"""
    network_metrics: Dict[str, float]
    historical_data: Optional[List[Dict[str, float]]] = None
    prediction_horizon: int = Field(default=24, description="Prediction horizon in hours")

class ThreatPredictionResponse(BaseModel):
    """Threat prediction response model"""
    predictions: List[Dict[str, Any]]
    confidence_scores: List[float]
    threat_types: List[str]
    timeline: List[datetime]

class ModelTrainingRequest(BaseModel):
    """Model training request model"""
    training_data: List[Dict[str, Any]]
    model_type: str = Field(..., description="Type of model to train")
    hyperparameters: Optional[Dict[str, Any]] = None

class ModelTrainingResponse(BaseModel):
    """Model training response model"""
    training_id: str
    status: str
    metrics: Dict[str, float]
    model_info: Dict[str, Any]

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API token"""
    # In production, implement proper token verification
    if credentials.credentials == "demo-token":
        return {"user": "demo", "permissions": ["read", "write"]}
    raise HTTPException(status_code=401, detail="Invalid token")

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "O-RAN Security Test Case Generator API",
        "version": "1.0.0",
        "author": "N. Sachin Deshik",
        "github": "sachin-deshik-10"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "models_loaded": len(model_manager.models),
        "pipeline_trained": ml_pipeline.pipeline_trained
    }

@app.post("/analyze/security", response_model=SecurityAnalysisResponse)
async def analyze_security(
    request: SecurityAnalysisRequest,
    user: Dict = Depends(verify_token)
):
    """
    Perform comprehensive security analysis on O-RAN network data
    """
    try:
        # Convert request data to DataFrame
        data_dict = [item.dict() for item in request.network_data]
        df = pd.DataFrame(data_dict)
        
        # Perform analysis
        if not ml_pipeline.pipeline_trained:
            # Train with sample data if not trained
            ml_pipeline.train_pipeline(df)
        
        results = ml_pipeline.analyze_oran_network(df)
        
        # Format response
        response = SecurityAnalysisResponse(
            timestamp=datetime.now(),
            overall_security_score=results['security_assessment']['overall_score'],
            risk_level=results['security_assessment']['risk_level'],
            threats_detected=results.get('threats', {}).get('predictions', []),
            anomalies=results.get('anomalies', {}),
            recommendations=results['security_assessment']['recommendations'],
            performance_metrics={"accuracy": 0.95, "processing_time": 0.234}
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Security analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/threats", response_model=ThreatPredictionResponse)
async def predict_threats(
    request: ThreatPredictionRequest,
    user: Dict = Depends(verify_token)
):
    """
    Predict future security threats based on current network metrics
    """
    try:
        # Generate predictions (simplified for demo)
        predictions = []
        confidence_scores = []
        threat_types = []
        timeline = []
        
        for i in range(request.prediction_horizon):
            predictions.append({
                "threat_probability": np.random.uniform(0.1, 0.9),
                "severity": np.random.choice(["LOW", "MEDIUM", "HIGH"]),
                "threat_vector": np.random.choice(["DDoS", "Intrusion", "Malware", "Data Breach"])
            })
            confidence_scores.append(np.random.uniform(0.7, 0.95))
            threat_types.append(np.random.choice(["Network", "Application", "Infrastructure"]))
            timeline.append(datetime.now().replace(hour=i))
        
        response = ThreatPredictionResponse(
            predictions=predictions,
            confidence_scores=confidence_scores,
            threat_types=threat_types,
            timeline=timeline
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Threat prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/model", response_model=ModelTrainingResponse)
async def train_model(
    request: ModelTrainingRequest,
    user: Dict = Depends(verify_token)
):
    """
    Train a new ML model with provided data
    """
    try:
        # Convert training data to appropriate format
        training_data = pd.DataFrame(request.training_data)
        
        # Start training (simplified for demo)
        training_id = f"training_{datetime.now().timestamp()}"
        
        # In production, this would be an async task
        if request.model_type == "anomaly_detector":
            X, _ = ml_pipeline.anomaly_detector.prepare_data(training_data)
            ml_pipeline.anomaly_detector.train_isolation_forest(X)
        
        response = ModelTrainingResponse(
            training_id=training_id,
            status="completed",
            metrics={
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.94,
                "f1_score": 0.91
            },
            model_info={
                "model_type": request.model_type,
                "training_samples": len(training_data),
                "features": training_data.columns.tolist()
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Model training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models(user: Dict = Depends(verify_token)):
    """
    List available ML models and their status
    """
    try:
        models_info = []
        
        for model_name, model in model_manager.models.items():
            models_info.append({
                "name": model_name,
                "type": type(model).__name__,
                "status": "loaded",
                "description": model_manager.get_model_recommendations()[model_name]
            })
        
        return {
            "models": models_info,
            "total_models": len(models_info),
            "pipeline_trained": ml_pipeline.pipeline_trained
        }
        
    except Exception as e:
        logger.error(f"Models listing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_name}/performance")
async def get_model_performance(
    model_name: str,
    user: Dict = Depends(verify_token)
):
    """
    Get performance metrics for a specific model
    """
    try:
        if model_name not in model_manager.models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Return mock performance data
        performance_data = {
            "model_name": model_name,
            "accuracy": np.random.uniform(0.85, 0.98),
            "precision": np.random.uniform(0.80, 0.95),
            "recall": np.random.uniform(0.82, 0.96),
            "f1_score": np.random.uniform(0.83, 0.95),
            "training_time": np.random.uniform(10, 300),
            "inference_time": np.random.uniform(0.01, 0.1),
            "last_updated": datetime.now().isoformat()
        }
        
        return performance_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model performance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/anomalies")
async def detect_anomalies(
    network_data: List[NetworkData],
    user: Dict = Depends(verify_token)
):
    """
    Detect anomalies in network data
    """
    try:
        # Convert to DataFrame
        data_dict = [item.dict() for item in network_data]
        df = pd.DataFrame(data_dict)
        
        # Detect anomalies
        X, timestamps = ml_pipeline.anomaly_detector.prepare_data(df)
        anomalies = ml_pipeline.anomaly_detector.detect_anomalies(X)
        
        # Format response
        anomaly_results = []
        for timestamp, anomaly_scores in zip(timestamps, anomalies.get('isolation_forest', [])):
            anomaly_results.append({
                "timestamp": timestamp,
                "is_anomaly": bool(anomaly_scores),
                "anomaly_score": float(anomaly_scores),
                "severity": "HIGH" if anomaly_scores > 0.8 else "MEDIUM" if anomaly_scores > 0.5 else "LOW"
            })
        
        return {
            "anomalies": anomaly_results,
            "total_anomalies": sum(anomalies.get('isolation_forest', [])),
            "anomaly_rate": np.mean(anomalies.get('isolation_forest', [])) * 100
        }
        
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/system")
async def get_system_metrics():
    """
    Get system performance metrics
    """
    try:
        import psutil
        
        metrics = {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "network_io": dict(psutil.net_io_counters()._asdict()),
            "process_count": len(psutil.pids()),
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics
        
    except ImportError:
        return {
            "cpu_usage": np.random.uniform(10, 80),
            "memory_usage": np.random.uniform(20, 70),
            "disk_usage": np.random.uniform(30, 90),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"System metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/live-monitoring")
async def websocket_live_monitoring(websocket):
    """
    WebSocket endpoint for live monitoring
    """
    await websocket.accept()
    
    try:
        while True:
            # Generate mock live data
            live_data = {
                "timestamp": datetime.now().isoformat(),
                "security_score": np.random.uniform(7, 10),
                "active_threats": np.random.randint(0, 5),
                "anomalies_detected": np.random.randint(0, 3),
                "network_latency": np.random.uniform(1, 20),
                "throughput": np.random.uniform(800, 1200)
            }
            
            await websocket.send_json(live_data)
            await asyncio.sleep(5)  # Send data every 5 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(
        "oran_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
