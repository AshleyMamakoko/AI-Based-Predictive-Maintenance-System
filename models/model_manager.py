import os
import logging
import numpy as np
import tensorflow as tf
# This is the safest way to import keras in TF 2.15
from tensorflow import keras 
from datetime import datetime
import joblib
import json
from threading import Lock

logger = logging.getLogger(__name__)


class ModelManager:
    """Manage LSTM model loading, inference, and versioning"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = None
        self.model_version = None
        self.model_metadata = {}
        self.lock = Lock()
        
        # Load current model
        self.load_model()
    
    def load_model(self, model_path=None, version=None):
        """Load trained LSTM model and preprocessing artifacts"""
        with self.lock:
            try:
                # Use .get() to avoid AttributeErrors from Flask Config objects
                if model_path is None:
                    model_path = self.config.get('CURRENT_MODEL_PATH')
                
                if not model_path or not os.path.exists(model_path):
                    logger.error(f"Model file not found at: {model_path}")
                    raise FileNotFoundError(f"Model not found at {model_path}")
                
                # Load Keras model with compile=False to bypass version metadata conflicts
                logger.info(f"Attempting to load model from {model_path}")
                self.model = keras.models.load_model(model_path, compile=False)
                
                # Load scaler - looks for 'current_model_scaler.pkl'
                scaler_path = model_path.replace('.h5', '_scaler.pkl')
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    logger.info(f"Loaded scaler from {scaler_path}")
                else:
                    logger.warning(f"No scaler found at {scaler_path}, will use raw values")
                
                # Load metadata
                metadata_path = model_path.replace('.h5', '_metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.model_metadata = json.load(f)
                    logger.info("Loaded model metadata")
                
                self.model_version = version or self.config.get('MODEL_VERSION', '1.0.0')
                logger.info(f"Model {self.model_version} loaded successfully")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                # We don't raise here so the app can still boot, 
                # but predictions will fail gracefully
                self.model = None 
                return False
    
    def predict(self, input_data):
        """Make prediction on input sequence"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Check logs for loading errors.")
        
        try:
            # Ensure correct shape (batch, sequence, features)
            if len(input_data.shape) == 2:
                input_data = np.expand_dims(input_data, axis=0)
            
            # Scale if scaler available
            if self.scaler is not None:
                original_shape = input_data.shape
                input_data_2d = input_data.reshape(-1, input_data.shape[-1])
                input_data_scaled = self.scaler.transform(input_data_2d)
                input_data = input_data_scaled.reshape(original_shape)
            
            # Make prediction
            start_time = datetime.utcnow()
            prediction = self.model.predict(input_data, verbose=0)
            inference_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Inverse transform if needed
            if self.scaler is not None:
                prediction = self.scaler.inverse_transform(
                    prediction.reshape(-1, prediction.shape[-1])
                )
            
            result = {
                'predicted_load': float(prediction[0][0]),
                'model_version': self.model_version,
                'inference_time_ms': inference_time,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def predict_sequence(self, input_data, horizon=24):
        """Predict multiple time steps ahead"""
        predictions = []
        current_sequence = input_data.copy()
        
        for _ in range(horizon):
            pred = self.predict(current_sequence)
            predictions.append(pred['predicted_load'])
            
            # Update sequence for next prediction (rolling window)
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = pred['predicted_load']
        
        return predictions

    def evaluate_on_batch(self, X, y):
        """Evaluate model performance on a batch"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1]))
                X_scaled = X_scaled.reshape(X.shape)
            else:
                X_scaled = X
            
            predictions = self.model.predict(X_scaled, verbose=0)
            
            if self.scaler is not None:
                predictions = self.scaler.inverse_transform(
                    predictions.reshape(-1, predictions.shape[-1])
                )
                predictions = predictions.reshape(-1)
            
            mae = np.mean(np.abs(predictions - y))
            mse = np.mean((predictions - y) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y - predictions) / y)) * 100
            
            return {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'samples': len(y)
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

    def get_model_info(self):
        """Get current model information"""
        if self.model is None:
            return {'status': 'not_loaded'}
        
        return {
            'version': self.model_version,
            'total_params': self.model.count_params() if hasattr(self.model, 'count_params') else 0,
            'metadata': self.model_metadata,
            'scaler_loaded': self.scaler is not None
        }

    def save_model(self, model, scaler, metadata, version):
        """Save new model version"""
        try:
            model_dir = self.config.get('MODEL_DIR')
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, f'model_{version}.h5')
            scaler_path = os.path.join(model_dir, f'model_{version}_scaler.pkl')
            metadata_path = os.path.join(model_dir, f'model_{version}_metadata.json')
            
            model.save(model_path)
            if scaler is not None:
                joblib.dump(scaler, scaler_path)
            
            metadata['saved_at'] = datetime.utcnow().isoformat()
            metadata['version'] = version
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return model_path
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

    def switch_to_version(self, version):
        """Switch to a different model version"""
        model_path = os.path.join(self.config.get('MODEL_DIR'), f'model_{version}.h5')
        if not os.path.exists(model_path):
            raise ValueError(f"Model version {version} not found")
        self.load_model(model_path, version)


class ModelEnsemble:
    """Manage multiple models for A/B testing or ensemble predictions"""
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.weights = {}

    def add_model(self, name, model_manager, weight=1.0):
        self.models[name] = model_manager
        self.weights[name] = weight

    def predict_ensemble(self, input_data, method='weighted_average'):
        if not self.models:
            raise RuntimeError("No models in ensemble")
        
        predictions = {}
        for name, model_manager in self.models.items():
            try:
                pred = model_manager.predict(input_data)
                predictions[name] = pred
            except Exception as e:
                logger.error(f"Model {name} prediction failed: {str(e)}")
        
        if method == 'weighted_average':
            total_weight = sum(self.weights[name] for name in predictions.keys())
            weighted_sum = sum(
                predictions[name]['predicted_load'] * self.weights[name]
                for name in predictions.keys()
            )
            return {
                'predicted_load': weighted_sum / total_weight,
                'method': 'weighted_average',
                'individual_predictions': predictions,
                'timestamp': datetime.utcnow().isoformat()
            }
        else:
            raise ValueError(f"Unknown ensemble method: {method}")