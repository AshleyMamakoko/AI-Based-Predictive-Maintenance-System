import os
from datetime import timedelta
import urllib.parse

# 1. Database Password Encoding
_db_password = "Mapale123@"
_encoded_password = urllib.parse.quote_plus(_db_password)

class Config:
    """Base configuration"""
    # Flask Security
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = False
    TESTING = False
    
    # Database Configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        f'postgresql://grid_user:{_encoded_password}@localhost:5432/smart_grid_maintenance'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
    
    # 2. Model Configuration
    MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'saved_models'))
    CURRENT_MODEL_PATH = os.path.join(MODEL_DIR, 'current_model.h5')
    MODEL_VERSION = '1.0.0'
    
    # Maintenance & Risk Thresholds
    ANOMALY_THRESHOLD = 2.5  
    HIGH_RISK_THRESHOLD = 0.75
    MEDIUM_RISK_THRESHOLD = 0.50
    LOW_RISK_THRESHOLD = 0.25
    
    # Forecasting Settings
    SEQUENCE_LENGTH = 24
    FORECAST_HORIZON = 24
    
    # Performance Monitoring
    MAE_ALERT_THRESHOLD = 50.0 
    MAPE_ALERT_THRESHOLD = 10.0
    DRIFT_DETECTION_WINDOW = 168
    
    # Maintenance Logic
    MAINTENANCE_WINDOW_HOURS = 72
    CRITICAL_FAILURE_PROBABILITY = 0.80
    MAINTENANCE_COOLDOWN_HOURS = 24
    
    # API Configuration
    MAX_BATCH_SIZE = 100
    API_RATE_LIMIT = "1000 per hour"
    REQUEST_TIMEOUT = 30 
    
    # Logging & Retention
    METRICS_RETENTION_DAYS = 90
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'logs/smart_grid_maintenance.log'
    
    # Retraining Pipeline
    RETRAIN_INTERVAL_DAYS = 7
    MIN_NEW_SAMPLES_FOR_RETRAIN = 1000
    VALIDATION_SPLIT = 0.2

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_ECHO = True
    LOG_LEVEL = 'DEBUG'
    
    # Ensure all thresholds are visible to the Flask config parser
    ANOMALY_THRESHOLD = Config.ANOMALY_THRESHOLD
    HIGH_RISK_THRESHOLD = Config.HIGH_RISK_THRESHOLD
    MEDIUM_RISK_THRESHOLD = Config.MEDIUM_RISK_THRESHOLD
    LOW_RISK_THRESHOLD = Config.LOW_RISK_THRESHOLD
    
    # Model settings
    MODEL_DIR = Config.MODEL_DIR
    CURRENT_MODEL_PATH = Config.CURRENT_MODEL_PATH

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Map thresholds to prevent crashes in production
    ANOMALY_THRESHOLD = Config.ANOMALY_THRESHOLD
    HIGH_RISK_THRESHOLD = Config.HIGH_RISK_THRESHOLD
    MEDIUM_RISK_THRESHOLD = Config.MEDIUM_RISK_THRESHOLD
    LOW_RISK_THRESHOLD = Config.LOW_RISK_THRESHOLD
    
    # Map model paths
    MODEL_DIR = Config.MODEL_DIR
    CURRENT_MODEL_PATH = Config.CURRENT_MODEL_PATH
    
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable must be set in production")
    
    ALERT_EMAIL_ENABLED = True
    WORKERS = int(os.environ.get('WORKERS', 8))
    
    # Stricter thresholds for production
    MAE_ALERT_THRESHOLD = 30.0
    MAPE_ALERT_THRESHOLD = 5.0

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False

# Configuration dictionary mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Returns the config class based on the FLASK_ENV environment variable"""
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])