import logging
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pythonjsonlogger import jsonlogger


def setup_logging(app):
    """Configure comprehensive logging system"""
    
    # Create logs directory
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Set log level
    log_level = getattr(logging, app.config['LOG_LEVEL'].upper())
    app.logger.setLevel(log_level)
    
    # JSON formatter for structured logging
    json_formatter = jsonlogger.JsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s %(pathname)s %(lineno)d'
    )
    
    # Standard formatter for human-readable logs
    standard_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    )
    
    # Application log handler (rotating by size)
    app_handler = RotatingFileHandler(
        os.path.join(log_dir, 'application.log'),
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    app_handler.setLevel(log_level)
    app_handler.setFormatter(standard_formatter)
    
    # Error log handler (rotating by size)
    error_handler = RotatingFileHandler(
        os.path.join(log_dir, 'error.log'),
        maxBytes=10485760,
        backupCount=10
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(standard_formatter)
    
    # Prediction log handler (time-based rotation)
    prediction_handler = TimedRotatingFileHandler(
        os.path.join(log_dir, 'predictions.log'),
        when='midnight',
        interval=1,
        backupCount=30
    )
    prediction_handler.setLevel(logging.INFO)
    prediction_handler.setFormatter(json_formatter)
    
    # Maintenance log handler
    maintenance_handler = TimedRotatingFileHandler(
        os.path.join(log_dir, 'maintenance.log'),
        when='midnight',
        interval=1,
        backupCount=90
    )
    maintenance_handler.setLevel(logging.INFO)
    maintenance_handler.setFormatter(json_formatter)
    
    # Performance metrics handler
    metrics_handler = TimedRotatingFileHandler(
        os.path.join(log_dir, 'metrics.log'),
        when='H',  # Hourly
        interval=1,
        backupCount=168  # One week
    )
    metrics_handler.setLevel(logging.INFO)
    metrics_handler.setFormatter(json_formatter)
    
    # Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(standard_formatter)
    
    # Add handlers to app logger
    app.logger.addHandler(app_handler)
    app.logger.addHandler(error_handler)
    app.logger.addHandler(console_handler)
    
    # Create specialized loggers
    prediction_logger = logging.getLogger('predictions')
    prediction_logger.setLevel(logging.INFO)
    prediction_logger.addHandler(prediction_handler)
    
    maintenance_logger = logging.getLogger('maintenance')
    maintenance_logger.setLevel(logging.INFO)
    maintenance_logger.addHandler(maintenance_handler)
    
    metrics_logger = logging.getLogger('metrics')
    metrics_logger.setLevel(logging.INFO)
    metrics_logger.addHandler(metrics_handler)
    
    # Prevent propagation to root logger
    prediction_logger.propagate = False
    maintenance_logger.propagate = False
    metrics_logger.propagate = False
    
    app.logger.info('Logging system initialized')
    
    return {
        'app': app.logger,
        'predictions': prediction_logger,
        'maintenance': maintenance_logger,
        'metrics': metrics_logger
    }


def log_prediction(logger, data):
    """Log prediction with structured format"""
    logger.info('Prediction made', extra={
        'event_type': 'prediction',
        'timestamp': data.get('timestamp'),
        'actual_load': data.get('actual_load'),
        'predicted_load': data.get('predicted_load'),
        'error': data.get('error'),
        'risk_score': data.get('risk_score'),
        'maintenance_recommended': data.get('maintenance_recommended')
    })


def log_maintenance_event(logger, data):
    """Log maintenance event"""
    logger.info('Maintenance event', extra={
        'event_type': 'maintenance',
        'timestamp': data.get('timestamp'),
        'asset_id': data.get('asset_id'),
        'event': data.get('event'),
        'risk_level': data.get('risk_level'),
        'estimated_downtime': data.get('estimated_downtime'),
        'preventive': data.get('preventive', True)
    })


def log_performance_metric(logger, metric_name, value, tags=None):
    """Log performance metric"""
    logger.info('Performance metric', extra={
        'metric': metric_name,
        'value': value,
        'tags': tags or {}
    })