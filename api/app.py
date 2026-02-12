from flask import Flask
from flask_cors import CORS
from config.settings import get_config
from config.logging_config import setup_logging
from database.models import db
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import os

def create_app(config_name=None):
    """Application factory pattern"""
    
    app = Flask(__name__)
    
    # Load configuration
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    config_class = get_config()
    app.config.from_object(config_class)
    
    # Initialize extensions
    db.init_app(app)
    CORS(app)
    
    # Setup logging
    loggers = setup_logging(app)
    app.loggers = loggers
    
    # Create database tables
    with app.app_context():
        db.create_all()
        app.logger.info("Database tables created")
    
    # Register blueprints
    from api.routes import api_bp, web_bp
    app.register_blueprint(api_bp, url_prefix='/api/v1')
    app.register_blueprint(web_bp)
    
    # Add Prometheus metrics endpoint
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
        '/metrics': make_wsgi_app()
    })
    
    # Initialize model manager (singleton)
    from models.model_manager import ModelManager
    from models.anomaly_detector import AnomalyDetector, DriftDetector
    from models.risk_calculator import RiskCalculator, MaintenanceScheduler
    
    app.model_manager = ModelManager(app.config)
    app.anomaly_detector = AnomalyDetector(app.config)
    app.drift_detector = DriftDetector(app.config)
    app.risk_calculator = RiskCalculator(app.config)
    app.maintenance_scheduler = MaintenanceScheduler(app.config)
    
    # Initialize monitoring
    from monitoring.performance_tracker import PerformanceTracker
    from monitoring.alert_system import AlertSystem
    
    app.performance_tracker = PerformanceTracker(app.config, db)
    app.alert_system = AlertSystem(app.config, db)
    
    app.logger.info(f"Application initialized in {config_name} mode")
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return {
            'status': 'healthy',
            'model_loaded': app.model_manager.model is not None,
            'database': 'connected',
            'version': app.config.MODEL_VERSION
        }
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG']
    )