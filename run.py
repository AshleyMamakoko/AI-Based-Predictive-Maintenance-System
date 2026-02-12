"""
Smart Grid Predictive Maintenance System
Main execution script
"""

import os 
import sys 
from api.app import create_app

def main():
    """Initialize and run the application"""
# Create application instance
app = create_app()

# Get configuration
port = int(os.environ.get('PORT', 5000))
host = os.environ.get('HOST', '0.0.0.0')
debug = app.config['DEBUG']

# Display startup information
app.logger.info("="* 60)
app.logger.info("Smart Grid Predictive Maintenance System")
app.logger.info("="* 60)
app.logger.info(f"Environment: {os.environ.get('FLASK_ENV', 'development')}")
app.logger.info(f"Debug Mode: {debug}")
app.logger.info(f"Model Version: {app.config['MODEL_VERSION']}")
app.logger.info(f"Model Loaded: {app.model_manager.model is not None}")
app.logger.info(f"Database: Connected")
app.logger.info(f"Server: http://{host}:{port}")
app.logger.info("="* 60)

if debug:
    app.logger.warning("Running in DEBUG mode - NOT suitable for production!")

# Run application
try:
    app.run(
        host=host,
        port=port,
        debug=debug,
        use_reloader=debug
    )
except KeyboardInterrupt:
    app.logger.info("Shutting down...")
    sys.exit(0)
except Exception as e:
    app.logger.error(f"Application error: {str(e)}")
    sys.exit(1)
if __name__ == "__main__":
    main()