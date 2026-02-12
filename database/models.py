import enum
from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Index, CheckConstraint, Numeric

db = SQLAlchemy()

# --- Enums for Data Integrity ---

class RiskLevel(enum.Enum):
    minimal = "minimal"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"

class MaintenanceStatus(enum.Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class EventType(enum.Enum):
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"

class UserRole(enum.Enum):
    ENGINEER = "engineer"
    OPERATOR = "operator"
    MANAGER = "manager"

class AlertSeverity(enum.Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

class AlertType(enum.Enum):
    PERFORMANCE = "performance"
    MAINTENANCE = "maintenance"
    SYSTEM = "system"

# --- Models ---

class Prediction(db.Model):
    """Store all predictions for tracking and analysis"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    # Modernized timezone-aware datetime
    timestamp = db.Column(db.DateTime(timezone=True), nullable=False, 
                          default=lambda: datetime.now(timezone.utc), index=True)
    asset_id = db.Column(db.String(100), nullable=False, index=True)
    
    # Input features
    historical_load = db.Column(db.JSON, nullable=False)
    temperature = db.Column(db.Float)
    
    # Predictions
    predicted_load = db.Column(db.Float, nullable=False)
    prediction_horizon = db.Column(db.Integer, nullable=False)  # hours ahead
    
    # Actual values (filled in later)
    actual_load = db.Column(db.Float)
    
    # Risk assessment
    risk_score = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.Enum(RiskLevel), nullable=False, default=RiskLevel.low)
    anomaly_detected = db.Column(db.Boolean, default=False)
    
    # Maintenance
    maintenance_recommended = db.Column(db.Boolean, default=False)
    maintenance_priority = db.Column(db.Enum(RiskLevel), nullable=True, default=RiskLevel.low)
    
    # Model metadata linked to ModelVersion table
    model_version = db.Column(db.String(50), db.ForeignKey('model_versions.version'), nullable=False)
    
    # Performance metrics
    absolute_error = db.Column(db.Float)
    percentage_error = db.Column(db.Float)
    
    # Relationships - optimized from dynamic to selectin for typical load sizes
    maintenance_events = db.relationship('MaintenanceEvent', backref='prediction', lazy='selectin')
    
    __table_args__ = (
        Index('idx_asset_timestamp', 'asset_id', 'timestamp'),
        CheckConstraint('risk_score >= 0 AND risk_score <= 1', name='valid_risk_score'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'asset_id': self.asset_id,
            'predicted_load': self.predicted_load,
            'actual_load': self.actual_load,
            'risk_score': self.risk_score,
            'risk_level': self.risk_level.value,
            'anomaly_detected': self.anomaly_detected,
            'maintenance_recommended': self.maintenance_recommended,
            'absolute_error': self.absolute_error,
            'model_version': self.model_version
        }


class MaintenanceEvent(db.Model):
    """Track maintenance events and outcomes"""
    __tablename__ = 'maintenance_events'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime(timezone=True), nullable=False, 
                          default=lambda: datetime.now(timezone.utc), index=True)
    asset_id = db.Column(db.String(100), nullable=False, index=True)
    
    # Event details using Enums
    event_type = db.Column(db.Enum(EventType), nullable=False)
    status = db.Column(db.Enum(MaintenanceStatus), nullable=False, default=MaintenanceStatus.SCHEDULED)
    
    # Prediction that triggered this
    prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'))
    
    # Risk information
    risk_score_at_detection = db.Column(db.Float)
    estimated_failure_probability = db.Column(db.Float)
    
    # Scheduling
    scheduled_date = db.Column(db.DateTime(timezone=True))
    completion_date = db.Column(db.DateTime(timezone=True))
    
    # Outcome
    failure_prevented = db.Column(db.Boolean)
    actual_issue_found = db.Column(db.String(500))
    downtime_hours = db.Column(db.Float)
    # Using Numeric for financial accuracy
    cost_estimate = db.Column(db.Numeric(precision=12, scale=2)) 
    
    # Notes
    technician_notes = db.Column(db.Text)
    
    __table_args__ = (
        Index('idx_status_scheduled', 'status', 'scheduled_date'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'asset_id': self.asset_id,
            'event_type': self.event_type.value,
            'status': self.status.value,
            'risk_score': self.risk_score_at_detection,
            'scheduled_date': self.scheduled_date.isoformat() if self.scheduled_date else None,
            'completion_date': self.completion_date.isoformat() if self.completion_date else None,
            'failure_prevented': self.failure_prevented,
            'downtime_hours': self.downtime_hours,
            'cost_estimate': float(self.cost_estimate) if self.cost_estimate else None
        }


class ModelVersion(db.Model):
    """Track model versions and their performance"""
    __tablename__ = 'model_versions'
    
    id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.String(50), unique=True, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, 
                           default=lambda: datetime.now(timezone.utc))
    
    # Model metadata
    model_path = db.Column(db.String(500), nullable=False)
    architecture = db.Column(db.JSON)
    hyperparameters = db.Column(db.JSON)
    
    # Training data
    training_samples = db.Column(db.Integer)
    training_date_range = db.Column(db.JSON)
    
    # Performance metrics
    train_mae = db.Column(db.Float)
    train_mse = db.Column(db.Float)
    val_mae = db.Column(db.Float)
    val_mse = db.Column(db.Float)
    test_mae = db.Column(db.Float)
    test_mse = db.Column(db.Float)
    
    # Production performance
    production_mae = db.Column(db.Float)
    production_mape = db.Column(db.Float)
    predictions_count = db.Column(db.Integer, default=0)
    
    # Status
    is_active = db.Column(db.Boolean, default=False)
    is_champion = db.Column(db.Boolean, default=False)
    deployment_date = db.Column(db.DateTime(timezone=True))
    retirement_date = db.Column(db.DateTime(timezone=True))
    
    # A/B testing
    ab_test_group = db.Column(db.String(10))  # A, B, or None
    
    def to_dict(self):
        return {
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'train_mae': self.train_mae,
            'val_mae': self.val_mae,
            'production_mae': self.production_mae,
            'is_active': self.is_active,
            'predictions_count': self.predictions_count
        }


class PerformanceMetric(db.Model):
    """Store time-series performance metrics"""
    __tablename__ = 'performance_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime(timezone=True), nullable=False, 
                          default=lambda: datetime.now(timezone.utc), index=True)
    model_version = db.Column(db.String(50), nullable=False)
    
    # Error metrics
    mae = db.Column(db.Float)
    mse = db.Column(db.Float)
    rmse = db.Column(db.Float)
    mape = db.Column(db.Float)
    
    # Prediction statistics
    predictions_count = db.Column(db.Integer)
    anomalies_detected = db.Column(db.Integer)
    maintenance_alerts = db.Column(db.Integer)
    
    # Risk distribution
    high_risk_count = db.Column(db.Integer)
    medium_risk_count = db.Column(db.Integer)
    low_risk_count = db.Column(db.Integer)
    
    # System health
    api_latency_ms = db.Column(db.Float)
    prediction_latency_ms = db.Column(db.Float)
    
    __table_args__ = (
        Index('idx_timestamp_model', 'timestamp', 'model_version'),
    )


class UserFeedback(db.Model):
    """Store user feedback on predictions and recommendations"""
    __tablename__ = 'user_feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime(timezone=True), nullable=False, 
                          default=lambda: datetime.now(timezone.utc))
    user_id = db.Column(db.String(100), nullable=False)
    
    # Feedback target
    prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'))
    
    # Ratings (1-5)
    accuracy_rating = db.Column(db.Integer)
    usefulness_rating = db.Column(db.Integer)
    interface_rating = db.Column(db.Integer)
    overall_rating = db.Column(db.Integer)
    
    # Qualitative feedback
    comments = db.Column(db.Text)
    suggested_improvements = db.Column(db.Text)
    
    # Context
    user_role = db.Column(db.Enum(UserRole))
    
    # Response
    was_maintenance_performed = db.Column(db.Boolean)
    was_prediction_accurate = db.Column(db.Boolean)
    
    __table_args__ = (
        CheckConstraint('accuracy_rating >= 1 AND accuracy_rating <= 5', name='valid_accuracy'),
        CheckConstraint('usefulness_rating >= 1 AND usefulness_rating <= 5', name='valid_usefulness'),
        CheckConstraint('overall_rating >= 1 AND overall_rating <= 5', name='valid_overall'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'accuracy_rating': self.accuracy_rating,
            'usefulness_rating': self.usefulness_rating,
            'overall_rating': self.overall_rating,
            'comments': self.comments,
            'was_prediction_accurate': self.was_prediction_accurate,
            'user_role': self.user_role.value if self.user_role else None
        }


class SystemAlert(db.Model):
    """Track system alerts and notifications"""
    __tablename__ = 'system_alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime(timezone=True), nullable=False, 
                          default=lambda: datetime.now(timezone.utc), index=True)
    
    # Alert details
    alert_type = db.Column(db.Enum(AlertType), nullable=False)
    severity = db.Column(db.Enum(AlertSeverity), nullable=False)
    
    # Message
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    
    # Context
    asset_id = db.Column(db.String(100))
    metric_value = db.Column(db.Float)
    threshold_value = db.Column(db.Float)
    
    # Status
    acknowledged = db.Column(db.Boolean, default=False)
    acknowledged_by = db.Column(db.String(100))
    acknowledged_at = db.Column(db.DateTime(timezone=True))
    resolved = db.Column(db.Boolean, default=False)
    resolved_at = db.Column(db.DateTime(timezone=True))
    
    __table_args__ = (
        Index('idx_severity_resolved', 'severity', 'resolved'),
    )