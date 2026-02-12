from flask import Blueprint, request, jsonify, current_app, render_template
from datetime import datetime
import numpy as np
from database.models import db, Prediction, MaintenanceEvent, UserFeedback, SystemAlert, PerformanceMetric
from functools import wraps
import time
import json

api_bp = Blueprint('api', __name__)
web_bp = Blueprint('web', __name__)

# Metrics
from prometheus_client import Counter, Histogram

prediction_counter = Counter('predictions_total', 'Total predictions made')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
api_errors = Counter('api_errors_total', 'Total API errors', ['endpoint', 'error_type'])


def track_performance(f):
    """Decorator to track API performance"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        try:
            result = f(*args, **kwargs)
            return result
        except Exception as e:
            api_errors.labels(endpoint=f.__name__, error_type=type(e).__name__).inc()
            raise
        finally:
            duration = time.time() - start_time
            prediction_latency.observe(duration)
    return decorated_function


def validate_input(required_fields):
    """Decorator to validate request input"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            
            missing = [field for field in required_fields if field not in data]
            if missing:
                return jsonify({'error': f'Missing required fields: {missing}'}), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# ============= PREDICTION ENDPOINTS =============

@api_bp.route('/predict', methods=['POST'])
@track_performance
@validate_input(['historical_data', 'asset_id'])
def predict():
    # 1. Initialize variables with safe defaults
    prediction_id = 0
    predicted_load = 0.0
    risk_score = 0.0
    risk_level = "minimal"
    failure_prob = 0.0
    is_anomaly = False
    should_maintain = False
    recommendations = []

    try:
        # 2. Get and validate data
        data = request.get_json()
        raw_history = data.get('historical_data', [])
        historical_data = np.array(raw_history).flatten()
        asset_id = str(data.get('asset_id', 'Unknown'))
        
        # 3. Model Prediction
        # Ensure input is 3D for LSTM: (1, 24, 1)
        model_input = historical_data[-24:].reshape(1, 24, 1)
        prediction_result = current_app.model_manager.predict(model_input)
        predicted_load = float(np.array(prediction_result.get('predicted_load', 0)).flatten()[0])

        # 4. Anomaly and Risk Logic
        current_app.anomaly_detector.update_baseline(historical_data)
        anomaly_result = current_app.anomaly_detector.detect_point_anomaly(predicted_load)
        is_anomaly = bool(np.array(anomaly_result.get('is_anomaly', False)).any())
        
        risk_result = current_app.risk_calculator.calculate_risk(
            prediction_data={'predicted_load': predicted_load},
            anomaly_data=anomaly_result,
            historical_data=historical_data
        )
        
        risk_score = float(np.array(risk_result.get('risk_score', 0)).flatten()[0])
        failure_prob = float(np.array(risk_result.get('estimated_failure_probability', 0)).flatten()[0])
        risk_level = str(risk_result.get('risk_level', 'minimal'))
        recommendations = risk_result.get('recommendations', [])

        maint_dec = current_app.maintenance_scheduler.should_schedule_maintenance(risk_result, asset_id)
        should_maintain = bool(np.array(maint_dec.get('should_schedule', False)).any())

        # 5. SAFE ALERTING (The old Step 7, updated for safety)
        hi_threshold = float(current_app.config.get('HIGH_RISK_THRESHOLD', 0.85))
        if risk_score >= hi_threshold:
            try:
                current_app.alert_system.create_alert(
                    alert_type='maintenance',
                    severity='critical' if risk_score > 0.9 else 'warning',
                    title=f'High Risk Alert: {asset_id}',
                    message=f'System predicts {risk_level} risk level.',
                    asset_id=asset_id,
                    metric_value=risk_score
                )
            except Exception as alert_e:
                current_app.logger.error(f"Alerting failed: {alert_e}")

        # 6. DATABASE SAVE (Alignment with Schema)
        try:
            db_history = historical_data.tolist()

            new_pred = Prediction(
                timestamp=datetime.utcnow(),
                asset_id=asset_id,
                historical_load=db_history, 
                predicted_load=predicted_load,
                risk_score=risk_score,
                risk_level=risk_level,
                anomaly_detected=is_anomaly,
                maintenance_recommended=should_maintain,
                model_version=str(current_app.config.get('MODEL_VERSION', '1.0.0')),
                
                # --- MISSING COLUMNS FIXED HERE ---
                prediction_horizon=24,  # Added: This matches your SEQUENCE_LENGTH
                temperature=float(data.get('temperature', 0.0) or 0.0),
                actual_load=None # Explicitly set as null
            )
            db.session.add(new_pred)
            db.session.commit()
            prediction_id = new_pred.id
            current_app.logger.info(f"Successfully saved prediction ID: {prediction_id}")
            
        except Exception as db_e:
            db.session.rollback()
            current_app.logger.error(f"DATABASE REJECTED SAVE: {str(db_e)}")
            # If you still see an error in the terminal, check if 'maintenance_priority' 
            # is in the schema further down the "-- More --" list!

        # 7. FINAL RESPONSE (The old Step 8, Triple-Nested for Frontend)
        full_payload = {
            "prediction_id": prediction_id,
            "asset_id": asset_id,
            "predicted_load": round(predicted_load, 2),
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "failure_probability": round(failure_prob, 4),
            "maintenance_recommended": should_maintain,
            "anomaly_detected": is_anomaly,
            "recommendations": recommendations,
            # Compatibility Wrappers for different JS versions
            "prediction": { "id": prediction_id, "predicted_load": predicted_load, "risk_level": risk_level },
            "risk_assessment": { "risk_score": risk_score, "risk_level": risk_level, "failure_probability": failure_prob },
            "maintenance": { "recommended": should_maintain, "recommendations": recommendations },
            "data": { "risk_level": risk_level, "risk_score": risk_score }
        }
        return jsonify(full_payload), 200

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Global Pipeline Crash: {str(e)}")
        return jsonify({"error": str(e), "status": "failed"}), 500

    
    
# ============= BATCH & FEEDBACK ENDPOINTS (BULLETPROOF VERSION) =============

@api_bp.route('/predict/batch', methods=['POST'])
@track_performance
@validate_input(['predictions'])
def predict_batch():
    """Make predictions for multiple assets with deep scalar safety"""
    try:
        data = request.get_json()
        predictions_input = data['predictions']
        
        if len(predictions_input) > current_app.config['MAX_BATCH_SIZE']:
            return jsonify({
                'error': f'Batch size exceeds maximum of {current_app.config["MAX_BATCH_SIZE"]}'
            }), 400
        
        results = []
        
        for pred_input in predictions_input:
            historical_data = np.array(pred_input['historical_data'])
            asset_id = pred_input['asset_id']
            
            if len(historical_data.shape) == 1:
                historical_data = historical_data.reshape(-1, 1)
            
            # 1. Make prediction
            prediction_result = current_app.model_manager.predict(historical_data)
            
            # 2. BULLETPROOF FIX: Force predicted_load to float
            raw_load = prediction_result.get('predicted_load', 0)
            predicted_load = float(np.array(raw_load).flatten()[0])
            
            # 3. Detect anomalies
            anomaly_result = current_app.anomaly_detector.detect_point_anomaly(predicted_load)
            
            # 4. Calculate risk
            risk_result = current_app.risk_calculator.calculate_risk(
                prediction_data={'predicted_load': predicted_load},
                anomaly_data=anomaly_result,
                historical_data=historical_data.flatten()
            )
            
            # 5. BULLETPROOF FIX: Force risk_score to float
            raw_risk = risk_result.get('risk_score', 0)
            risk_score = float(np.array(raw_risk).flatten()[0])
            
            # 6. Safety check on maintenance recommendation
            should_maintain = bool(np.array(risk_result.get('maintenance_recommended', False)).any())
            
            results.append({
                'asset_id': asset_id,
                'predicted_load': predicted_load,
                'risk_score': risk_score,
                'risk_level': risk_result.get('risk_level', 'Unknown'),
                'maintenance_recommended': should_maintain
            })
        
        return jsonify({'predictions': results}), 200
        
    except Exception as e:
        current_app.logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/predict/<int:prediction_id>/feedback', methods=['POST'])
@validate_input(['actual_load'])
def update_prediction_feedback(prediction_id):
    """Update prediction with actual observed value and check for model drift"""
    try:
        data = request.get_json()
        
        # Ensure input actual_load is a clean float
        actual_load = float(np.array(data['actual_load']).flatten()[0])
        
        prediction = Prediction.query.get_or_404(prediction_id)
        prediction.actual_load = actual_load
        
        # Calculate errors
        error = abs(prediction.predicted_load - actual_load)
        percentage_error = (error / actual_load * 100) if actual_load != 0 else 0
        
        prediction.absolute_error = error
        prediction.percentage_error = percentage_error
        
        db.session.commit()
        
        # Update drift detector
        current_app.drift_detector.add_error(error)
        
        # Check for drift
        drift_result = current_app.drift_detector.detect_drift()
        
        # BULLETPROOF FIX: Ensure drift detection comparison doesn't fail
        is_drifted = bool(np.array(drift_result.get('drift_detected', False)).any())
        
        if is_drifted:
            # Convert metric values to float safely
            recent_err = drift_result.get('recent_error', 0)
            metric_val = float(np.array(recent_err).flatten()[0])
            
            increase_pct = drift_result.get('error_increase_pct', 0)
            increase_val = float(np.array(increase_pct).flatten()[0])
            
            current_app.alert_system.create_alert(
                alert_type='performance',
                severity='warning',
                title='Model drift detected',
                message=f'Error increased by {increase_val:.1f}%',
                metric_value=metric_val
            )
        
        return jsonify({
            'message': 'Feedback recorded',
            'error': error,
            'percentage_error': percentage_error
        }), 200
        
    except Exception as e:
        db.session.rollback() 
        current_app.logger.error(f"Feedback error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============= MAINTENANCE ENDPOINTS =============

@api_bp.route('/maintenance/events', methods=['GET'])
def get_maintenance_events():
    """Get maintenance events with optional filtering"""
    asset_id = request.args.get('asset_id')
    status = request.args.get('status')
    limit = min(int(request.args.get('limit', 100)), 1000)
    
    query = MaintenanceEvent.query
    
    if asset_id:
        query = query.filter_by(asset_id=asset_id)
    if status:
        query = query.filter_by(status=status)
    
    events = query.order_by(MaintenanceEvent.timestamp.desc()).limit(limit).all()
    
    return jsonify({
        'events': [event.to_dict() for event in events],
        'count': len(events)
    }), 200


@api_bp.route('/maintenance/events', methods=['POST'])
@validate_input(['asset_id', 'event_type'])
def create_maintenance_event():
    """Create a new maintenance event"""
    try:
        data = request.get_json()
        
        event = MaintenanceEvent(
            asset_id=data['asset_id'],
            event_type=data['event_type'],
            status=data.get('status', 'scheduled'),
            prediction_id=data.get('prediction_id'),
            risk_score_at_detection=data.get('risk_score'),
            scheduled_date=datetime.fromisoformat(data['scheduled_date']) if 'scheduled_date' in data else None,
            technician_notes=data.get('notes')
        )
        
        db.session.add(event)
        db.session.commit()
        
        current_app.loggers['maintenance'].info(
            'Maintenance event created',
            extra={
                'asset_id': event.asset_id,
                'event_type': event.event_type,
                'status': event.status
            }
        )
        
        return jsonify(event.to_dict()), 201
        
    except Exception as e:
        current_app.logger.error(f"Create maintenance event error: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/maintenance/events/<int:event_id>', methods=['PATCH'])
def update_maintenance_event(event_id):
    """Update maintenance event status"""
    try:
        event = MaintenanceEvent.query.get_or_404(event_id)
        data = request.get_json()
        
        if 'status' in data:
            event.status = data['status']
        if 'completion_date' in data:
            event.completion_date = datetime.fromisoformat(data['completion_date'])
        if 'failure_prevented' in data:
            event.failure_prevented = data['failure_prevented']
        if 'actual_issue_found' in data:
            event.actual_issue_found = data['actual_issue_found']
        if 'downtime_hours' in data:
            event.downtime_hours = data['downtime_hours']
        if 'technician_notes' in data:
            event.technician_notes = data['technician_notes']
        
        db.session.commit()
        
        return jsonify(event.to_dict()), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


# ============= MONITORING ENDPOINTS =============

@api_bp.route('/monitoring/performance', methods=['GET'])
def get_performance_metrics():
    """Get current performance metrics"""
    try:
        hours = int(request.args.get('hours', 24))
        
        metrics = current_app.performance_tracker.get_recent_metrics(hours)
        
        return jsonify(metrics), 200
    except Exception as e:
        current_app.logger.error(f"Performance metrics error: {str(e)}")
        # Return mock data if no real data exists yet
        return jsonify({
            'time_period': {
                'hours': hours,
                'start': datetime.utcnow().isoformat(),
                'end': datetime.utcnow().isoformat()
            },
            'volume': {
                'total_predictions': 0,
                'predictions_with_feedback': 0
            },
            'accuracy': {
                'mae': None,
                'mse': None,
                'rmse': None,
                'mape': None
            },
            'risk_distribution': {
                'high': 0,
                'medium': 0,
                'low': 0,
                'minimal': 0
            },
            'anomalies': {
                'count': 0,
                'rate': 0.0
            },
            'maintenance': {
                'alerts_generated': 0,
                'alert_rate': 0.0
            },
            'latency': {
                'avg_ms': None,
                'p50_ms': None,
                'p95_ms': None,
                'p99_ms': None
            }
        }), 200


@api_bp.route('/monitoring/drift', methods=['GET'])
def check_drift():
    """Check for model drift"""
    drift_result = current_app.drift_detector.detect_drift()
    
    return jsonify(drift_result), 200


@api_bp.route('/monitoring/alerts', methods=['GET'])
def get_alerts():
    """Get system alerts"""
    try:
        severity = request.args.get('severity')
        resolved = request.args.get('resolved', 'false').lower() == 'true'
        limit = min(int(request.args.get('limit', 50)), 500)
        
        query = SystemAlert.query.filter_by(resolved=resolved)
        
        if severity:
            query = query.filter_by(severity=severity)
        
        alerts = query.order_by(SystemAlert.timestamp.desc()).limit(limit).all()
        
        return jsonify({
            'alerts': [{
                'id': a.id,
                'timestamp': a.timestamp.isoformat(),
                'alert_type': a.alert_type,
                'severity': a.severity,
                'title': a.title,
                'message': a.message,
                'asset_id': a.asset_id
            } for a in alerts],
            'count': len(alerts)
        }), 200
    except Exception as e:
        current_app.logger.error(f"Alerts fetch error: {str(e)}")
        # Return empty alerts if database not initialized
        return jsonify({
            'alerts': [],
            'count': 0
        }), 200


@api_bp.route('/monitoring/model/info', methods=['GET'])
def get_model_info():
    """Get current model information"""
    info = current_app.model_manager.get_model_info()
    return jsonify(info), 200


# ============= USER FEEDBACK ENDPOINTS =============

@api_bp.route('/feedback', methods=['POST'])
@validate_input(['user_id', 'overall_rating'])
def submit_feedback():
    """Submit user feedback"""
    try:
        data = request.get_json()
        
        feedback = UserFeedback(
            user_id=data['user_id'],
            prediction_id=data.get('prediction_id'),
            accuracy_rating=data.get('accuracy_rating'),
            usefulness_rating=data.get('usefulness_rating'),
            interface_rating=data.get('interface_rating'),
            overall_rating=data['overall_rating'],
            comments=data.get('comments'),
            suggested_improvements=data.get('suggested_improvements'),
            user_role=data.get('user_role'),
            was_maintenance_performed=data.get('was_maintenance_performed'),
            was_prediction_accurate=data.get('was_prediction_accurate')
        )
        
        db.session.add(feedback)
        db.session.commit()
        
        return jsonify(feedback.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


# ============= WEB INTERFACE =============

@web_bp.route('/')
@web_bp.route('/dashboard')
def dashboard():
    """Main dashboard - both URLs now point to your main file"""
    return render_template('index.html')


@web_bp.route('/feedback_form')
def feedback_form():
    """Feedback form page"""
    return render_template('feedback.html')