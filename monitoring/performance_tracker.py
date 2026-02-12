import logging
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from database.models import PerformanceMetric, Prediction
from prometheus_client import Gauge, Counter, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_mae = Gauge('prediction_mae', 'Mean Absolute Error')
prediction_mape = Gauge('prediction_mape', 'Mean Absolute Percentage Error')
risk_distribution = Gauge('risk_distribution', 'Risk level distribution', ['level'])
anomaly_rate = Gauge('anomaly_rate', 'Anomaly detection rate')


class PerformanceTracker:
    """Track and analyze system performance metrics"""
    
    def __init__(self, config, db_session):
        self.config = config
        self.db = db_session
        
        # In-memory buffers for real-time metrics
        self.recent_errors = deque(maxlen=1000)
        self.recent_latencies = deque(maxlen=1000)
        self.recent_predictions = deque(maxlen=1000)
        
        # Counters
        self.total_predictions = 0
        self.total_anomalies = 0
        self.total_maintenance_alerts = 0
        
    def record_prediction(self, prediction_result, risk_score):
        """Record a prediction event"""
        self.total_predictions += 1
        
        # Store prediction data
        self.recent_predictions.append({
            'timestamp': datetime.utcnow(),
            'predicted_load': prediction_result['predicted_load'],
            'inference_time': prediction_result['inference_time_ms'],
            'risk_score': risk_score
        })
        
        # Update latency metrics
        self.recent_latencies.append(prediction_result['inference_time_ms'])
        
    def record_error(self, predicted, actual):
        """Record prediction error"""
        error = abs(predicted - actual)
        percentage_error = (error / actual * 100) if actual != 0 else 0
        
        self.recent_errors.append({
            'absolute': error,
            'percentage': percentage_error,
            'timestamp': datetime.utcnow()
        })
        
        # Update Prometheus metrics
        if len(self.recent_errors) >= 10:
            mae = np.mean([e['absolute'] for e in self.recent_errors])
            mape = np.mean([e['percentage'] for e in self.recent_errors])
            prediction_mae.set(mae)
            prediction_mape.set(mape)
    
    def record_anomaly(self):
        """Record anomaly detection"""
        self.total_anomalies += 1
        if self.total_predictions > 0:
            rate = self.total_anomalies / self.total_predictions
            anomaly_rate.set(rate)
    
    def record_maintenance_alert(self):
        """Record maintenance alert"""
        self.total_maintenance_alerts += 1
    
    def get_recent_metrics(self, hours=24):
        """Get performance metrics for recent time period"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Query database for predictions in time window
            predictions = Prediction.query.filter(
                Prediction.timestamp >= cutoff_time
            ).all()
            
            if not predictions:
                return {
                    'error': 'No data available for time period',
                    'hours': hours
                }
            
            # Calculate error metrics (only for predictions with actual values)
            predictions_with_actuals = [p for p in predictions if p.actual_load is not None]
            
            if predictions_with_actuals:
                errors = [p.absolute_error for p in predictions_with_actuals]
                percentage_errors = [p.percentage_error for p in predictions_with_actuals]
                
                mae = np.mean(errors)
                mse = np.mean([e**2 for e in errors])
                rmse = np.sqrt(mse)
                mape = np.mean(percentage_errors)
            else:
                mae = mse = rmse = mape = None
            
            # Risk distribution
            risk_counts = {
                'high': len([p for p in predictions if p.risk_level == 'high']),
                'medium': len([p for p in predictions if p.risk_level == 'medium']),
                'low': len([p for p in predictions if p.risk_level == 'low']),
                'minimal': len([p for p in predictions if p.risk_level == 'minimal'])
            }
            
            # Update Prometheus gauges
            for level, count in risk_counts.items():
                risk_distribution.labels(level=level).set(count)
            
            # Anomaly rate
            anomalies = len([p for p in predictions if p.anomaly_detected])
            anomaly_rate_val = anomalies / len(predictions) if predictions else 0
            
            # Maintenance alerts
            maintenance_alerts = len([p for p in predictions if p.maintenance_recommended])
            
            # Latency metrics from recent buffer
            if self.recent_latencies:
                avg_latency = np.mean(self.recent_latencies)
                p50_latency = np.percentile(self.recent_latencies, 50)
                p95_latency = np.percentile(self.recent_latencies, 95)
                p99_latency = np.percentile(self.recent_latencies, 99)
            else:
                avg_latency = p50_latency = p95_latency = p99_latency = None
            
            metrics = {
                'time_period': {
                    'hours': hours,
                    'start': cutoff_time.isoformat(),
                    'end': datetime.utcnow().isoformat()
                },
                'volume': {
                    'total_predictions': len(predictions),
                    'predictions_with_feedback': len(predictions_with_actuals)
                },
                'accuracy': {
                    'mae': round(mae, 2) if mae else None,
                    'mse': round(mse, 2) if mse else None,
                    'rmse': round(rmse, 2) if rmse else None,
                    'mape': round(mape, 2) if mape else None
                },
                'risk_distribution': risk_counts,
                'anomalies': {
                    'count': anomalies,
                    'rate': round(anomaly_rate_val, 4)
                },
                'maintenance': {
                    'alerts_generated': maintenance_alerts,
                    'alert_rate': round(maintenance_alerts / len(predictions), 4) if predictions else 0
                },
                'latency': {
                    'avg_ms': round(avg_latency, 2) if avg_latency else None,
                    'p50_ms': round(p50_latency, 2) if p50_latency else None,
                    'p95_ms': round(p95_latency, 2) if p95_latency else None,
                    'p99_ms': round(p99_latency, 2) if p99_latency else None
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def get_performance_trend(self, days=7):
        """Get performance trend over multiple days"""
        try:
            trends = []
            
            for day in range(days):
                start_time = datetime.utcnow() - timedelta(days=day+1)
                end_time = datetime.utcnow() - timedelta(days=day)
                
                predictions = Prediction.query.filter(
                    Prediction.timestamp >= start_time,
                    Prediction.timestamp < end_time,
                    Prediction.actual_load.isnot(None)
                ).all()
                
                if predictions:
                    mae = np.mean([p.absolute_error for p in predictions])
                    mape = np.mean([p.percentage_error for p in predictions])
                    anomaly_rate_val = len([p for p in predictions if p.anomaly_detected]) / len(predictions)
                else:
                    mae = mape = anomaly_rate_val = None
                
                trends.append({
                    'date': start_time.date().isoformat(),
                    'mae': round(mae, 2) if mae else None,
                    'mape': round(mape, 2) if mape else None,
                    'anomaly_rate': round(anomaly_rate_val, 4) if anomaly_rate_val else None,
                    'prediction_count': len(predictions)
                })
            
            return {
                'trends': list(reversed(trends)),
                'days': days
            }
            
        except Exception as e:
            logger.error(f"Error calculating trend: {str(e)}")
            raise
    
    def check_performance_alerts(self):
        """Check if performance metrics trigger alerts"""
        alerts = []
        
        # Get recent metrics
        metrics = self.get_recent_metrics(hours=1)
        
        # Check MAE threshold
        if metrics['accuracy']['mae'] and metrics['accuracy']['mae'] > self.config.MAE_ALERT_THRESHOLD:
            alerts.append({
                'type': 'high_mae',
                'severity': 'warning',
                'message': f'MAE {metrics["accuracy"]["mae"]} exceeds threshold {self.config.MAE_ALERT_THRESHOLD}',
                'value': metrics['accuracy']['mae']
            })
        
        # Check MAPE threshold
        if metrics['accuracy']['mape'] and metrics['accuracy']['mape'] > self.config.MAPE_ALERT_THRESHOLD:
            alerts.append({
                'type': 'high_mape',
                'severity': 'warning',
                'message': f'MAPE {metrics["accuracy"]["mape"]}% exceeds threshold {self.config.MAPE_ALERT_THRESHOLD}%',
                'value': metrics['accuracy']['mape']
            })
        
        # Check high anomaly rate
        if metrics['anomalies']['rate'] > 0.15:  # More than 15% anomalies
            alerts.append({
                'type': 'high_anomaly_rate',
                'severity': 'warning',
                'message': f'Anomaly rate {metrics["anomalies"]["rate"]*100:.1f}% is unusually high',
                'value': metrics['anomalies']['rate']
            })
        
        # Check latency
        if metrics['latency']['p99_ms'] and metrics['latency']['p99_ms'] > 200:
            alerts.append({
                'type': 'high_latency',
                'severity': 'warning',
                'message': f'P99 latency {metrics["latency"]["p99_ms"]}ms exceeds 200ms',
                'value': metrics['latency']['p99_ms']
            })
        
        return alerts
    
    def save_metrics_snapshot(self, model_version):
        """Save current metrics to database"""
        try:
            metrics = self.get_recent_metrics(hours=1)
            
            snapshot = PerformanceMetric(
                timestamp=datetime.utcnow(),
                model_version=model_version,
                mae=metrics['accuracy']['mae'],
                mse=metrics['accuracy']['mse'],
                rmse=metrics['accuracy']['rmse'],
                mape=metrics['accuracy']['mape'],
                predictions_count=metrics['volume']['total_predictions'],
                anomalies_detected=metrics['anomalies']['count'],
                maintenance_alerts=metrics['maintenance']['alerts_generated'],
                high_risk_count=metrics['risk_distribution']['high'],
                medium_risk_count=metrics['risk_distribution']['medium'],
                low_risk_count=metrics['risk_distribution']['low'],
                api_latency_ms=metrics['latency']['avg_ms'],
                prediction_latency_ms=metrics['latency']['p50_ms']
            )
            
            self.db.session.add(snapshot)
            self.db.session.commit()
            
            logger.info(f"Performance snapshot saved for model {model_version}")
            
        except Exception as e:
            logger.error(f"Error saving metrics snapshot: {str(e)}")
            self.db.session.rollback()
    
    def get_model_comparison(self, model_a, model_b, hours=24):
        """Compare performance of two model versions"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Get predictions for each model
            predictions_a = Prediction.query.filter(
                Prediction.model_version == model_a,
                Prediction.timestamp >= cutoff_time,
                Prediction.actual_load.isnot(None)
            ).all()
            
            predictions_b = Prediction.query.filter(
                Prediction.model_version == model_b,
                Prediction.timestamp >= cutoff_time,
                Prediction.actual_load.isnot(None)
            ).all()
            
            def calc_metrics(preds):
                if not preds:
                    return None
                return {
                    'mae': np.mean([p.absolute_error for p in preds]),
                    'mse': np.mean([p.absolute_error**2 for p in preds]),
                    'mape': np.mean([p.percentage_error for p in preds]),
                    'count': len(preds)
                }
            
            metrics_a = calc_metrics(predictions_a)
            metrics_b = calc_metrics(predictions_b)
            
            if metrics_a and metrics_b:
                # Calculate improvement
                mae_improvement = ((metrics_a['mae'] - metrics_b['mae']) / metrics_a['mae'] * 100)
                mape_improvement = ((metrics_a['mape'] - metrics_b['mape']) / metrics_a['mape'] * 100)
                
                # Statistical test (simplified)
                from scipy import stats
                errors_a = [p.absolute_error for p in predictions_a]
                errors_b = [p.absolute_error for p in predictions_b]
                t_stat, p_value = stats.ttest_ind(errors_a, errors_b)
                
                is_significant = p_value < 0.05
            else:
                mae_improvement = mape_improvement = None
                is_significant = False
                p_value = None
            
            return {
                'model_a': {
                    'version': model_a,
                    'metrics': metrics_a
                },
                'model_b': {
                    'version': model_b,
                    'metrics': metrics_b
                },
                'comparison': {
                    'mae_improvement_pct': round(mae_improvement, 2) if mae_improvement else None,
                    'mape_improvement_pct': round(mape_improvement, 2) if mape_improvement else None,
                    'statistically_significant': is_significant,
                    'p_value': round(p_value, 4) if p_value else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            raise