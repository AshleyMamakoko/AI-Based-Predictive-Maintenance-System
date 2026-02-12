import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RiskCalculator:
    """Calculate maintenance risk scores based on predictions and anomalies with scalar safety"""
    
    def __init__(self, config):
        self.config = config
        self.risk_factors = {
            'forecast_error': 0.3,
            'load_magnitude': 0.2,
            'anomaly_score': 0.3,
            'recent_history': 0.2
        }
        
    def calculate_risk(self, prediction_data, anomaly_data, historical_data=None):
        """
        Calculate comprehensive risk score with bulletproof scalar conversion
        """
        risk_components = {}
        
        # Ensure input predicted_load is a scalar float
        p_load = float(np.array(prediction_data.get('predicted_load', 0)).flatten()[0])
        
        # 1. Forecast error component
        if 'actual_load' in prediction_data and prediction_data['actual_load'] is not None:
            actual_load = float(np.array(prediction_data['actual_load']).flatten()[0])
            forecast_error = abs(p_load - actual_load)
            error_percentage = (forecast_error / actual_load * 100) if actual_load != 0 else 0
            
            # Normalize to 0-1 scale
            error_risk = min(error_percentage / 20.0, 1.0)
            risk_components['forecast_error'] = float(error_risk)
        else:
            risk_components['forecast_error'] = 0.0
        
        # 2. Load magnitude component
        if historical_data is not None and len(historical_data) > 0:
            historical_max = float(np.max(historical_data))
            load_ratio = p_load / historical_max if historical_max > 0 else 0
        else:
            load_ratio = min(p_load / 2000, 1.0)
        
        # Sigmoid function for smooth scaling - Force to float
        load_risk = 1 / (1 + np.exp(-5 * (load_ratio - 0.7)))
        risk_components['load_magnitude'] = float(load_risk)
        
        # 3. Anomaly component - Force is_anomaly to standard bool
        is_anom = bool(np.array(anomaly_data.get('is_anomaly', False)).any())
        if is_anom:
            z_score = float(np.array(anomaly_data.get('z_score', 0)).flatten()[0])
            anomaly_risk = min(z_score / 5.0, 1.0)
            risk_components['anomaly_score'] = float(anomaly_risk)
        else:
            risk_components['anomaly_score'] = 0.0
        
        # 4. Recent history component
        if historical_data is not None and len(historical_data) >= 24:
            recent_data = np.array(historical_data[-24:])
            volatility = float(np.std(recent_data))
            mean_load = float(np.mean(recent_data))
            
            cv = volatility / mean_load if mean_load > 0 else 0
            volatility_risk = min(cv / 0.3, 1.0)
            risk_components['recent_history'] = float(volatility_risk)
        else:
            risk_components['recent_history'] = 0.0
        
        # Calculate weighted risk score - Force to float
        total_risk = float(sum(
            (float(risk_components.get(component, 0.0) or 0.0)) * self.risk_factors.get(component, 0.0)
            for component in self.risk_factors
        ))
        
        # Classify risk level
        risk_level = self._classify_risk_level(total_risk)
        
        # Maintenance recommendation threshold with fallback
        medium_threshold = float(getattr(self.config, 'MEDIUM_RISK_THRESHOLD', 0.50))
        m_rec = bool(total_risk >= medium_threshold)
        
        maintenance_priority = self._get_maintenance_priority(total_risk)
        
        # Estimate failure probability
        failure_probability = self._estimate_failure_probability(
            total_risk,
            risk_components
        )
        
        return {
            'risk_score': total_risk,
            'risk_level': risk_level,
            'risk_components': risk_components,
            'maintenance_recommended': m_rec,
            'maintenance_priority': maintenance_priority,
            'estimated_failure_probability': float(failure_probability),
            'recommendations': self._generate_recommendations(total_risk, risk_components)
        }
    
    def _classify_risk_level(self, risk_score):
        """Classify risk into categories with attribute safety"""
        score = float(risk_score)
        
        high = float(getattr(self.config, 'HIGH_RISK_THRESHOLD', 0.75))
        medium = float(getattr(self.config, 'MEDIUM_RISK_THRESHOLD', 0.50))
        low = float(getattr(self.config, 'LOW_RISK_THRESHOLD', 0.25))

        if score >= high:
            return 'high'
        elif score >= medium:
            return 'medium'
        elif score >= low:
            return 'low'
        else:
            return 'minimal'

    def _get_maintenance_priority(self, risk_score):
        """Determine priority with attribute safety"""
        score = float(risk_score)
        high = float(getattr(self.config, 'HIGH_RISK_THRESHOLD', 0.75))
        medium = float(getattr(self.config, 'MEDIUM_RISK_THRESHOLD', 0.50))

        if score >= 0.85:
            return 'critical'
        elif score >= high:
            return 'high'
        elif score >= medium:
            return 'medium'
        else:
            return None
    
    def _estimate_failure_probability(self, risk_score, components):
        base_prob = float(risk_score) * 0.8
        if float(components['anomaly_score']) > 0.7:
            base_prob += 0.15
        if float(components['forecast_error']) > 0.8:
            base_prob += 0.1
        return float(min(base_prob, 0.95))
    
    def _generate_recommendations(self, risk_score, components):
        recommendations = []
        score = float(risk_score)
        high_threshold = float(getattr(self.config, 'HIGH_RISK_THRESHOLD', 0.75))
        
        if score >= high_threshold:
            recommendations.append({
                'action': 'schedule_immediate_inspection',
                'reason': 'High risk of equipment failure detected',
                'urgency': 'high',
                'timeframe': 'within_24_hours'
            })
        
        if float(components['anomaly_score']) > 0.6:
            recommendations.append({
                'action': 'investigate_load_anomaly',
                'reason': 'Unusual load pattern detected',
                'urgency': 'medium',
                'timeframe': 'within_48_hours'
            })
        
        if not recommendations:
            recommendations.append({
                'action': 'continue_monitoring',
                'reason': 'System operating within normal parameters',
                'urgency': 'low',
                'timeframe': 'routine'
            })
        
        return recommendations


class MaintenanceScheduler:
    """Schedule maintenance with scalar safety and attribute fallbacks"""
    
    def __init__(self, config):
        self.config = config
        self.scheduled_maintenance = []
        
    def should_schedule_maintenance(self, risk_assessment, asset_id, last_maintenance=None):
        risk_score = float(np.array(risk_assessment['risk_score']).flatten()[0])
        
        if self._is_already_scheduled(asset_id):
            return {'should_schedule': False, 'reason': 'already_scheduled'}
        
        if last_maintenance:
            cooldown = float(getattr(self.config, 'MAINTENANCE_COOLDOWN_HOURS', 24))
            hours_passed = (datetime.utcnow() - last_maintenance).total_seconds() / 3600
            if hours_passed < cooldown:
                return {
                    'should_schedule': False, 
                    'reason': 'maintenance_cooldown',
                    'hours_remaining': cooldown - hours_passed
                }

        critical_limit = float(getattr(self.config, 'CRITICAL_FAILURE_PROBABILITY', 0.8))
        high_limit = float(getattr(self.config, 'HIGH_RISK_THRESHOLD', 0.75))
        medium_limit = float(getattr(self.config, 'MEDIUM_RISK_THRESHOLD', 0.5))
        maintenance_window = float(getattr(self.config, 'MAINTENANCE_WINDOW_HOURS', 72))

        if risk_score >= critical_limit:
            return {
                'should_schedule': True,
                'reason': 'critical_risk',
                'urgency': 'immediate',
                'recommended_date': (datetime.utcnow() + timedelta(hours=4)).isoformat()
            }
        elif risk_score >= high_limit:
            return {
                'should_schedule': True,
                'reason': 'high_risk',
                'urgency': 'high',
                'recommended_date': (datetime.utcnow() + timedelta(hours=24)).isoformat()
            }
        elif risk_score >= medium_limit:
            return {
                'should_schedule': True,
                'reason': 'medium_risk',
                'urgency': 'medium',
                'recommended_date': (datetime.utcnow() + timedelta(hours=maintenance_window)).isoformat()
            }
        
        return {
            'should_schedule': False, 
            'reason': 'risk_below_threshold',
            'continue_monitoring': True
        }

    def _is_already_scheduled(self, asset_id):
        return any(m['asset_id'] == asset_id and m['status'] == 'scheduled' 
                   for m in self.scheduled_maintenance)