import numpy as np
import logging
from scipy import stats
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect anomalies in energy consumption patterns"""
    
    def __init__(self, config):
        self.config = config
        self.threshold = float(config.get('ANOMALY_THRESHOLD', 3.0))
        self.historical_window = deque(maxlen=168)  # 1 week of hourly data
        self.baseline_stats = {'mean': None, 'std': None}
        
    def update_baseline(self, values):
        """Update baseline statistics with scalar safety"""
        if not isinstance(values, (list, np.ndarray)):
            values = [values]
        
        self.historical_window.extend(values)
        
        if len(self.historical_window) >= 24:
            data = np.array(self.historical_window)
            # SCALAR FIX: Force stats to standard floats
            self.baseline_stats['mean'] = float(np.mean(data))
            self.baseline_stats['std'] = float(np.std(data))
            logger.debug(f"Updated baseline: mean={self.baseline_stats['mean']:.2f}, std={self.baseline_stats['std']:.2f}")
    
    def detect_point_anomaly(self, value):
        """Detect if a single value is anomalous using scalar-safe z-score"""
        if self.baseline_stats['mean'] is None or self.baseline_stats['std'] is None:
            return {
                'is_anomaly': False,
                'z_score': None,
                'reason': 'insufficient_baseline_data'
            }
        
        # SCALAR FIX: Ensure input and baseline stats are treated as standard floats
        val_float = float(np.array(value).flatten()[0])
        mean = float(self.baseline_stats['mean'])
        std = float(self.baseline_stats['std'])
        
        if std <= 1e-9: # Avoid division by zero with a tiny epsilon
            z_score = 0.0
        else:
            z_score = abs((val_float - mean) / std)
        
        # SCALAR FIX: Ensure z_score is a standard float before comparison
        z_score = float(z_score)
        is_anomaly = bool(z_score > self.threshold)
        
        return {
            'is_anomaly': is_anomaly,
            'z_score': z_score,
            'threshold': self.threshold,
            'deviation_from_mean': float(val_float - mean),
            'severity': self._get_severity(z_score)
        }
    
    def detect_sequence_anomaly(self, sequence):
        """Detect anomalies in a sequence with array safety"""
        if len(sequence) == 0:
            return {'total_anomalies': 0, 'anomalies': []}
            
        anomalies = []
        seq_array = np.array(sequence)
        
        # 1. Statistical outliers
        if np.std(seq_array) > 1e-9:
            z_scores = np.abs(stats.zscore(seq_array))
            # SCALAR FIX: Ensure comparison results are handled correctly
            statistical_anomalies = np.where(z_scores > self.threshold)[0]
        else:
            z_scores = np.zeros(len(seq_array))
            statistical_anomalies = np.array([])
        
        # 2. Sudden spikes
        if len(seq_array) > 1:
            derivatives = np.diff(seq_array)
            deriv_std = np.std(derivatives)
            derivative_threshold = float(deriv_std * 2.5 if deriv_std > 0 else 1.0)
            spike_indices = np.where(np.abs(derivatives) > derivative_threshold)[0]
        else:
            derivatives = np.array([])
            spike_indices = np.array([])
        
        # 3. Pattern deviation
        if len(seq_array) >= 24:
            last_24 = seq_array[-24:]
            daily_mean = float(np.mean(last_24))
            daily_std = float(np.std(last_24))
            
            for i, val in enumerate(last_24):
                if daily_std > 1e-9:
                    z = abs((float(val) - daily_mean) / daily_std)
                    if z > self.threshold:
                        anomalies.append({
                            'index': len(seq_array) - 24 + i,
                            'value': float(val),
                            'type': 'pattern_deviation',
                            'z_score': float(z)
                        })
        
        # Combine all anomalies
        for idx in statistical_anomalies:
            anomalies.append({
                'index': int(idx),
                'value': float(seq_array[idx]),
                'type': 'statistical_outlier',
                'z_score': float(z_scores[idx])
            })
        
        for idx in spike_indices:
            anomalies.append({
                'index': int(idx),
                'value': float(seq_array[idx]),
                'type': 'sudden_spike',
                'change': float(derivatives[idx])
            })
        
        unique_indices = set(a['index'] for a in anomalies)
        
        return {
            'total_anomalies': len(unique_indices),
            'anomaly_rate': float(len(unique_indices) / len(seq_array)),
            'anomalies': anomalies,
            'sequence_length': len(seq_array)
        }
    
    def detect_forecast_deviation(self, predicted, actual):
        """Detect deviation with scalar-safe comparisons"""
        p = float(np.array(predicted).flatten()[0])
        a = float(np.array(actual).flatten()[0])
        
        error = abs(a - p)
        percentage_error = (error / a * 100) if a != 0 else 0
        
        adaptive_threshold = max(50.0, a * 0.1)
        is_significant = bool(error > adaptive_threshold)
        
        return {
            'is_significant_deviation': is_significant,
            'absolute_error': float(error),
            'percentage_error': float(percentage_error),
            'predicted': p,
            'actual': a,
            'threshold_used': float(adaptive_threshold),
            'deviation_severity': self._get_deviation_severity(percentage_error)
        }
    
    def _get_severity(self, z_score):
        z = float(z_score)
        if z < self.threshold:
            return 'normal'
        elif z < self.threshold * 1.5:
            return 'moderate'
        elif z < self.threshold * 2:
            return 'high'
        else:
            return 'critical'
    
    def _get_deviation_severity(self, percentage_error):
        pe = float(percentage_error)
        if pe < 5: return 'low'
        elif pe < 10: return 'moderate'
        elif pe < 20: return 'high'
        else: return 'critical'

    def get_statistics(self):
        return {
            'baseline_mean': float(self.baseline_stats['mean']) if self.baseline_stats['mean'] is not None else None,
            'baseline_std': float(self.baseline_stats['std']) if self.baseline_stats['std'] is not None else None,
            'samples_in_window': len(self.historical_window),
            'threshold': float(self.threshold)
        }


class DriftDetector:
    """Detect model drift over time with scalar safety"""
    
    def __init__(self, config):
        self.config = config
        self.window_size = int(config.get('DRIFT_DETECTION_WINDOW', 168))
        self.recent_errors = deque(maxlen=self.window_size)
        self.baseline_error = None
        
    def add_error(self, error):
        # Force error to scalar float before storing
        self.recent_errors.append(float(np.array(error).flatten()[0]))
        if self.baseline_error is None and len(self.recent_errors) >= 24:
            self.baseline_error = float(np.mean(list(self.recent_errors)[:24]))
    
    def detect_drift(self):
        if len(self.recent_errors) < self.window_size // 2:
            return {'drift_detected': False, 'reason': 'insufficient_data', 'samples': len(self.recent_errors)}
        
        if self.baseline_error is None or self.baseline_error == 0:
            return {'drift_detected': False, 'reason': 'no_baseline'}
        
        recent_window = list(self.recent_errors)[-24:]
        recent_mean_error = float(np.mean(recent_window))
        
        # SCALAR FIX: Ensure comparison values are standard floats
        error_increase = float((recent_mean_error - self.baseline_error) / self.baseline_error)
        
        baseline_sample = list(self.recent_errors)[:48]
        recent_sample = recent_window
        
        statistical_drift = False
        ks_stat, p_val = None, None
        
        if len(baseline_sample) >= 20 and len(recent_sample) >= 20:
            ks_result = stats.ks_2samp(baseline_sample, recent_sample)
            ks_stat = float(ks_result.statistic)
            p_val = float(ks_result.pvalue)
            statistical_drift = bool(p_val < 0.05)
        
        threshold_drift = bool(error_increase > 0.25)
        drift_detected = threshold_drift or statistical_drift
        
        result = {
            'drift_detected': drift_detected,
            'baseline_error': float(self.baseline_error),
            'recent_error': recent_mean_error,
            'error_increase_pct': float(error_increase * 100),
            'ks_statistic': ks_stat,
            'p_value': p_val,
            'recommendation': 'retrain_model' if drift_detected else 'continue_monitoring'
        }
        
        if drift_detected:
            logger.warning(f"Model drift detected: {error_increase*100:.1f}% error increase")
        
        return result
    
    def reset_baseline(self):
        if len(self.recent_errors) >= 24:
            self.baseline_error = float(np.mean(list(self.recent_errors)[-24:]))
            logger.info("Drift detector baseline reset")