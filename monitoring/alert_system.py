import logging
import requests
from datetime import datetime
from database.models import SystemAlert
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AlertSystem:
    """Manage system alerts and notifications"""
    
    def __init__(self, config, db_session):
        self.config = config
        self.db = db_session
        # Use .get() to avoid AttributeError: 'Config' object has no attribute 'ALERT_EMAIL_ENABLED'
        self.email_enabled = config.get('ALERT_EMAIL_ENABLED', False)
        self.webhook_url = config.get('ALERT_WEBHOOK_URL', None)
        
    def create_alert(
        self,
        alert_type: str,
        severity: str,
        title: str,
        message: str,
        asset_id: Optional[str] = None,
        metric_value: Optional[float] = None,
        threshold_value: Optional[float] = None
    ) -> SystemAlert:
        """Create and store a new alert"""
        try:
            # Create alert record
            alert = SystemAlert(
                timestamp=datetime.utcnow(),
                alert_type=alert_type,
                severity=severity,
                title=title,
                message=message,
                asset_id=asset_id,
                metric_value=metric_value,
                threshold_value=threshold_value,
                acknowledged=False,
                resolved=False
            )
            
            self.db.session.add(alert)
            self.db.session.commit()
            
            # Log alert
            log_method = {
                'critical': logger.critical,
                'warning': logger.warning,
                'info': logger.info
            }.get(severity, logger.info)
            
            log_method(f"Alert created: {title} - {message}")
            
            # Send notifications
            self._send_notifications(alert)
            
            return alert
            
        except Exception as e:
            logger.error(f"Failed to create alert: {str(e)}")
            self.db.session.rollback()
            raise
    
    def _send_notifications(self, alert: SystemAlert):
        """Send alert notifications via configured channels"""
        
        # Webhook notification
        if self.webhook_url and alert.severity in ['critical', 'warning']:
            try:
                self._send_webhook(alert)
            except Exception as e:
                logger.error(f"Webhook notification failed: {str(e)}")
        
        # Email notification (if enabled)
        if self.email_enabled and alert.severity == 'critical':
            try:
                self._send_email(alert)
            except Exception as e:
                logger.error(f"Email notification failed: {str(e)}")
    
    def _send_webhook(self, alert: SystemAlert):
        """Send alert to webhook endpoint"""
        payload = {
            'alert_id': alert.id,
            'timestamp': alert.timestamp.isoformat(),
            'type': alert.alert_type,
            'severity': alert.severity,
            'title': alert.title,
            'message': alert.message,
            'asset_id': alert.asset_id,
            'metric_value': alert.metric_value
        }
        
        response = requests.post(
            self.webhook_url,
            json=payload,
            timeout=10
        )
        
        if response.status_code != 200:
            logger.warning(f"Webhook returned status {response.status_code}")
        else:
            logger.info(f"Alert {alert.id} sent to webhook")
    
    def _send_email(self, alert: SystemAlert):
        """Send alert via email"""
        logger.info(f"Would send email for alert: {alert.title}")
    
    def acknowledge_alert(self, alert_id: int, acknowledged_by: str) -> bool:
        """Mark alert as acknowledged"""
        try:
            alert = SystemAlert.query.get(alert_id)
            if not alert:
                return False
            
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.utcnow()
            
            self.db.session.commit()
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {str(e)}")
            self.db.session.rollback()
            return False
    
    def resolve_alert(self, alert_id: int) -> bool:
        """Mark alert as resolved"""
        try:
            alert = SystemAlert.query.get(alert_id)
            if not alert:
                return False
            
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            
            self.db.session.commit()
            logger.info(f"Alert {alert_id} resolved")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to resolve alert: {str(e)}")
            self.db.session.rollback()
            return False
    
    def get_active_alerts(self, severity=None, asset_id=None, limit=50):
        """Get currently active (unresolved) alerts"""
        query = SystemAlert.query.filter_by(resolved=False)
        
        if severity:
            query = query.filter_by(severity=severity)
        if asset_id:
            query = query.filter_by(asset_id=asset_id)
        
        alerts = query.order_by(
            SystemAlert.timestamp.desc()
        ).limit(limit).all()
        
        return alerts
    
    def get_alert_summary(self, hours=24) -> Dict:
        """Get summary of alerts in time period"""
        from datetime import timedelta
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        alerts = SystemAlert.query.filter(
            SystemAlert.timestamp >= cutoff_time
        ).all()
        
        summary = {
            'total': len(alerts),
            'by_severity': {
                'critical': len([a for a in alerts if a.severity == 'critical']),
                'warning': len([a for a in alerts if a.severity == 'warning']),
                'info': len([a for a in alerts if a.severity == 'info'])
            },
            'by_type': {
                'performance': len([a for a in alerts if a.alert_type == 'performance']),
                'maintenance': len([a for a in alerts if a.alert_type == 'maintenance']),
                'system': len([a for a in alerts if a.alert_type == 'system'])
            },
            'acknowledged': len([a for a in alerts if a.acknowledged]),
            'resolved': len([a for a in alerts if a.resolved])
        }
        
        return summary
    
    def check_and_create_alerts(self, metrics: Dict):
        """Check metrics and create alerts if thresholds exceeded"""
        
        # MAE threshold alert
        mae_threshold = self.config.get('MAE_ALERT_THRESHOLD', 100.0)
        if metrics.get('mae') and metrics['mae'] > mae_threshold:
            self.create_alert(
                alert_type='performance',
                severity='warning',
                title='High Prediction Error',
                message=f'MAE {metrics["mae"]:.1f} MW exceeds threshold {mae_threshold} MW',
                metric_value=metrics['mae'],
                threshold_value=mae_threshold
            )
        
        # MAPE threshold alert
        mape_threshold = self.config.get('MAPE_ALERT_THRESHOLD', 10.0)
        if metrics.get('mape') and metrics['mape'] > mape_threshold:
            self.create_alert(
                alert_type='performance',
                severity='warning',
                title='High Percentage Error',
                message=f'MAPE {metrics["mape"]:.1f}% exceeds threshold {mape_threshold}%',
                metric_value=metrics['mape'],
                threshold_value=mape_threshold
            )
        
        # High anomaly rate alert
        if metrics.get('anomaly_rate') and metrics['anomaly_rate'] > 0.15:
            self.create_alert(
                alert_type='system',
                severity='warning',
                title='High Anomaly Rate',
                message=f'Anomaly rate {metrics["anomaly_rate"]*100:.1f}% is unusually high',
                metric_value=metrics['anomaly_rate'],
                threshold_value=0.15
            )


class AlertRule:
    """Define custom alert rules"""
    
    def __init__(self, name, condition, severity, message_template):
        self.name = name
        self.condition = condition
        self.severity = severity
        self.message_template = message_template
    
    def evaluate(self, data: Dict) -> Optional[Dict]:
        """Evaluate rule against data"""
        try:
            if self.condition(data):
                return {
                    'title': self.name,
                    'severity': self.severity,
                    'message': self.message_template.format(**data)
                }
        except Exception as e:
            logger.error(f"Error evaluating rule {self.name}: {str(e)}")
        
        return None


class RuleEngine:
    """Manage and evaluate alert rules"""
    
    def __init__(self, alert_system: AlertSystem):
        self.alert_system = alert_system
        self.rules = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alert rules"""
        
        self.add_rule(AlertRule(
            name='High Risk Score Detected',
            condition=lambda d: d.get('risk_score', 0) > 0.85,
            severity='critical',
            message_template='Asset {asset_id} has critical risk score: {risk_score:.2f}'
        ))
        
        self.add_rule(AlertRule(
            name='Consecutive High Prediction Errors',
            condition=lambda d: len(d.get('recent_errors', [])) >= 5 and 
                               all(e > 100 for e in d.get('recent_errors', [])[-5:]),
            severity='warning',
            message_template='Model showing consistent high errors (MAE > 100 MW)'
        ))
        
        self.add_rule(AlertRule(
            name='Rapid Load Change',
            condition=lambda d: abs(d.get('load_change_rate', 0)) > 500,
            severity='warning',
            message_template='Rapid load change detected: {load_change_rate:.1f} MW/hour'
        ))
    
    def add_rule(self, rule: AlertRule):
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def evaluate_all(self, data: Dict):
        """Evaluate all rules against data"""
        for rule in self.rules:
            alert_data = rule.evaluate(data)
            if alert_data:
                self.alert_system.create_alert(
                    alert_type='system',
                    severity=alert_data['severity'],
                    title=alert_data['title'],
                    message=alert_data['message'],
                    asset_id=data.get('asset_id')
                )