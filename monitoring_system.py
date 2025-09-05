"""monitoring_system.py
Advanced monitoring and metrics system for DRM.

Features:
- Real-time performance metrics
- Rule evolution tracking
- Anomaly detection
- Alert system
- Dashboard generation
- Performance analytics
- Trend analysis
"""

import time
import json
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import statistics
import logging
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

@dataclass
class Alert:
    """Alert definition"""
    id: str
    level: AlertLevel
    message: str
    timestamp: float
    rule_id: Optional[str] = None
    metric_name: Optional[str] = None
    threshold: Optional[float] = None
    current_value: Optional[float] = None
    acknowledged: bool = False

class Metric:
    """Base metric class"""
    
    def __init__(self, name: str, metric_type: MetricType, description: str = ""):
        self.name = name
        self.type = metric_type
        self.description = description
        self.data_points: deque = deque(maxlen=10000)  # Keep last 10k points
        self.tags: Dict[str, str] = {}
        self._lock = threading.Lock()
    
    def add_point(self, value: float, tags: Dict[str, str] = None):
        """Add a data point"""
        with self._lock:
            point = MetricPoint(
                timestamp=time.time(),
                value=float(value),
                tags=tags or {}
            )
            self.data_points.append(point)
    
    def get_recent_values(self, seconds: int = 300) -> List[float]:
        """Get values from last N seconds"""
        cutoff = time.time() - seconds
        with self._lock:
            return [p.value for p in self.data_points if p.timestamp >= cutoff]
    
    def get_current_value(self) -> Optional[float]:
        """Get most recent value"""
        with self._lock:
            return self.data_points[-1].value if self.data_points else None
    
    def get_statistics(self, seconds: int = 300) -> Dict[str, float]:
        """Get statistical summary"""
        values = self.get_recent_values(seconds)
        if not values:
            return {}
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0
        }

class Counter(Metric):
    """Counter metric - monotonically increasing"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, MetricType.COUNTER, description)
        self._value = 0.0
    
    def increment(self, amount: float = 1.0, tags: Dict[str, str] = None):
        """Increment counter"""
        self._value += amount
        self.add_point(self._value, tags)
    
    def reset(self):
        """Reset counter to zero"""
        self._value = 0.0
        self.add_point(self._value)

class Gauge(Metric):
    """Gauge metric - can go up or down"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, MetricType.GAUGE, description)
    
    def set(self, value: float, tags: Dict[str, str] = None):
        """Set gauge value"""
        self.add_point(value, tags)

class Histogram(Metric):
    """Histogram metric - tracks distribution of values"""
    
    def __init__(self, name: str, description: str = "", buckets: List[float] = None):
        super().__init__(name, MetricType.HISTOGRAM, description)
        self.buckets = buckets or [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]
        self.bucket_counts = defaultdict(int)
    
    def observe(self, value: float, tags: Dict[str, str] = None):
        """Observe a value"""
        self.add_point(value, tags)
        
        # Update bucket counts
        for bucket in self.buckets:
            if value <= bucket:
                self.bucket_counts[bucket] += 1
    
    def get_percentile(self, percentile: float, seconds: int = 300) -> Optional[float]:
        """Get percentile value"""
        values = self.get_recent_values(seconds)
        if not values:
            return None
        
        values.sort()
        index = int(len(values) * percentile / 100)
        return values[min(index, len(values) - 1)]

class Timer(Metric):
    """Timer metric - measures duration"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, MetricType.TIMER, description)
    
    def time(self, tags: Dict[str, str] = None):
        """Context manager for timing"""
        return TimerContext(self, tags)
    
    def record(self, duration: float, tags: Dict[str, str] = None):
        """Record a duration"""
        self.add_point(duration, tags)

class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, timer: Timer, tags: Dict[str, str] = None):
        self.timer = timer
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.timer.record(duration, self.tags)

class AnomalyDetector:
    """Detects anomalies in metric data"""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Standard deviations for anomaly threshold
    
    def detect_anomalies(self, metric: Metric, window_seconds: int = 300) -> List[Dict[str, Any]]:
        """Detect anomalies in metric data"""
        values = metric.get_recent_values(window_seconds)
        if len(values) < 10:  # Need minimum data points
            return []
        
        anomalies = []
        
        # Statistical anomaly detection
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        
        if std_val > 0:
            threshold = self.sensitivity * std_val
            
            for i, value in enumerate(values[-10:]):  # Check last 10 points
                if abs(value - mean_val) > threshold:
                    anomalies.append({
                        'type': 'statistical',
                        'value': value,
                        'expected_range': (mean_val - threshold, mean_val + threshold),
                        'deviation': abs(value - mean_val) / std_val,
                        'timestamp': time.time() - (len(values) - i) * 30  # Approximate
                    })
        
        # Trend-based anomaly detection
        if len(values) >= 20:
            recent_trend = self._calculate_trend(values[-10:])
            historical_trend = self._calculate_trend(values[-20:-10])
            
            if abs(recent_trend - historical_trend) > 0.5:  # Significant trend change
                anomalies.append({
                    'type': 'trend_change',
                    'recent_trend': recent_trend,
                    'historical_trend': historical_trend,
                    'change': recent_trend - historical_trend,
                    'timestamp': time.time()
                })
        
        return anomalies
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of values"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_vals = list(range(n))
        
        # Simple linear regression slope
        x_mean = sum(x_vals) / n
        y_mean = sum(values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, values))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)
        
        return numerator / denominator if denominator != 0 else 0.0

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_rules: List[Dict[str, Any]] = []
        self.callbacks: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
    
    def add_alert_rule(self, metric_name: str, condition: str, threshold: float,
                      level: AlertLevel, message: str):
        """Add an alert rule"""
        rule = {
            'metric_name': metric_name,
            'condition': condition,  # 'gt', 'lt', 'eq'
            'threshold': threshold,
            'level': level,
            'message': message
        }
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule for {metric_name}: {condition} {threshold}")
    
    def register_callback(self, callback: Callable[[Alert], None]):
        """Register alert callback"""
        self.callbacks.append(callback)
    
    def check_alerts(self, metrics: Dict[str, Metric]):
        """Check all alert rules against current metrics"""
        for rule in self.alert_rules:
            metric_name = rule['metric_name']
            if metric_name not in metrics:
                continue
            
            metric = metrics[metric_name]
            current_value = metric.get_current_value()
            
            if current_value is None:
                continue
            
            # Check condition
            triggered = False
            condition = rule['condition']
            threshold = rule['threshold']
            
            if condition == 'gt' and current_value > threshold:
                triggered = True
            elif condition == 'lt' and current_value < threshold:
                triggered = True
            elif condition == 'eq' and abs(current_value - threshold) < 1e-6:
                triggered = True
            
            if triggered:
                alert = Alert(
                    id=f"{metric_name}_{int(time.time())}",
                    level=rule['level'],
                    message=rule['message'],
                    timestamp=time.time(),
                    metric_name=metric_name,
                    threshold=threshold,
                    current_value=current_value
                )
                
                self._fire_alert(alert)
    
    def _fire_alert(self, alert: Alert):
        """Fire an alert"""
        with self._lock:
            self.alerts.append(alert)
        
        logger.warning(f"ALERT [{alert.level.value}]: {alert.message}")
        
        # Call registered callbacks
        for callback in self.callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get unacknowledged alerts"""
        with self._lock:
            return [a for a in self.alerts if not a.acknowledged]
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        with self._lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    break

class DRMMonitor:
    """Main monitoring system for DRM"""
    
    def __init__(self, drm_system):
        self.drm_system = drm_system
        self.metrics: Dict[str, Metric] = {}
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager()
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Initialize core metrics
        self._initialize_metrics()
        self._setup_default_alerts()
    
    def _initialize_metrics(self):
        """Initialize core DRM metrics"""
        # Rule metrics
        self.metrics['rule_count'] = Gauge('rule_count', 'Total number of rules')
        self.metrics['active_rule_count'] = Gauge('active_rule_count', 'Number of active rules')
        self.metrics['quarantined_rule_count'] = Gauge('quarantined_rule_count', 'Number of quarantined rules')
        
        # Performance metrics
        self.metrics['avg_rule_performance'] = Gauge('avg_rule_performance', 'Average rule performance')
        self.metrics['diversity_score'] = Gauge('diversity_score', 'Rule diversity score')
        self.metrics['system_entropy'] = Gauge('system_entropy', 'System entropy')
        
        # Activity metrics
        self.metrics['rule_activations'] = Counter('rule_activations', 'Total rule activations')
        self.metrics['rule_mutations'] = Counter('rule_mutations', 'Total rule mutations')
        self.metrics['conflicts_detected'] = Counter('conflicts_detected', 'Conflicts detected')
        self.metrics['rules_generated'] = Counter('rules_generated', 'Rules generated')
        
        # Timing metrics
        self.metrics['cycle_duration'] = Timer('cycle_duration', 'DRM cycle execution time')
        self.metrics['validation_duration'] = Timer('validation_duration', 'Rule validation time')
        
        # Quality metrics
        self.metrics['validation_success_rate'] = Gauge('validation_success_rate', 'Validation success rate')
        self.metrics['rule_success_rate'] = Histogram('rule_success_rate', 'Distribution of rule success rates')
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        # Critical alerts
        self.alert_manager.add_alert_rule(
            'active_rule_count', 'lt', 1.0, AlertLevel.CRITICAL,
            'No active rules remaining in system'
        )
        
        # Warning alerts
        self.alert_manager.add_alert_rule(
            'avg_rule_performance', 'lt', 0.3, AlertLevel.WARNING,
            'Average rule performance below threshold'
        )
        
        self.alert_manager.add_alert_rule(
            'diversity_score', 'lt', 0.2, AlertLevel.WARNING,
            'Rule diversity too low - risk of stagnation'
        )
        
        # Info alerts
        self.alert_manager.add_alert_rule(
            'quarantined_rule_count', 'gt', 10.0, AlertLevel.INFO,
            'High number of quarantined rules'
        )

    def start_monitoring(self, interval: float = 30.0):
        """Start background monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Started DRM monitoring with {interval}s interval")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Stopped DRM monitoring")

    def _monitoring_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self.collect_metrics()
                self.check_anomalies()
                self.alert_manager.check_alerts(self.metrics)
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)

    def collect_metrics(self):
        """Collect current metrics from DRM system"""
        stats = self.drm_system.get_stats()

        # Update rule counts
        self.metrics['rule_count'].set(stats['total_rules'])
        self.metrics['active_rule_count'].set(stats['active_rules'])
        self.metrics['quarantined_rule_count'].set(stats['quarantined_rules'])

        # Update performance metrics
        self.metrics['avg_rule_performance'].set(stats['avg_performance'])
        self.metrics['diversity_score'].set(stats['diversity_score'])

        # Calculate system entropy
        distribution = self.drm_system.get_distribution()
        entropy = self._calculate_entropy(distribution)
        self.metrics['system_entropy'].set(entropy)

        # Update success rates
        active_rules = self.drm_system.get_active_rules()
        if active_rules:
            success_rates = [rule.get_success_rate() for rule in active_rules.values()]
            avg_success_rate = sum(success_rates) / len(success_rates)
            self.metrics['validation_success_rate'].set(avg_success_rate)

            # Record individual success rates in histogram
            for rate in success_rates:
                self.metrics['rule_success_rate'].observe(rate)

    def _calculate_entropy(self, distribution: Dict[str, float]) -> float:
        """Calculate Shannon entropy of rule distribution"""
        if not distribution:
            return 0.0

        entropy = 0.0
        for prob in distribution.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)

        return entropy

    def check_anomalies(self):
        """Check for anomalies in metrics"""
        for metric_name, metric in self.metrics.items():
            anomalies = self.anomaly_detector.detect_anomalies(metric)

            for anomaly in anomalies:
                alert = Alert(
                    id=f"anomaly_{metric_name}_{int(time.time())}",
                    level=AlertLevel.WARNING,
                    message=f"Anomaly detected in {metric_name}: {anomaly['type']}",
                    timestamp=time.time(),
                    metric_name=metric_name
                )
                self.alert_manager._fire_alert(alert)

    def record_rule_activation(self, rule_id: str):
        """Record rule activation"""
        self.metrics['rule_activations'].increment(tags={'rule_id': rule_id})

    def record_rule_mutation(self, rule_id: str, mutation_type: str):
        """Record rule mutation"""
        self.metrics['rule_mutations'].increment(tags={'rule_id': rule_id, 'type': mutation_type})

    def record_conflict_detection(self, conflict_count: int):
        """Record conflict detection"""
        self.metrics['conflicts_detected'].increment(conflict_count)

    def record_rule_generation(self, count: int = 1):
        """Record rule generation"""
        self.metrics['rules_generated'].increment(count)

    def time_cycle(self):
        """Context manager for timing DRM cycles"""
        return self.metrics['cycle_duration'].time()

    def time_validation(self):
        """Context manager for timing validations"""
        return self.metrics['validation_duration'].time()

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display"""
        dashboard_data = {
            'timestamp': time.time(),
            'metrics': {},
            'alerts': {
                'active': len(self.alert_manager.get_active_alerts()),
                'recent': [asdict(alert) for alert in self.alert_manager.alerts[-10:]]
            },
            'system_health': self._calculate_system_health()
        }

        # Collect metric summaries
        for name, metric in self.metrics.items():
            current_value = metric.get_current_value()
            stats = metric.get_statistics(300)  # Last 5 minutes

            dashboard_data['metrics'][name] = {
                'current': current_value,
                'type': metric.type.value,
                'stats': stats,
                'description': metric.description
            }

            # Add percentiles for histograms
            if isinstance(metric, Histogram):
                dashboard_data['metrics'][name]['percentiles'] = {
                    'p50': metric.get_percentile(50),
                    'p90': metric.get_percentile(90),
                    'p95': metric.get_percentile(95),
                    'p99': metric.get_percentile(99)
                }

        return dashboard_data

    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health score"""
        health_score = 1.0
        health_factors = {}

        # Rule count health
        active_rules = self.metrics['active_rule_count'].get_current_value() or 0
        if active_rules < 1:
            health_score *= 0.0  # Critical
            health_factors['rules'] = 'critical'
        elif active_rules < 3:
            health_score *= 0.5
            health_factors['rules'] = 'warning'
        else:
            health_factors['rules'] = 'good'

        # Performance health
        avg_perf = self.metrics['avg_rule_performance'].get_current_value() or 0
        if avg_perf < 0.2:
            health_score *= 0.3
            health_factors['performance'] = 'poor'
        elif avg_perf < 0.5:
            health_score *= 0.7
            health_factors['performance'] = 'fair'
        else:
            health_factors['performance'] = 'good'

        # Diversity health
        diversity = self.metrics['diversity_score'].get_current_value() or 0
        if diversity < 0.1:
            health_score *= 0.5
            health_factors['diversity'] = 'low'
        elif diversity < 0.3:
            health_score *= 0.8
            health_factors['diversity'] = 'moderate'
        else:
            health_factors['diversity'] = 'good'

        # Alert health
        active_alerts = len(self.alert_manager.get_active_alerts())
        if active_alerts > 5:
            health_score *= 0.6
            health_factors['alerts'] = 'many'
        elif active_alerts > 0:
            health_score *= 0.9
            health_factors['alerts'] = 'some'
        else:
            health_factors['alerts'] = 'none'

        return {
            'score': health_score,
            'level': self._health_level(health_score),
            'factors': health_factors
        }

    def _health_level(self, score: float) -> str:
        """Convert health score to level"""
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'fair'
        elif score >= 0.2:
            return 'poor'
        else:
            return 'critical'

    def export_metrics(self, format: str = 'json', time_range: int = 3600) -> str:
        """Export metrics data"""
        cutoff = time.time() - time_range
        export_data = {
            'timestamp': time.time(),
            'time_range_seconds': time_range,
            'metrics': {}
        }

        for name, metric in self.metrics.items():
            with metric._lock:
                recent_points = [
                    {'timestamp': p.timestamp, 'value': p.value, 'tags': p.tags}
                    for p in metric.data_points
                    if p.timestamp >= cutoff
                ]

            export_data['metrics'][name] = {
                'type': metric.type.value,
                'description': metric.description,
                'data_points': recent_points,
                'statistics': metric.get_statistics(time_range)
            }

        if format == 'json':
            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def generate_report(self, time_range: int = 3600) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'time_range_hours': time_range / 3600,
            'summary': {},
            'metrics_analysis': {},
            'alerts_summary': {},
            'recommendations': []
        }

        # Summary statistics
        stats = self.drm_system.get_stats()
        report['summary'] = {
            'total_rules': stats['total_rules'],
            'active_rules': stats['active_rules'],
            'quarantined_rules': stats['quarantined_rules'],
            'diversity_score': stats['diversity_score'],
            'avg_performance': stats['avg_performance'],
            'system_health': self._calculate_system_health()
        }

        # Metrics analysis
        for name, metric in self.metrics.items():
            metric_stats = metric.get_statistics(time_range)
            if metric_stats:
                report['metrics_analysis'][name] = metric_stats

        # Alerts summary
        recent_alerts = [
            alert for alert in self.alert_manager.alerts
            if alert.timestamp >= time.time() - time_range
        ]

        alert_counts = defaultdict(int)
        for alert in recent_alerts:
            alert_counts[alert.level.value] += 1

        report['alerts_summary'] = {
            'total_alerts': len(recent_alerts),
            'by_level': dict(alert_counts),
            'active_alerts': len(self.alert_manager.get_active_alerts())
        }

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)

        return report

    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on monitoring data"""
        recommendations = []

        summary = report['summary']

        # Rule count recommendations
        if summary['active_rules'] < 3:
            recommendations.append("Consider generating more rules - low active rule count")

        if summary['quarantined_rules'] > summary['active_rules']:
            recommendations.append("High quarantine rate - review rule generation quality")

        # Performance recommendations
        if summary['avg_performance'] < 0.3:
            recommendations.append("Low average performance - consider rule optimization")

        # Diversity recommendations
        if summary['diversity_score'] < 0.2:
            recommendations.append("Low diversity - enable diversity enforcement")

        # Health recommendations
        health_level = summary['system_health']['level']
        if health_level in ['poor', 'critical']:
            recommendations.append(f"System health is {health_level} - immediate attention required")

        # Alert recommendations
        alerts = report['alerts_summary']
        if alerts['active_alerts'] > 5:
            recommendations.append("Many active alerts - review and acknowledge resolved issues")

        return recommendations

# Integration functions
def integrate_monitoring_with_drm(drm_system) -> DRMMonitor:
    """Integrate monitoring system with DRM"""
    monitor = DRMMonitor(drm_system)

    # Override key DRM methods to add monitoring
    original_run_cycle = drm_system.run_cycle
    original_validate_rule = drm_system.validate_rule
    original_mutate_rule = drm_system.mutate_rule
    original_add_rule = drm_system.add_rule

    def monitored_run_cycle(*args, **kwargs):
        with monitor.time_cycle():
            result = original_run_cycle(*args, **kwargs)
            monitor.collect_metrics()
            return result

    def monitored_validate_rule(rule_id, context):
        with monitor.time_validation():
            result = original_validate_rule(rule_id, context)
            if result.get('activate'):
                monitor.record_rule_activation(rule_id)
            return result

    def monitored_mutate_rule(rule_id, op='tweak_param', magnitude=0.1):
        result = original_mutate_rule(rule_id, op, magnitude)
        if result.get('mutated'):
            monitor.record_rule_mutation(rule_id, op)
        return result

    def monitored_add_rule(rule, check_conflicts=True):
        result = original_add_rule(rule, check_conflicts)
        if result.get('success'):
            monitor.record_rule_generation()
            if result.get('conflicts'):
                monitor.record_conflict_detection(len(result['conflicts']))
        return result

    # Replace methods
    drm_system.run_cycle = monitored_run_cycle
    drm_system.validate_rule = monitored_validate_rule
    drm_system.mutate_rule = monitored_mutate_rule
    drm_system.add_rule = monitored_add_rule

    logger.info("Integrated monitoring system with DRM")
    return monitor

import math  # Add missing import
