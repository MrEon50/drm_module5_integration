"""meta_learning.py
Meta-learning system for DRM - learning how to learn.

Features:
- Automatic hyperparameter optimization
- Strategy adaptation based on performance
- Learning rate scheduling
- Population-based training
- Bayesian optimization for hyperparameters
- Adaptive mutation strategies
- Performance-based strategy selection
"""

import random
import math
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)

class OptimizationStrategy(Enum):
    """Types of optimization strategies"""
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    GRADIENT_BASED = "gradient_based"

@dataclass
class HyperParameter:
    """Hyperparameter definition"""
    name: str
    param_type: str  # 'float', 'int', 'categorical'
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    current_value: Any = None
    
    def sample(self) -> Any:
        """Sample a value for this hyperparameter"""
        if self.param_type == 'categorical':
            return random.choice(self.choices)
        elif self.param_type == 'int':
            return random.randint(int(self.min_val), int(self.max_val))
        elif self.param_type == 'float':
            if self.log_scale:
                log_min = math.log(self.min_val)
                log_max = math.log(self.max_val)
                return math.exp(random.uniform(log_min, log_max))
            else:
                return random.uniform(self.min_val, self.max_val)
        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")

@dataclass
class ExperimentResult:
    """Result of a hyperparameter experiment"""
    hyperparams: Dict[str, Any]
    performance: float
    duration: float
    timestamp: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class HyperparameterOptimizer(ABC):
    """Abstract base class for hyperparameter optimizers"""
    
    @abstractmethod
    def suggest_hyperparams(self, param_space: Dict[str, HyperParameter], 
                           history: List[ExperimentResult]) -> Dict[str, Any]:
        """Suggest next hyperparameters to try"""
        pass
    
    @abstractmethod
    def update(self, result: ExperimentResult):
        """Update optimizer with new result"""
        pass

class RandomSearchOptimizer(HyperparameterOptimizer):
    """Random search hyperparameter optimizer"""
    
    def suggest_hyperparams(self, param_space: Dict[str, HyperParameter], 
                           history: List[ExperimentResult]) -> Dict[str, Any]:
        """Randomly sample hyperparameters"""
        return {name: param.sample() for name, param in param_space.items()}
    
    def update(self, result: ExperimentResult):
        """No update needed for random search"""
        pass

class BayesianOptimizer(HyperparameterOptimizer):
    """Bayesian optimization using Gaussian Process (simplified)"""
    
    def __init__(self, acquisition_function: str = "ei"):
        self.acquisition_function = acquisition_function
        self.history: List[ExperimentResult] = []
        self.best_performance = -float('inf')
    
    def suggest_hyperparams(self, param_space: Dict[str, HyperParameter], 
                           history: List[ExperimentResult]) -> Dict[str, Any]:
        """Suggest hyperparameters using Bayesian optimization"""
        self.history = history
        
        if len(history) < 3:
            # Not enough data for Bayesian optimization, use random
            return {name: param.sample() for name, param in param_space.items()}
        
        # Update best performance
        self.best_performance = max(result.performance for result in history)
        
        # Simplified acquisition function - explore areas with high uncertainty
        # In a full implementation, this would use a proper GP
        best_candidates = []
        
        for _ in range(100):  # Sample candidates
            candidate = {name: param.sample() for name, param in param_space.items()}
            score = self._acquisition_score(candidate, param_space)
            best_candidates.append((score, candidate))
        
        # Return best candidate
        best_candidates.sort(reverse=True)
        return best_candidates[0][1]
    
    def _acquisition_score(self, candidate: Dict[str, Any], 
                          param_space: Dict[str, HyperParameter]) -> float:
        """Simplified acquisition function"""
        # Expected improvement approximation
        if not self.history:
            return random.random()
        
        # Find most similar historical point
        min_distance = float('inf')
        closest_performance = 0.0
        
        for result in self.history:
            distance = self._parameter_distance(candidate, result.hyperparams, param_space)
            if distance < min_distance:
                min_distance = distance
                closest_performance = result.performance
        
        # Simple expected improvement
        improvement = max(0, closest_performance - self.best_performance)
        uncertainty = 1.0 / (1.0 + min_distance)  # Higher uncertainty for distant points
        
        return improvement + 0.1 * uncertainty
    
    def _parameter_distance(self, params1: Dict[str, Any], params2: Dict[str, Any],
                           param_space: Dict[str, HyperParameter]) -> float:
        """Calculate normalized distance between parameter sets"""
        distance = 0.0
        
        for name, param_def in param_space.items():
            if name not in params1 or name not in params2:
                continue
            
            val1, val2 = params1[name], params2[name]
            
            if param_def.param_type == 'categorical':
                distance += 0.0 if val1 == val2 else 1.0
            else:
                # Normalize to [0, 1]
                if param_def.param_type == 'float':
                    range_size = param_def.max_val - param_def.min_val
                    norm_diff = abs(val1 - val2) / range_size
                else:  # int
                    range_size = param_def.max_val - param_def.min_val
                    norm_diff = abs(val1 - val2) / range_size
                
                distance += norm_diff ** 2
        
        return math.sqrt(distance)
    
    def update(self, result: ExperimentResult):
        """Update with new result"""
        self.history.append(result)

class EvolutionaryOptimizer(HyperparameterOptimizer):
    """Evolutionary hyperparameter optimization"""
    
    def __init__(self, population_size: int = 20, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population: List[Dict[str, Any]] = []
        self.fitness_scores: List[float] = []
    
    def suggest_hyperparams(self, param_space: Dict[str, HyperParameter], 
                           history: List[ExperimentResult]) -> Dict[str, Any]:
        """Suggest hyperparameters using evolutionary approach"""
        
        # Initialize population if empty
        if not self.population:
            self.population = [
                {name: param.sample() for name, param in param_space.items()}
                for _ in range(self.population_size)
            ]
            self.fitness_scores = [0.0] * self.population_size
        
        # Update fitness scores from history
        self._update_fitness(history)
        
        # Select parents and create offspring
        if len(history) >= self.population_size:
            offspring = self._create_offspring(param_space)
            return offspring
        else:
            # Return next individual from initial population
            return self.population[len(history) % self.population_size]
    
    def _update_fitness(self, history: List[ExperimentResult]):
        """Update fitness scores based on history"""
        for i, individual in enumerate(self.population):
            # Find matching result in history
            for result in history:
                if self._params_match(individual, result.hyperparams):
                    self.fitness_scores[i] = result.performance
                    break
    
    def _params_match(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> bool:
        """Check if parameter sets match"""
        for key in params1:
            if key not in params2 or params1[key] != params2[key]:
                return False
        return True
    
    def _create_offspring(self, param_space: Dict[str, HyperParameter]) -> Dict[str, Any]:
        """Create offspring through crossover and mutation"""
        # Tournament selection
        parent1 = self._tournament_selection()
        parent2 = self._tournament_selection()
        
        # Crossover
        offspring = {}
        for name, param_def in param_space.items():
            if random.random() < 0.5:
                offspring[name] = parent1[name]
            else:
                offspring[name] = parent2[name]
        
        # Mutation
        for name, param_def in param_space.items():
            if random.random() < self.mutation_rate:
                offspring[name] = param_def.sample()
        
        return offspring
    
    def _tournament_selection(self, tournament_size: int = 3) -> Dict[str, Any]:
        """Tournament selection"""
        tournament_indices = random.sample(range(len(self.population)), 
                                         min(tournament_size, len(self.population)))
        best_idx = max(tournament_indices, key=lambda i: self.fitness_scores[i])
        return self.population[best_idx]
    
    def update(self, result: ExperimentResult):
        """Update with new result"""
        # Find corresponding individual and update fitness
        for i, individual in enumerate(self.population):
            if self._params_match(individual, result.hyperparams):
                self.fitness_scores[i] = result.performance
                break

class AdaptiveStrategy:
    """Adaptive strategy that switches between different approaches"""
    
    def __init__(self, strategies: List[str] = None):
        if strategies is None:
            strategies = ["conservative", "aggressive", "balanced", "exploratory"]
        
        self.strategies = strategies
        self.strategy_performance: Dict[str, deque] = {
            strategy: deque(maxlen=10) for strategy in strategies
        }
        self.current_strategy = "balanced"
        self.strategy_switch_threshold = 0.1
        self.min_samples_for_switch = 5
    
    def get_current_strategy(self) -> str:
        """Get current strategy"""
        return self.current_strategy
    
    def update_performance(self, strategy: str, performance: float):
        """Update performance for a strategy"""
        if strategy in self.strategy_performance:
            self.strategy_performance[strategy].append(performance)
            self._maybe_switch_strategy()
    
    def _maybe_switch_strategy(self):
        """Consider switching strategy based on performance"""
        # Calculate average performance for each strategy
        strategy_averages = {}
        
        for strategy, performances in self.strategy_performance.items():
            if len(performances) >= self.min_samples_for_switch:
                strategy_averages[strategy] = sum(performances) / len(performances)
        
        if len(strategy_averages) < 2:
            return  # Not enough data
        
        # Find best strategy
        best_strategy = max(strategy_averages.keys(), 
                           key=lambda s: strategy_averages[s])
        current_avg = strategy_averages.get(self.current_strategy, 0.0)
        best_avg = strategy_averages[best_strategy]
        
        # Switch if improvement is significant
        if best_avg - current_avg > self.strategy_switch_threshold:
            logger.info(f"Switching strategy from {self.current_strategy} to {best_strategy} "
                       f"(improvement: {best_avg - current_avg:.3f})")
            self.current_strategy = best_strategy
    
    def get_strategy_config(self, strategy: str) -> Dict[str, Any]:
        """Get configuration for a strategy"""
        configs = {
            "conservative": {
                "eta": 0.01,
                "beta": 0.3,
                "lam": 0.7,
                "mutation_magnitude": 0.05,
                "exploration_bonus": 0.01
            },
            "aggressive": {
                "eta": 0.1,
                "beta": 0.8,
                "lam": 0.2,
                "mutation_magnitude": 0.3,
                "exploration_bonus": 0.1
            },
            "balanced": {
                "eta": 0.05,
                "beta": 0.5,
                "lam": 0.5,
                "mutation_magnitude": 0.1,
                "exploration_bonus": 0.02
            },
            "exploratory": {
                "eta": 0.03,
                "beta": 0.7,
                "lam": 0.3,
                "mutation_magnitude": 0.2,
                "exploration_bonus": 0.05
            }
        }
        
        return configs.get(strategy, configs["balanced"])

class MetaLearningEngine:
    """Main meta-learning engine for DRM"""
    
    def __init__(self, drm_system):
        self.drm_system = drm_system
        self.hyperparameter_space = self._define_hyperparameter_space()
        self.optimizer = BayesianOptimizer()
        self.adaptive_strategy = AdaptiveStrategy()
        self.experiment_history: List[ExperimentResult] = []
        self.current_experiment: Optional[Dict[str, Any]] = None
        self.experiment_start_time: Optional[float] = None
        
        # Performance tracking
        self.baseline_performance = 0.0
        self.improvement_threshold = 0.05
        self.experiment_duration = 100  # cycles per experiment
        self.cycles_in_current_experiment = 0
        
        logger.info("Initialized meta-learning engine")
    
    def _define_hyperparameter_space(self) -> Dict[str, HyperParameter]:
        """Define the hyperparameter search space"""
        return {
            "eta": HyperParameter("eta", "float", 0.001, 0.2, log_scale=True),
            "beta": HyperParameter("beta", "float", 0.1, 0.9),
            "lam": HyperParameter("lam", "float", 0.1, 0.9),
            "kl_max": HyperParameter("kl_max", "float", 0.1, 1.0),
            "mu_min": HyperParameter("mu_min", "float", 0.05, 0.3),
            "tau": HyperParameter("tau", "float", 0.8, 0.99),
            "mutation_magnitude": HyperParameter("mutation_magnitude", "float", 0.01, 0.5),
            "exploration_bonus": HyperParameter("exploration_bonus", "float", 0.001, 0.1, log_scale=True),
            "diversity_threshold": HyperParameter("diversity_threshold", "float", 0.1, 0.5),
            "stagnation_window": HyperParameter("stagnation_window", "int", 10, 100),
            "revival_count": HyperParameter("revival_count", "int", 1, 10),
            "conflict_resolution_strategy": HyperParameter(
                "conflict_resolution_strategy", "categorical", 
                choices=["priority", "performance", "consensus"]
            )
        }

    def start_experiment(self) -> Dict[str, Any]:
        """Start a new hyperparameter experiment"""
        # Get suggested hyperparameters
        suggested_params = self.optimizer.suggest_hyperparams(
            self.hyperparameter_space, self.experiment_history
        )

        # Get adaptive strategy configuration
        current_strategy = self.adaptive_strategy.get_current_strategy()
        strategy_config = self.adaptive_strategy.get_strategy_config(current_strategy)

        # Merge suggested params with strategy config
        experiment_params = {**suggested_params, **strategy_config}

        self.current_experiment = experiment_params
        self.experiment_start_time = time.time()
        self.cycles_in_current_experiment = 0

        logger.info(f"Started experiment with strategy '{current_strategy}' and params: {experiment_params}")
        return experiment_params

    def update_experiment_progress(self, cycle_performance: float):
        """Update progress of current experiment"""
        if self.current_experiment is None:
            return

        self.cycles_in_current_experiment += 1

        # Check if experiment should end
        if self.cycles_in_current_experiment >= self.experiment_duration:
            self.end_experiment(cycle_performance)

    def end_experiment(self, final_performance: float):
        """End current experiment and record results"""
        if self.current_experiment is None:
            return

        duration = time.time() - self.experiment_start_time

        # Create experiment result
        result = ExperimentResult(
            hyperparams=self.current_experiment.copy(),
            performance=final_performance,
            duration=duration,
            timestamp=time.time(),
            metadata={
                "cycles": self.cycles_in_current_experiment,
                "strategy": self.adaptive_strategy.get_current_strategy()
            }
        )

        # Update optimizer and strategy
        self.optimizer.update(result)
        self.adaptive_strategy.update_performance(
            self.adaptive_strategy.get_current_strategy(),
            final_performance
        )

        # Record in history
        self.experiment_history.append(result)

        # Update baseline if this was better
        if final_performance > self.baseline_performance:
            improvement = final_performance - self.baseline_performance
            self.baseline_performance = final_performance
            logger.info(f"New baseline performance: {final_performance:.4f} "
                       f"(improvement: {improvement:.4f})")

        logger.info(f"Experiment completed: performance={final_performance:.4f}, "
                   f"duration={duration:.1f}s, cycles={self.cycles_in_current_experiment}")

        # Reset for next experiment
        self.current_experiment = None
        self.experiment_start_time = None
        self.cycles_in_current_experiment = 0

    def get_best_hyperparameters(self) -> Optional[Dict[str, Any]]:
        """Get best hyperparameters found so far"""
        if not self.experiment_history:
            return None

        best_result = max(self.experiment_history, key=lambda r: r.performance)
        return best_result.hyperparams

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization progress"""
        if not self.experiment_history:
            return {"experiments": 0}

        performances = [r.performance for r in self.experiment_history]

        return {
            "experiments": len(self.experiment_history),
            "best_performance": max(performances),
            "worst_performance": min(performances),
            "average_performance": sum(performances) / len(performances),
            "improvement_over_baseline": max(performances) - self.baseline_performance,
            "current_strategy": self.adaptive_strategy.get_current_strategy(),
            "strategy_performance": {
                strategy: list(perfs)
                for strategy, perfs in self.adaptive_strategy.strategy_performance.items()
            }
        }

    def should_start_new_experiment(self) -> bool:
        """Check if a new experiment should be started"""
        return self.current_experiment is None

    def apply_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Apply hyperparameters to DRM system"""
        # This would modify the DRM system's parameters
        # Implementation depends on DRM system structure

        # Update DRM cycle parameters
        if hasattr(self.drm_system, 'default_cycle_params'):
            for param, value in hyperparams.items():
                if param in ['eta', 'beta', 'lam', 'kl_max', 'mu_min', 'tau']:
                    self.drm_system.default_cycle_params[param] = value

        # Update mutation parameters
        if hasattr(self.drm_system, 'default_mutation_magnitude'):
            if 'mutation_magnitude' in hyperparams:
                self.drm_system.default_mutation_magnitude = hyperparams['mutation_magnitude']

        # Update diversity enforcement
        if hasattr(self.drm_system.diversity, 'sim_threshold'):
            if 'diversity_threshold' in hyperparams:
                self.drm_system.diversity.sim_threshold = 1.0 - hyperparams['diversity_threshold']

        # Update stagnation detection
        if hasattr(self.drm_system.stagnation, 'window'):
            if 'stagnation_window' in hyperparams:
                self.drm_system.stagnation.window = int(hyperparams['stagnation_window'])

        logger.debug(f"Applied hyperparameters: {hyperparams}")

    def export_optimization_history(self) -> str:
        """Export optimization history as JSON"""
        export_data = {
            "meta_learning_summary": self.get_optimization_summary(),
            "hyperparameter_space": {
                name: {
                    "type": param.param_type,
                    "min_val": param.min_val,
                    "max_val": param.max_val,
                    "choices": param.choices,
                    "log_scale": param.log_scale
                }
                for name, param in self.hyperparameter_space.items()
            },
            "experiment_history": [asdict(result) for result in self.experiment_history]
        }

        return json.dumps(export_data, indent=2)

class AutoTuner:
    """Automatic tuning system that combines meta-learning with DRM"""

    def __init__(self, drm_system, monitoring_system=None):
        self.drm_system = drm_system
        self.monitoring_system = monitoring_system
        self.meta_learner = MetaLearningEngine(drm_system)
        self.auto_tuning_enabled = False
        self.tuning_interval = 50  # cycles between tuning attempts
        self.cycles_since_last_tune = 0
        self.performance_window = deque(maxlen=20)

    def enable_auto_tuning(self, interval: int = 50):
        """Enable automatic tuning"""
        self.auto_tuning_enabled = True
        self.tuning_interval = interval
        logger.info(f"Enabled auto-tuning with interval {interval} cycles")

    def disable_auto_tuning(self):
        """Disable automatic tuning"""
        self.auto_tuning_enabled = False
        logger.info("Disabled auto-tuning")

    def on_cycle_complete(self, cycle_result: Dict[str, Any]):
        """Called after each DRM cycle"""
        if not self.auto_tuning_enabled:
            return

        # Track performance
        performance = cycle_result.get('avg_performance', 0.0)
        self.performance_window.append(performance)

        # Update current experiment
        if self.meta_learner.current_experiment is not None:
            self.meta_learner.update_experiment_progress(performance)

        self.cycles_since_last_tune += 1

        # Check if we should start tuning
        if self.cycles_since_last_tune >= self.tuning_interval:
            self._maybe_start_tuning()
            self.cycles_since_last_tune = 0

    def _maybe_start_tuning(self):
        """Decide whether to start a new tuning experiment"""
        if not self.meta_learner.should_start_new_experiment():
            return

        # Check if performance is stagnating
        if len(self.performance_window) >= 10:
            recent_avg = sum(list(self.performance_window)[-5:]) / 5
            older_avg = sum(list(self.performance_window)[-10:-5]) / 5

            # If performance is not improving, start tuning
            if recent_avg <= older_avg + 0.01:  # Small improvement threshold
                logger.info("Performance stagnation detected, starting hyperparameter tuning")
                self._start_tuning_experiment()

    def _start_tuning_experiment(self):
        """Start a new tuning experiment"""
        experiment_params = self.meta_learner.start_experiment()
        self.meta_learner.apply_hyperparameters(experiment_params)

        # Log to monitoring system if available
        if self.monitoring_system:
            self.monitoring_system.record_rule_generation(0)  # Mark tuning event

    def get_tuning_status(self) -> Dict[str, Any]:
        """Get current tuning status"""
        return {
            "auto_tuning_enabled": self.auto_tuning_enabled,
            "current_experiment_active": self.meta_learner.current_experiment is not None,
            "cycles_since_last_tune": self.cycles_since_last_tune,
            "tuning_interval": self.tuning_interval,
            "optimization_summary": self.meta_learner.get_optimization_summary(),
            "recent_performance": list(self.performance_window)
        }

    def force_tuning_experiment(self):
        """Force start a tuning experiment"""
        if self.meta_learner.current_experiment is not None:
            # End current experiment first
            avg_performance = sum(self.performance_window) / len(self.performance_window) if self.performance_window else 0.0
            self.meta_learner.end_experiment(avg_performance)

        self._start_tuning_experiment()
        logger.info("Forced start of tuning experiment")

# Integration functions
def integrate_meta_learning_with_drm(drm_system, monitoring_system=None) -> AutoTuner:
    """Integrate meta-learning with DRM system"""
    auto_tuner = AutoTuner(drm_system, monitoring_system)

    # Override run_cycle to add meta-learning hooks
    original_run_cycle = drm_system.run_cycle

    def meta_learning_run_cycle(*args, **kwargs):
        result = original_run_cycle(*args, **kwargs)

        # Add performance metrics to result
        stats = drm_system.get_stats()
        result['avg_performance'] = stats['avg_performance']
        result['diversity_score'] = stats['diversity_score']

        # Notify auto-tuner
        auto_tuner.on_cycle_complete(result)

        return result

    # Replace method
    drm_system.run_cycle = meta_learning_run_cycle

    # Add default cycle parameters if not present
    if not hasattr(drm_system, 'default_cycle_params'):
        drm_system.default_cycle_params = {
            'eta': 0.05,
            'beta': 0.5,
            'lam': 0.5,
            'kl_max': 0.5,
            'mu_min': 0.1,
            'tau': 0.95
        }

    if not hasattr(drm_system, 'default_mutation_magnitude'):
        drm_system.default_mutation_magnitude = 0.1

    logger.info("Integrated meta-learning system with DRM")
    return auto_tuner

def create_hyperparameter_report(meta_learner: MetaLearningEngine) -> Dict[str, Any]:
    """Create comprehensive hyperparameter optimization report"""
    if not meta_learner.experiment_history:
        return {"error": "No experiments completed yet"}

    history = meta_learner.experiment_history

    # Performance analysis
    performances = [r.performance for r in history]
    best_idx = performances.index(max(performances))
    worst_idx = performances.index(min(performances))

    # Parameter impact analysis
    param_impact = {}
    for param_name in meta_learner.hyperparameter_space.keys():
        param_values = [r.hyperparams.get(param_name) for r in history if param_name in r.hyperparams]
        if len(param_values) > 1:
            # Simple correlation with performance
            correlation = _calculate_correlation(param_values, performances[:len(param_values)])
            param_impact[param_name] = {
                "correlation_with_performance": correlation,
                "best_value": history[best_idx].hyperparams.get(param_name),
                "worst_value": history[worst_idx].hyperparams.get(param_name),
                "value_range": [min(param_values), max(param_values)]
            }

    return {
        "summary": meta_learner.get_optimization_summary(),
        "best_experiment": asdict(history[best_idx]),
        "worst_experiment": asdict(history[worst_idx]),
        "parameter_impact_analysis": param_impact,
        "recommendations": _generate_hyperparameter_recommendations(param_impact)
    }

def _calculate_correlation(x_values: List[float], y_values: List[float]) -> float:
    """Calculate simple correlation coefficient"""
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0

    n = len(x_values)
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    sum_xy = sum(x * y for x, y in zip(x_values, y_values))
    sum_x2 = sum(x * x for x in x_values)
    sum_y2 = sum(y * y for y in y_values)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))

    return numerator / denominator if denominator != 0 else 0.0

def _generate_hyperparameter_recommendations(param_impact: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on parameter impact analysis"""
    recommendations = []

    for param_name, impact in param_impact.items():
        correlation = impact["correlation_with_performance"]

        if abs(correlation) > 0.5:  # Strong correlation
            if correlation > 0:
                recommendations.append(f"Increase {param_name} for better performance (correlation: {correlation:.3f})")
            else:
                recommendations.append(f"Decrease {param_name} for better performance (correlation: {correlation:.3f})")
        elif abs(correlation) < 0.1:  # Weak correlation
            recommendations.append(f"Parameter {param_name} has minimal impact on performance")

    return recommendations
