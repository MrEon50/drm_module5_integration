"""enhanced_drm_system.py
Enhanced DRM System integrating all improvements.

This module combines:
- Original DRM functionality
- PyTorch integration for neural rule training
- Advanced semantic validation
- Comprehensive monitoring and metrics
- Meta-learning and auto-tuning
- LLM-based rule generation
- Performance optimization
"""

import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

# Import all enhancement modules
try:
    from drm_module5_improved import DRMSystem, Rule, RuleParameter
    from torch_integration import PyTorchDRMTrainer, TrainingConfig
    from semantic_validation import SemanticValidationEngine, integrate_semantic_validation_with_drm
    from monitoring_system import DRMMonitor, integrate_monitoring_with_drm
    from meta_learning import MetaLearningEngine, AutoTuner, integrate_meta_learning_with_drm
    from llm_integration import LLMRuleIntegrator, MockLLMProvider, create_llm_rule_generator
except ImportError as e:
    logging.warning(f"Some enhancement modules not available: {e}")

logger = logging.getLogger(__name__)

@dataclass
class EnhancedDRMConfig:
    """Configuration for Enhanced DRM System"""
    # Core DRM settings
    enable_pytorch_training: bool = True
    enable_semantic_validation: bool = True
    enable_monitoring: bool = True
    enable_meta_learning: bool = True
    enable_llm_generation: bool = True
    
    # PyTorch settings
    pytorch_config: Optional[TrainingConfig] = None
    
    # Monitoring settings
    monitoring_interval: float = 30.0
    
    # Meta-learning settings
    auto_tuning_interval: int = 50
    
    # LLM settings
    llm_provider: str = "mock"  # "mock", "openai"
    llm_api_key: Optional[str] = None

class EnhancedDRMSystem:
    """Enhanced DRM System with all improvements integrated"""
    
    def __init__(self, config: EnhancedDRMConfig = None):
        self.config = config or EnhancedDRMConfig()
        
        # Initialize core DRM system
        self.drm = DRMSystem()
        
        # Initialize enhancement components
        self.pytorch_trainer: Optional[PyTorchDRMTrainer] = None
        self.semantic_validator: Optional[SemanticValidationEngine] = None
        self.monitor: Optional[DRMMonitor] = None
        self.auto_tuner: Optional[AutoTuner] = None
        self.llm_integrator: Optional[LLMRuleIntegrator] = None
        
        # Setup enhancements
        self._setup_enhancements()
        
        logger.info("Enhanced DRM System initialized with all improvements")
    
    def _setup_enhancements(self):
        """Setup all enhancement components"""
        
        # Setup PyTorch integration
        if self.config.enable_pytorch_training:
            try:
                pytorch_config = self.config.pytorch_config or TrainingConfig()
                self.pytorch_trainer = PyTorchDRMTrainer(pytorch_config)
                logger.info("PyTorch integration enabled")
            except Exception as e:
                logger.warning(f"Failed to setup PyTorch integration: {e}")
        
        # Setup semantic validation
        if self.config.enable_semantic_validation:
            try:
                self.semantic_validator = SemanticValidationEngine()
                integrate_semantic_validation_with_drm(self.drm, self.semantic_validator)
                logger.info("Semantic validation enabled")
            except Exception as e:
                logger.warning(f"Failed to setup semantic validation: {e}")
        
        # Setup monitoring
        if self.config.enable_monitoring:
            try:
                self.monitor = integrate_monitoring_with_drm(self.drm)
                self.monitor.start_monitoring(self.config.monitoring_interval)
                logger.info("Monitoring system enabled")
            except Exception as e:
                logger.warning(f"Failed to setup monitoring: {e}")
        
        # Setup meta-learning
        if self.config.enable_meta_learning:
            try:
                self.auto_tuner = integrate_meta_learning_with_drm(self.drm, self.monitor)
                self.auto_tuner.enable_auto_tuning(self.config.auto_tuning_interval)
                logger.info("Meta-learning and auto-tuning enabled")
            except Exception as e:
                logger.warning(f"Failed to setup meta-learning: {e}")
        
        # Setup LLM integration
        if self.config.enable_llm_generation:
            try:
                if self.config.llm_provider == "mock":
                    llm_provider = MockLLMProvider()
                elif self.config.llm_provider == "openai":
                    from llm_integration import OpenAIProvider
                    if not self.config.llm_api_key:
                        raise ValueError("OpenAI API key required")
                    llm_provider = OpenAIProvider(self.config.llm_api_key)
                else:
                    raise ValueError(f"Unknown LLM provider: {self.config.llm_provider}")
                
                self.llm_integrator = LLMRuleIntegrator(self.drm, llm_provider)
                logger.info(f"LLM integration enabled with {self.config.llm_provider} provider")
            except Exception as e:
                logger.warning(f"Failed to setup LLM integration: {e}")
    
    # Core DRM operations with enhancements
    def add_rule(self, rule: Rule, check_conflicts: bool = True) -> Dict[str, Any]:
        """Add rule with enhanced validation and monitoring"""
        result = self.drm.add_rule(rule, check_conflicts)
        
        # Register with PyTorch trainer if applicable
        if (self.pytorch_trainer and rule.type == "HYBRID" and 
            rule.get_differentiable_params()):
            param_specs = {
                name: {
                    "value": param.value,
                    "requires_grad": param.requires_grad,
                    "min_val": param.min_val,
                    "max_val": param.max_val
                }
                for name, param in rule.get_differentiable_params().items()
            }
            self.pytorch_trainer.register_rule(rule.id, param_specs)
        
        return result
    
    def generate_rule_from_description(self, description: str, domain: Optional[str] = None,
                                     rule_type: str = "HEURISTIC") -> Dict[str, Any]:
        """Generate rule from natural language description"""
        if not self.llm_integrator:
            raise RuntimeError("LLM integration not enabled")
        
        return self.llm_integrator.generate_and_add_rule(description, domain, rule_type)
    
    def train_hybrid_rules(self, training_data: List[Dict[str, Any]], 
                          epochs: int = 100) -> Dict[str, Any]:
        """Train HYBRID rules using PyTorch"""
        if not self.pytorch_trainer:
            raise RuntimeError("PyTorch integration not enabled")
        
        # Setup generator if not already done
        if self.pytorch_trainer.rule_generator is None:
            self.pytorch_trainer.initialize_generator()
        
        # Train
        history = self.pytorch_trainer.train(training_data)
        
        # Update DRM rules with learned parameters
        for rule_id in self.drm.rules:
            if self.drm.rules[rule_id].type == "HYBRID":
                self.pytorch_trainer.update_drm_rule(self.drm.rules[rule_id], rule_id)
        
        return {"training_history": history, "trained_rules": len(self.pytorch_trainer.differentiable_rules)}
    
    def run_enhanced_cycle(self, evaluator: Callable[[Rule], Optional[float]],
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run DRM cycle with all enhancements"""
        
        # Use auto-tuner parameters if available
        cycle_params = {}
        if self.auto_tuner and hasattr(self.drm, 'default_cycle_params'):
            cycle_params = self.drm.default_cycle_params.copy()
        
        # Run cycle with monitoring
        result = self.drm.run_cycle(
            evaluator=evaluator,
            context=context,
            **cycle_params
        )
        
        return result
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            "drm_stats": self.drm.get_stats(),
            "timestamp": time.time()
        }
        
        # Add monitoring data
        if self.monitor:
            stats["monitoring"] = self.monitor.get_dashboard_data()
        
        # Add meta-learning data
        if self.auto_tuner:
            stats["meta_learning"] = self.auto_tuner.get_tuning_status()
        
        # Add LLM generation data
        if self.llm_integrator:
            stats["llm_generation"] = self.llm_integrator.get_generation_summary()
        
        # Add PyTorch training data
        if self.pytorch_trainer:
            stats["pytorch_training"] = {
                "registered_rules": len(self.pytorch_trainer.differentiable_rules),
                "training_history": dict(self.pytorch_trainer.training_history)
            }
        
        # Add semantic validation data
        if self.semantic_validator:
            stats["semantic_validation"] = self.semantic_validator.get_validation_summary()
        
        return stats
    
    def explain_rule(self, rule_id: str, use_llm: bool = True) -> Dict[str, Any]:
        """Get comprehensive rule explanation"""
        if rule_id not in self.drm.rules:
            return {"error": "Rule not found"}
        
        rule = self.drm.rules[rule_id]
        explanation = {
            "basic_explanation": rule.explain(),
            "timestamp": time.time()
        }
        
        # Add LLM explanation if available
        if use_llm and self.llm_integrator:
            try:
                llm_explanation = self.llm_integrator.rule_generator.explain_rule(rule)
                explanation["llm_explanation"] = llm_explanation
            except Exception as e:
                logger.warning(f"Failed to get LLM explanation: {e}")
        
        # Add semantic validation results
        if self.semantic_validator:
            try:
                validation_results = self.semantic_validator.validate_rule(rule, {})
                explanation["semantic_validation"] = {
                    domain: {
                        "passed": result.passed,
                        "score": result.score,
                        "errors": result.errors,
                        "warnings": result.warnings,
                        "suggestions": result.suggestions
                    }
                    for domain, result in validation_results.items()
                }
            except Exception as e:
                logger.warning(f"Failed to get semantic validation: {e}")
        
        return explanation
    
    def optimize_performance(self, target_metric: str = "avg_performance", 
                           target_value: float = 0.8, max_experiments: int = 10) -> Dict[str, Any]:
        """Optimize system performance using meta-learning"""
        if not self.auto_tuner:
            raise RuntimeError("Meta-learning not enabled")
        
        optimization_results = []
        
        for experiment in range(max_experiments):
            # Force new tuning experiment
            self.auto_tuner.force_tuning_experiment()
            
            # Run some cycles to evaluate
            total_performance = 0.0
            num_cycles = 20
            
            def dummy_evaluator(rule):
                # Simple dummy evaluator for optimization
                return rule.post_mean + 0.1 * (1.0 - rule.post_var)
            
            for cycle in range(num_cycles):
                result = self.run_enhanced_cycle(dummy_evaluator)
                total_performance += result.get('avg_performance', 0.0)
            
            avg_performance = total_performance / num_cycles
            optimization_results.append({
                "experiment": experiment,
                "performance": avg_performance,
                "hyperparams": self.auto_tuner.meta_learner.current_experiment
            })
            
            # Check if target reached
            if avg_performance >= target_value:
                logger.info(f"Target performance {target_value} reached in experiment {experiment}")
                break
        
        return {
            "optimization_results": optimization_results,
            "best_performance": max(r["performance"] for r in optimization_results),
            "meta_learning_summary": self.auto_tuner.meta_learner.get_optimization_summary()
        }
    
    def export_system_state(self, include_history: bool = True) -> Dict[str, Any]:
        """Export complete system state"""
        export_data = {
            "timestamp": time.time(),
            "config": self.config.__dict__,
            "drm_state": {
                "rules": {rule_id: rule.to_dict() for rule_id, rule in self.drm.rules.items()},
                "archived": {rule_id: rule.to_dict() for rule_id, rule in self.drm.archived.items()},
                "stats": self.drm.get_stats(),
                "audit_log": self.drm.audit_log if include_history else []
            }
        }
        
        # Add enhancement data
        if self.monitor and include_history:
            export_data["monitoring_history"] = self.monitor.export_metrics()
        
        if self.auto_tuner and include_history:
            export_data["meta_learning_history"] = self.auto_tuner.meta_learner.export_optimization_history()
        
        if self.llm_integrator and include_history:
            export_data["llm_generation_history"] = self.llm_integrator.generation_history
        
        return export_data
    
    def shutdown(self):
        """Gracefully shutdown the enhanced DRM system"""
        logger.info("Shutting down Enhanced DRM System...")
        
        # Stop monitoring
        if self.monitor:
            self.monitor.stop_monitoring()
        
        # Disable auto-tuning
        if self.auto_tuner:
            self.auto_tuner.disable_auto_tuning()
        
        # Save PyTorch models if needed
        if self.pytorch_trainer:
            try:
                self.pytorch_trainer.save_model("enhanced_drm_pytorch_state.pt")
            except Exception as e:
                logger.warning(f"Failed to save PyTorch state: {e}")
        
        logger.info("Enhanced DRM System shutdown complete")

# Convenience functions
def create_enhanced_drm(config: EnhancedDRMConfig = None) -> EnhancedDRMSystem:
    """Create an Enhanced DRM System with default configuration"""
    return EnhancedDRMSystem(config)

def create_physics_drm(llm_api_key: Optional[str] = None) -> EnhancedDRMSystem:
    """Create Enhanced DRM optimized for physics domain"""
    config = EnhancedDRMConfig(
        llm_provider="openai" if llm_api_key else "mock",
        llm_api_key=llm_api_key,
        auto_tuning_interval=30  # More frequent tuning for physics
    )
    return EnhancedDRMSystem(config)

def create_minimal_drm() -> EnhancedDRMSystem:
    """Create Enhanced DRM with minimal features for testing"""
    config = EnhancedDRMConfig(
        enable_pytorch_training=False,
        enable_meta_learning=False,
        enable_llm_generation=False,
        monitoring_interval=60.0
    )
    return EnhancedDRMSystem(config)
