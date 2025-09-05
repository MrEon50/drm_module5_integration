"""demo_enhanced_drm.py
Demonstration of Enhanced DRM System capabilities.

This script showcases all the improvements:
- PyTorch integration for neural rule training
- Semantic validation across multiple domains
- Real-time monitoring and metrics
- Meta-learning and automatic hyperparameter tuning
- LLM-based rule generation
- Comprehensive analytics and reporting
"""

import time
import json
import random
import logging
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_enhanced_drm():
    """Main demonstration function"""
    print("🚀 Enhanced DRM System Demonstration")
    print("=" * 50)
    
    # Import the enhanced system
    try:
        from enhanced_drm_system import create_enhanced_drm, EnhancedDRMConfig
        from drm_module5_improved import Rule, RuleParameter
    except ImportError as e:
        print(f"❌ Failed to import enhanced DRM: {e}")
        return
    
    # Create enhanced DRM system
    print("\n📦 Creating Enhanced DRM System...")
    config = EnhancedDRMConfig(
        enable_pytorch_training=True,
        enable_semantic_validation=True,
        enable_monitoring=True,
        enable_meta_learning=True,
        enable_llm_generation=True,
        monitoring_interval=10.0,
        auto_tuning_interval=25
    )
    
    enhanced_drm = create_enhanced_drm(config)
    print("✅ Enhanced DRM System created successfully!")
    
    # Demonstrate LLM rule generation
    print("\n🤖 Demonstrating LLM Rule Generation...")
    demo_llm_generation(enhanced_drm)
    
    # Demonstrate semantic validation
    print("\n🔍 Demonstrating Semantic Validation...")
    demo_semantic_validation(enhanced_drm)
    
    # Demonstrate PyTorch training
    print("\n🧠 Demonstrating PyTorch Training...")
    demo_pytorch_training(enhanced_drm)
    
    # Demonstrate monitoring and metrics
    print("\n📊 Demonstrating Monitoring System...")
    demo_monitoring(enhanced_drm)
    
    # Demonstrate meta-learning
    print("\n🎯 Demonstrating Meta-Learning...")
    demo_meta_learning(enhanced_drm)
    
    # Run enhanced cycles
    print("\n⚡ Running Enhanced DRM Cycles...")
    demo_enhanced_cycles(enhanced_drm)
    
    # Generate comprehensive report
    print("\n📋 Generating Comprehensive Report...")
    demo_reporting(enhanced_drm)
    
    # Cleanup
    print("\n🧹 Shutting down system...")
    enhanced_drm.shutdown()
    print("✅ Demonstration completed successfully!")

def demo_llm_generation(enhanced_drm):
    """Demonstrate LLM-based rule generation"""
    
    rule_descriptions = [
        "Energy is conserved in isolated physical systems",
        "Prime numbers are only divisible by 1 and themselves",
        "If all premises are true and the argument is valid, then the conclusion must be true",
        "Objects in motion tend to stay in motion unless acted upon by an external force",
        "The derivative of a constant function is zero"
    ]
    
    domains = ["physics", "mathematics", "logic", "physics", "mathematics"]
    rule_types = ["LOGICAL", "LOGICAL", "LOGICAL", "HEURISTIC", "LOGICAL"]
    
    for i, (description, domain, rule_type) in enumerate(zip(rule_descriptions, domains, rule_types)):
        try:
            print(f"  🔄 Generating rule {i+1}: {description[:50]}...")
            result = enhanced_drm.generate_rule_from_description(description, domain, rule_type)
            
            if result["add_result"].get("success"):
                print(f"  ✅ Rule generated successfully: {result['drm_rule'].name}")
                print(f"     Confidence: {result['generated_rule'].confidence:.2f}")
            else:
                print(f"  ⚠️ Rule generation had issues: {result['add_result']}")
                
        except Exception as e:
            print(f"  ❌ Failed to generate rule: {e}")
    
    # Show generation summary
    if enhanced_drm.llm_integrator:
        summary = enhanced_drm.llm_integrator.get_generation_summary()
        print(f"  📈 Generation Summary: {summary['successful']}/{summary['total_generated']} successful")

def demo_semantic_validation(enhanced_drm):
    """Demonstrate semantic validation capabilities"""
    
    # Get some rules to validate
    rules = list(enhanced_drm.drm.rules.values())[:3]
    
    for rule in rules:
        print(f"  🔍 Validating rule: {rule.name}")
        
        # Create test context
        context = {
            "text": f"Testing rule {rule.name} in domain context",
            "domain": "physics" if "energy" in rule.name.lower() else "mathematics"
        }
        
        try:
            validation_result = enhanced_drm.drm.validate_rule(rule.id, context)
            
            if validation_result.get("semantic_validation"):
                for domain, result in validation_result["semantic_validation"].items():
                    status = "✅" if result["passed"] else "❌"
                    print(f"    {status} {domain}: score={result['score']:.2f}")
                    
                    if result["errors"]:
                        print(f"      Errors: {result['errors'][:2]}")
                    if result["suggestions"]:
                        print(f"      Suggestions: {result['suggestions'][:1]}")
            else:
                print(f"    ℹ️ No semantic validation results")
                
        except Exception as e:
            print(f"    ❌ Validation failed: {e}")

def demo_pytorch_training(enhanced_drm):
    """Demonstrate PyTorch training capabilities"""
    
    # Create some HYBRID rules for training
    from drm_module5_improved import Rule, RuleParameter
    
    hybrid_rule = Rule(
        id="hybrid_demo_rule",
        name="Demo Hybrid Rule",
        rtype="HYBRID",
        init_weight=1.0,
        init_mean=0.5,
        init_var=0.1,
        params={
            "learning_rate": RuleParameter(0.1, "float", 0.01, 0.5, requires_grad=True),
            "momentum": RuleParameter(0.9, "float", 0.1, 0.99, requires_grad=True)
        },
        pre_conditions=["training_context"],
        post_conditions=["model_updated"]
    )
    
    enhanced_drm.add_rule(hybrid_rule)
    print(f"  ➕ Added hybrid rule: {hybrid_rule.name}")
    
    # Create training data
    training_data = []
    for i in range(50):
        training_data.append({
            "rule_id": "hybrid_demo_rule",
            "performance": random.uniform(0.3, 0.9),
            "context": {"iteration": i, "batch_size": 32}
        })
    
    try:
        print("  🏋️ Training hybrid rules...")
        training_result = enhanced_drm.train_hybrid_rules(training_data, epochs=20)
        print(f"  ✅ Training completed: {training_result['trained_rules']} rules trained")
        
        # Show parameter updates
        updated_rule = enhanced_drm.drm.rules["hybrid_demo_rule"]
        print(f"  📊 Updated parameters:")
        for name, param in updated_rule.params.items():
            print(f"    {name}: {param.value:.4f}")
            
    except Exception as e:
        print(f"  ❌ Training failed: {e}")

def demo_monitoring(enhanced_drm):
    """Demonstrate monitoring system"""
    
    if not enhanced_drm.monitor:
        print("  ⚠️ Monitoring not available")
        return
    
    # Let the system run for a bit to collect metrics
    print("  📊 Collecting metrics...")
    time.sleep(2)
    
    # Get dashboard data
    dashboard = enhanced_drm.monitor.get_dashboard_data()
    
    print(f"  📈 System Health: {dashboard['system_health']['level']}")
    print(f"  📊 Active Alerts: {dashboard['alerts']['active']}")
    
    # Show key metrics
    key_metrics = ['rule_count', 'avg_rule_performance', 'diversity_score']
    for metric_name in key_metrics:
        if metric_name in dashboard['metrics']:
            metric = dashboard['metrics'][metric_name]
            current = metric.get('current', 'N/A')
            print(f"    {metric_name}: {current}")
    
    # Generate monitoring report
    try:
        report = enhanced_drm.monitor.generate_report(time_range=300)
        print(f"  📋 Report generated with {len(report['recommendations'])} recommendations")
        if report['recommendations']:
            print(f"    Top recommendation: {report['recommendations'][0]}")
    except Exception as e:
        print(f"  ❌ Report generation failed: {e}")

def demo_meta_learning(enhanced_drm):
    """Demonstrate meta-learning and auto-tuning"""
    
    if not enhanced_drm.auto_tuner:
        print("  ⚠️ Meta-learning not available")
        return
    
    # Get tuning status
    status = enhanced_drm.auto_tuner.get_tuning_status()
    print(f"  🎯 Auto-tuning enabled: {status['auto_tuning_enabled']}")
    print(f"  🔄 Current experiment active: {status['current_experiment_active']}")
    
    # Force a tuning experiment
    try:
        print("  🚀 Starting optimization experiment...")
        enhanced_drm.auto_tuner.force_tuning_experiment()
        
        # Run a few cycles to see the effect
        def simple_evaluator(rule):
            return rule.post_mean + random.uniform(-0.1, 0.1)
        
        for i in range(5):
            enhanced_drm.run_enhanced_cycle(simple_evaluator)
        
        # Get optimization summary
        summary = enhanced_drm.auto_tuner.meta_learner.get_optimization_summary()
        print(f"  📊 Experiments completed: {summary['experiments']}")
        if summary['experiments'] > 0:
            print(f"  🏆 Best performance: {summary['best_performance']:.4f}")
            print(f"  📈 Current strategy: {summary['current_strategy']}")
            
    except Exception as e:
        print(f"  ❌ Meta-learning demo failed: {e}")

def demo_enhanced_cycles(enhanced_drm):
    """Demonstrate enhanced DRM cycles"""
    
    def physics_evaluator(rule):
        """Evaluator that favors physics-related rules"""
        base_score = rule.post_mean
        
        # Bonus for physics-related rules
        if any(term in rule.name.lower() for term in ['energy', 'force', 'motion', 'physics']):
            base_score += 0.1
        
        # Add some randomness
        return base_score + random.uniform(-0.05, 0.05)
    
    print("  ⚡ Running 10 enhanced cycles...")
    
    cycle_results = []
    for cycle in range(10):
        try:
            result = enhanced_drm.run_enhanced_cycle(
                evaluator=physics_evaluator,
                context={"cycle": cycle, "demo": True}
            )
            
            cycle_results.append(result)
            
            if cycle % 3 == 0:  # Show progress every 3 cycles
                stats = enhanced_drm.drm.get_stats()
                print(f"    Cycle {cycle}: {stats['active_rules']} active rules, "
                      f"avg performance: {stats['avg_performance']:.3f}")
                
        except Exception as e:
            print(f"    ❌ Cycle {cycle} failed: {e}")
    
    # Summary
    if cycle_results:
        final_stats = enhanced_drm.drm.get_stats()
        print(f"  ✅ Cycles completed successfully")
        print(f"    Final state: {final_stats['active_rules']} active rules")
        print(f"    Diversity score: {final_stats['diversity_score']:.3f}")

def demo_reporting(enhanced_drm):
    """Demonstrate comprehensive reporting"""
    
    try:
        # Get comprehensive stats
        stats = enhanced_drm.get_comprehensive_stats()
        
        print("  📊 System Overview:")
        drm_stats = stats['drm_stats']
        print(f"    Total rules: {drm_stats['total_rules']}")
        print(f"    Active rules: {drm_stats['active_rules']}")
        print(f"    Average performance: {drm_stats['avg_performance']:.3f}")
        print(f"    Diversity score: {drm_stats['diversity_score']:.3f}")
        
        # Show component status
        components = ['monitoring', 'meta_learning', 'llm_generation', 'pytorch_training', 'semantic_validation']
        for component in components:
            if component in stats:
                print(f"    {component}: ✅ Active")
            else:
                print(f"    {component}: ❌ Not available")
        
        # Rule explanations
        if enhanced_drm.drm.rules:
            sample_rule_id = list(enhanced_drm.drm.rules.keys())[0]
            print(f"\n  📝 Sample Rule Explanation ({sample_rule_id}):")
            
            explanation = enhanced_drm.explain_rule(sample_rule_id, use_llm=True)
            if 'llm_explanation' in explanation:
                llm_text = explanation['llm_explanation'][:200] + "..." if len(explanation['llm_explanation']) > 200 else explanation['llm_explanation']
                print(f"    LLM Explanation: {llm_text}")
            
            if 'semantic_validation' in explanation:
                print(f"    Semantic validation domains: {list(explanation['semantic_validation'].keys())}")
        
        # Export system state
        print(f"\n  💾 Exporting system state...")
        export_data = enhanced_drm.export_system_state(include_history=False)
        print(f"    Export size: {len(json.dumps(export_data))} characters")
        print(f"    Rules exported: {len(export_data['drm_state']['rules'])}")
        
    except Exception as e:
        print(f"  ❌ Reporting failed: {e}")

if __name__ == "__main__":
    demo_enhanced_drm()
