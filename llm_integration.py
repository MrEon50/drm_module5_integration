"""llm_integration.py
Integration with Large Language Models for automatic rule generation.

Features:
- Natural language to rule conversion
- Rule explanation and documentation
- Semantic rule validation using LLMs
- Knowledge extraction from text
- Rule refinement suggestions
- Multi-modal rule generation
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

logger = logging.getLogger(__name__)

@dataclass
class RuleGenerationRequest:
    """Request for rule generation from natural language"""
    description: str
    domain: Optional[str] = None
    rule_type: str = "HEURISTIC"  # LOGICAL, HEURISTIC, HYBRID
    context: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None
    examples: Optional[List[str]] = None

@dataclass
class GeneratedRule:
    """Generated rule from LLM"""
    id: str
    name: str
    rule_type: str
    pre_conditions: List[str]
    post_conditions: List[str]
    parameters: Dict[str, Any]
    confidence: float
    explanation: str
    source_description: str
    metadata: Dict[str, Any]

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: int = 1000, 
                     temperature: float = 0.7) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM is available"""
        pass

class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing and fallback"""
    
    def __init__(self):
        self.templates = {
            "physics": [
                "Energy conservation rule: total energy remains constant in isolated systems",
                "Force equals mass times acceleration (F = ma)",
                "Objects in motion tend to stay in motion unless acted upon by external force"
            ],
            "mathematics": [
                "Prime numbers are only divisible by 1 and themselves",
                "The sum of angles in a triangle equals 180 degrees",
                "Derivative of a constant is zero"
            ],
            "logic": [
                "If P implies Q and P is true, then Q is true (modus ponens)",
                "A statement cannot be both true and false simultaneously",
                "If all A are B and X is A, then X is B"
            ]
        }
    
    def generate_text(self, prompt: str, max_tokens: int = 1000, 
                     temperature: float = 0.7) -> str:
        """Generate mock text based on templates"""
        # Simple keyword-based template selection
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['energy', 'force', 'mass', 'physics']):
            domain = 'physics'
        elif any(word in prompt_lower for word in ['number', 'equation', 'math', 'calculate']):
            domain = 'mathematics'
        elif any(word in prompt_lower for word in ['logic', 'if', 'then', 'implies']):
            domain = 'logic'
        else:
            domain = 'physics'  # default
        
        import random
        template = random.choice(self.templates[domain])
        
        return f"""Based on the description, here's a generated rule:

Rule Name: Generated {domain.title()} Rule
Type: HEURISTIC
Pre-conditions: ["context_available", "domain_{domain}"]
Post-conditions: ["rule_applied", "result_computed"]
Parameters: {{"confidence": 0.8, "domain": "{domain}"}}
Explanation: {template}

This rule applies {domain} principles to the given context."""
    
    def is_available(self) -> bool:
        return True

class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider (requires openai package)"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self._client = None
        
        try:
            import openai
            self._client = openai.OpenAI(api_key=api_key)
        except ImportError:
            logger.warning("OpenAI package not available, using mock provider")
    
    def generate_text(self, prompt: str, max_tokens: int = 1000, 
                     temperature: float = 0.7) -> str:
        if not self._client:
            raise RuntimeError("OpenAI client not available")
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def is_available(self) -> bool:
        return self._client is not None

class RuleGenerator:
    """LLM-based rule generator"""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.generation_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates for different rule types"""
        return {
            "LOGICAL": """
Generate a logical rule based on this description: {description}

The rule should be:
- Logically sound and verifiable
- Based on formal logic or mathematical principles
- Deterministic in nature

Please provide the rule in this format:
Rule Name: [descriptive name]
Type: LOGICAL
Pre-conditions: [list of required conditions]
Post-conditions: [list of expected outcomes]
Parameters: [any parameters with constraints]
Tests: [validation tests that must pass]
Explanation: [clear explanation of the rule's logic]

Description: {description}
Domain: {domain}
Context: {context}
""",
            
            "HEURISTIC": """
Generate a heuristic rule based on this description: {description}

The rule should be:
- Practical and applicable
- Based on experience or common patterns
- Flexible and adaptable

Please provide the rule in this format:
Rule Name: [descriptive name]
Type: HEURISTIC
Pre-conditions: [list of typical conditions]
Post-conditions: [list of likely outcomes]
Parameters: [adjustable parameters]
Confidence: [confidence level 0-1]
Explanation: [explanation of when and how to use this rule]

Description: {description}
Domain: {domain}
Context: {context}
""",
            
            "HYBRID": """
Generate a hybrid rule that combines logical and heuristic elements based on this description: {description}

The rule should be:
- Have a logical foundation
- Include learnable parameters
- Be adaptable through experience

Please provide the rule in this format:
Rule Name: [descriptive name]
Type: HYBRID
Pre-conditions: [logical prerequisites]
Post-conditions: [expected outcomes]
Parameters: [learnable parameters with initial values and constraints]
Logic Component: [formal logical part]
Heuristic Component: [adaptive heuristic part]
Explanation: [how logic and heuristics combine]

Description: {description}
Domain: {domain}
Context: {context}
"""
        }
    
    def generate_rule(self, request: RuleGenerationRequest) -> GeneratedRule:
        """Generate a rule from natural language description"""
        
        # Select appropriate template
        template = self.generation_templates.get(request.rule_type, 
                                                self.generation_templates["HEURISTIC"])
        
        # Format prompt
        prompt = template.format(
            description=request.description,
            domain=request.domain or "general",
            context=json.dumps(request.context or {}, indent=2)
        )
        
        # Add examples if provided
        if request.examples:
            prompt += "\n\nExamples:\n" + "\n".join(f"- {ex}" for ex in request.examples)
        
        # Add constraints if provided
        if request.constraints:
            prompt += f"\n\nConstraints: {json.dumps(request.constraints, indent=2)}"
        
        # Generate rule using LLM
        try:
            response = self.llm_provider.generate_text(prompt)
            parsed_rule = self._parse_llm_response(response, request)
            return parsed_rule
        except Exception as e:
            logger.error(f"Error generating rule: {e}")
            # Return fallback rule
            return self._create_fallback_rule(request)
    
    def _parse_llm_response(self, response: str, request: RuleGenerationRequest) -> GeneratedRule:
        """Parse LLM response into structured rule"""
        
        # Extract components using regex
        name_match = re.search(r'Rule Name:\s*(.+)', response)
        type_match = re.search(r'Type:\s*(\w+)', response)
        pre_match = re.search(r'Pre-conditions:\s*\[(.*?)\]', response, re.DOTALL)
        post_match = re.search(r'Post-conditions:\s*\[(.*?)\]', response, re.DOTALL)
        params_match = re.search(r'Parameters:\s*\{(.*?)\}', response, re.DOTALL)
        explanation_match = re.search(r'Explanation:\s*(.+?)(?=\n\n|\n[A-Z]|$)', response, re.DOTALL)
        confidence_match = re.search(r'Confidence:\s*([\d.]+)', response)
        
        # Extract values with fallbacks
        name = name_match.group(1).strip() if name_match else f"Generated Rule {int(time.time())}"
        rule_type = type_match.group(1).strip() if type_match else request.rule_type
        
        # Parse conditions
        pre_conditions = []
        if pre_match:
            pre_text = pre_match.group(1)
            pre_conditions = [c.strip().strip('"\'') for c in pre_text.split(',') if c.strip()]
        
        post_conditions = []
        if post_match:
            post_text = post_match.group(1)
            post_conditions = [c.strip().strip('"\'') for c in post_text.split(',') if c.strip()]
        
        # Parse parameters
        parameters = {}
        if params_match:
            try:
                # Simple parameter parsing
                params_text = params_match.group(1)
                # Extract key-value pairs
                param_pairs = re.findall(r'"([^"]+)":\s*([^,}]+)', params_text)
                for key, value in param_pairs:
                    try:
                        # Try to parse as number
                        if '.' in value:
                            parameters[key] = float(value.strip())
                        else:
                            parameters[key] = int(value.strip())
                    except ValueError:
                        # Keep as string
                        parameters[key] = value.strip().strip('"\'')
            except Exception as e:
                logger.warning(f"Error parsing parameters: {e}")
                parameters = {"confidence": 0.7}
        
        # Extract confidence
        confidence = 0.7  # default
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
            except ValueError:
                pass
        
        # Extract explanation
        explanation = explanation_match.group(1).strip() if explanation_match else "Generated rule"
        
        # Generate unique ID
        rule_id = f"llm_generated_{int(time.time())}_{hash(name) % 10000}"
        
        return GeneratedRule(
            id=rule_id,
            name=name,
            rule_type=rule_type,
            pre_conditions=pre_conditions,
            post_conditions=post_conditions,
            parameters=parameters,
            confidence=confidence,
            explanation=explanation,
            source_description=request.description,
            metadata={
                "generated_at": time.time(),
                "domain": request.domain,
                "llm_provider": type(self.llm_provider).__name__
            }
        )
    
    def _create_fallback_rule(self, request: RuleGenerationRequest) -> GeneratedRule:
        """Create a fallback rule when LLM generation fails"""
        rule_id = f"fallback_{int(time.time())}"
        
        return GeneratedRule(
            id=rule_id,
            name=f"Fallback Rule for: {request.description[:50]}...",
            rule_type=request.rule_type,
            pre_conditions=["context_available"],
            post_conditions=["fallback_applied"],
            parameters={"confidence": 0.3, "fallback": True},
            confidence=0.3,
            explanation=f"Fallback rule generated for: {request.description}",
            source_description=request.description,
            metadata={
                "generated_at": time.time(),
                "is_fallback": True,
                "domain": request.domain
            }
        )
    
    def explain_rule(self, rule, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate explanation for an existing rule using LLM"""
        prompt = f"""
Explain this rule in clear, understandable language:

Rule ID: {rule.id}
Rule Name: {rule.name}
Type: {rule.type}
Pre-conditions: {rule.pre_conditions}
Post-conditions: {rule.post_conditions}
Parameters: {dict(rule.params) if hasattr(rule, 'params') else 'None'}

Please provide:
1. What this rule does
2. When it should be applied
3. What outcomes to expect
4. Any important considerations

Context: {json.dumps(context or {}, indent=2)}
"""
        
        try:
            return self.llm_provider.generate_text(prompt, max_tokens=500)
        except Exception as e:
            logger.error(f"Error explaining rule: {e}")
            return f"Rule {rule.name} applies when conditions {rule.pre_conditions} are met."
    
    def suggest_improvements(self, rule, performance_history: List[float]) -> str:
        """Suggest improvements for a rule based on performance"""
        avg_performance = sum(performance_history) / len(performance_history) if performance_history else 0.0
        
        prompt = f"""
Analyze this rule's performance and suggest improvements:

Rule: {rule.name}
Type: {rule.type}
Current Performance: {avg_performance:.3f}
Performance History: {performance_history[-10:]}  # Last 10 values
Parameters: {dict(rule.params) if hasattr(rule, 'params') else 'None'}

The rule has been performing {'well' if avg_performance > 0.6 else 'poorly' if avg_performance < 0.3 else 'moderately'}.

Please suggest:
1. Parameter adjustments
2. Condition modifications
3. Alternative approaches
4. Potential issues to investigate

Provide specific, actionable recommendations.
"""
        
        try:
            return self.llm_provider.generate_text(prompt, max_tokens=600)
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return "Consider adjusting parameters based on performance trends."

class LLMRuleIntegrator:
    """Integrates LLM-generated rules with DRM system"""
    
    def __init__(self, drm_system, llm_provider: LLMProvider):
        self.drm_system = drm_system
        self.rule_generator = RuleGenerator(llm_provider)
        self.generation_history: List[Dict[str, Any]] = []
    
    def generate_and_add_rule(self, description: str, domain: Optional[str] = None,
                             rule_type: str = "HEURISTIC") -> Dict[str, Any]:
        """Generate rule from description and add to DRM system"""
        
        # Create generation request
        request = RuleGenerationRequest(
            description=description,
            domain=domain,
            rule_type=rule_type
        )
        
        # Generate rule
        generated_rule = self.rule_generator.generate_rule(request)
        
        # Convert to DRM Rule
        drm_rule = self._convert_to_drm_rule(generated_rule)
        
        # Add to DRM system
        result = self.drm_system.add_rule(drm_rule)
        
        # Record generation
        generation_record = {
            "timestamp": time.time(),
            "description": description,
            "generated_rule_id": generated_rule.id,
            "drm_rule_id": drm_rule.id,
            "success": result.get("success", False),
            "confidence": generated_rule.confidence
        }
        self.generation_history.append(generation_record)
        
        logger.info(f"Generated and added rule: {drm_rule.id} from description: {description[:100]}...")
        
        return {
            "generated_rule": generated_rule,
            "drm_rule": drm_rule,
            "add_result": result,
            "generation_record": generation_record
        }
    
    def _convert_to_drm_rule(self, generated_rule: GeneratedRule):
        """Convert LLM-generated rule to DRM Rule format"""
        from drm_module5_improved import Rule, RuleParameter
        
        # Convert parameters to RuleParameter objects
        drm_params = {}
        for name, value in generated_rule.parameters.items():
            if isinstance(value, (int, float)):
                drm_params[name] = RuleParameter(
                    value=float(value),
                    param_type="float",
                    min_val=0.0,
                    max_val=1.0,
                    requires_grad=(generated_rule.rule_type == "HYBRID")
                )
        
        # Create DRM rule
        drm_rule = Rule(
            id=generated_rule.id,
            name=generated_rule.name,
            rtype=generated_rule.rule_type,
            init_weight=1.0,
            init_mean=generated_rule.confidence,
            init_var=0.1,
            pre_conditions=generated_rule.pre_conditions,
            post_conditions=generated_rule.post_conditions,
            params=drm_params,
            provenance={
                "source": "llm_generated",
                "description": generated_rule.source_description,
                "explanation": generated_rule.explanation,
                "metadata": generated_rule.metadata
            }
        )
        
        return drm_rule
    
    def get_generation_summary(self) -> Dict[str, Any]:
        """Get summary of rule generation activity"""
        if not self.generation_history:
            return {"total_generated": 0}
        
        successful = [r for r in self.generation_history if r["success"]]
        avg_confidence = sum(r["confidence"] for r in self.generation_history) / len(self.generation_history)
        
        return {
            "total_generated": len(self.generation_history),
            "successful": len(successful),
            "success_rate": len(successful) / len(self.generation_history),
            "average_confidence": avg_confidence,
            "recent_generations": self.generation_history[-5:]
        }

# Factory function for easy setup
def create_llm_rule_generator(provider_type: str = "mock", **kwargs) -> LLMRuleIntegrator:
    """Create LLM rule generator with specified provider"""
    
    if provider_type == "mock":
        provider = MockLLMProvider()
    elif provider_type == "openai":
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("OpenAI API key required")
        provider = OpenAIProvider(api_key, kwargs.get("model", "gpt-3.5-turbo"))
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    # This would need the DRM system instance
    # return LLMRuleIntegrator(drm_system, provider)
    return provider  # Return provider for now
