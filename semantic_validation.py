"""semantic_validation.py
Advanced semantic validation system for DRM rules.

Features:
- Domain-specific validators (physics, mathematics, logic, chemistry, etc.)
- Automatic test generation based on rule semantics
- Symbolic reasoning and constraint checking
- Integration with external knowledge bases
- Multi-level validation (syntax, semantics, pragmatics)
"""

import re
import math
import sympy as sp
from sympy import symbols, solve, simplify, diff, integrate
from typing import Dict, List, Optional, Callable, Any, Tuple, Set, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Levels of semantic validation"""
    SYNTAX = "syntax"
    SEMANTICS = "semantics"
    PRAGMATICS = "pragmatics"
    DOMAIN_SPECIFIC = "domain_specific"

class ValidationResult:
    """Result of semantic validation"""
    def __init__(self):
        self.passed = True
        self.level = ValidationLevel.SYNTAX
        self.score = 1.0
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.suggestions: List[str] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_error(self, message: str, level: ValidationLevel = ValidationLevel.SEMANTICS):
        self.errors.append(f"[{level.value}] {message}")
        self.passed = False
        self.score *= 0.5
    
    def add_warning(self, message: str, level: ValidationLevel = ValidationLevel.SEMANTICS):
        self.warnings.append(f"[{level.value}] {message}")
        self.score *= 0.9
    
    def add_suggestion(self, message: str):
        self.suggestions.append(message)

class DomainValidator(ABC):
    """Abstract base class for domain-specific validators"""
    
    @abstractmethod
    def validate(self, rule, context: Dict[str, Any]) -> ValidationResult:
        pass
    
    @abstractmethod
    def generate_tests(self, rule) -> List[Callable]:
        pass
    
    @abstractmethod
    def get_domain_keywords(self) -> Set[str]:
        pass

class PhysicsValidator(DomainValidator):
    """Validator for physics-related rules"""
    
    def __init__(self):
        self.fundamental_constants = {
            'c': 299792458,  # speed of light
            'h': 6.62607015e-34,  # Planck constant
            'k_B': 1.380649e-23,  # Boltzmann constant
            'e': 1.602176634e-19,  # elementary charge
            'G': 6.67430e-11,  # gravitational constant
        }
        
        self.units = {
            'length': ['m', 'km', 'cm', 'mm', 'nm'],
            'time': ['s', 'ms', 'ns', 'min', 'h'],
            'mass': ['kg', 'g', 'mg', 'u'],
            'energy': ['J', 'eV', 'cal', 'kWh'],
            'force': ['N', 'dyn', 'lbf'],
            'power': ['W', 'hp', 'kW'],
            'temperature': ['K', 'C', 'F'],
            'pressure': ['Pa', 'atm', 'bar', 'mmHg']
        }
    
    def validate(self, rule, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult()
        result.level = ValidationLevel.DOMAIN_SPECIFIC
        
        # Check for physics keywords
        text = self._extract_text(rule, context)
        physics_terms = self._find_physics_terms(text)
        
        if not physics_terms:
            result.add_warning("No physics terms detected")
            return result
        
        # Validate dimensional consistency
        self._validate_dimensions(rule, context, result)
        
        # Validate conservation laws
        self._validate_conservation_laws(rule, context, result)
        
        # Validate physical constraints
        self._validate_physical_constraints(rule, context, result)
        
        result.metadata['physics_terms'] = list(physics_terms)
        return result
    
    def _extract_text(self, rule, context: Dict[str, Any]) -> str:
        """Extract text from rule and context"""
        text_parts = []
        
        # From rule
        text_parts.extend(rule.pre_conditions)
        text_parts.extend(rule.post_conditions)
        text_parts.append(rule.name or "")
        
        # From context
        if 'text' in context:
            text_parts.append(context['text'])
        if 'description' in context:
            text_parts.append(context['description'])
        
        return " ".join(text_parts).lower()
    
    def _find_physics_terms(self, text: str) -> Set[str]:
        """Find physics-related terms in text"""
        physics_keywords = self.get_domain_keywords()
        found_terms = set()
        
        for keyword in physics_keywords:
            if keyword in text:
                found_terms.add(keyword)
        
        return found_terms
    
    def _validate_dimensions(self, rule, context: Dict[str, Any], result: ValidationResult):
        """Validate dimensional consistency"""
        text = self._extract_text(rule, context)
        
        # Look for equations
        equations = re.findall(r'([a-zA-Z_]\w*)\s*=\s*([^,\n]+)', text)
        
        for var, expr in equations:
            try:
                # Simple dimensional analysis
                if self._has_dimensional_inconsistency(expr):
                    result.add_error(f"Dimensional inconsistency in equation: {var} = {expr}")
            except Exception as e:
                result.add_warning(f"Could not analyze dimensions for: {var} = {expr}")
    
    def _has_dimensional_inconsistency(self, expression: str) -> bool:
        """Check for obvious dimensional inconsistencies"""
        # Simple heuristics - can be expanded
        
        # Check for addition/subtraction of different units
        if '+' in expression or '-' in expression:
            terms = re.split(r'[+\-]', expression)
            units_found = []
            for term in terms:
                for unit_type, unit_list in self.units.items():
                    for unit in unit_list:
                        if unit in term:
                            units_found.append(unit_type)
            
            # If different unit types are being added/subtracted
            if len(set(units_found)) > 1:
                return True
        
        return False
    
    def _validate_conservation_laws(self, rule, context: Dict[str, Any], result: ValidationResult):
        """Validate conservation laws"""
        text = self._extract_text(rule, context)
        
        # Energy conservation
        if any(term in text for term in ['energy', 'kinetic', 'potential', 'thermal']):
            if 'create' in text and 'energy' in text:
                result.add_error("Violation of energy conservation - energy cannot be created")
            if 'destroy' in text and 'energy' in text:
                result.add_error("Violation of energy conservation - energy cannot be destroyed")
        
        # Mass conservation
        if any(term in text for term in ['mass', 'matter']):
            if 'create' in text and 'mass' in text:
                result.add_warning("Potential mass conservation issue - check for nuclear reactions")
        
        # Momentum conservation
        if 'momentum' in text:
            if 'external force' not in text and 'change' in text:
                result.add_suggestion("Consider momentum conservation in isolated systems")
    
    def _validate_physical_constraints(self, rule, context: Dict[str, Any], result: ValidationResult):
        """Validate physical constraints and limits"""
        text = self._extract_text(rule, context)
        
        # Speed of light constraint
        if 'velocity' in text or 'speed' in text:
            # Look for numerical values
            numbers = re.findall(r'\d+\.?\d*', text)
            for num in numbers:
                try:
                    value = float(num)
                    if value > self.fundamental_constants['c']:
                        result.add_error(f"Velocity {value} exceeds speed of light")
                except ValueError:
                    continue
        
        # Temperature constraints
        if 'temperature' in text:
            if 'absolute zero' in text or 'below zero' in text:
                result.add_warning("Check temperature constraints - cannot go below absolute zero")
    
    def generate_tests(self, rule) -> List[Callable]:
        """Generate physics-specific tests"""
        tests = []
        
        # Test for energy conservation
        def test_energy_conservation(rule, context):
            text = self._extract_text(rule, context)
            if 'energy' in text:
                # Simple check - no creation/destruction of energy
                return not ('create energy' in text or 'destroy energy' in text)
            return True
        
        # Test for dimensional consistency
        def test_dimensional_consistency(rule, context):
            result = ValidationResult()
            self._validate_dimensions(rule, context, result)
            return result.passed
        
        # Test for physical limits
        def test_physical_limits(rule, context):
            result = ValidationResult()
            self._validate_physical_constraints(rule, context, result)
            return result.passed
        
        tests.extend([test_energy_conservation, test_dimensional_consistency, test_physical_limits])
        return tests
    
    def get_domain_keywords(self) -> Set[str]:
        """Get physics domain keywords"""
        return {
            'energy', 'kinetic', 'potential', 'thermal', 'mechanical',
            'force', 'acceleration', 'velocity', 'momentum', 'mass',
            'gravity', 'electromagnetic', 'electric', 'magnetic',
            'wave', 'frequency', 'wavelength', 'amplitude',
            'temperature', 'pressure', 'volume', 'density',
            'conservation', 'entropy', 'thermodynamics',
            'quantum', 'photon', 'electron', 'atom', 'nucleus',
            'relativity', 'spacetime', 'field'
        }

class MathematicsValidator(DomainValidator):
    """Validator for mathematics-related rules"""
    
    def __init__(self):
        self.mathematical_constants = {
            'pi': math.pi,
            'e': math.e,
            'phi': (1 + math.sqrt(5)) / 2,  # golden ratio
        }
    
    def validate(self, rule, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult()
        result.level = ValidationLevel.DOMAIN_SPECIFIC
        
        text = self._extract_text(rule, context)
        
        # Validate mathematical expressions
        self._validate_expressions(text, result)
        
        # Validate logical consistency
        self._validate_logical_consistency(text, result)
        
        # Validate mathematical properties
        self._validate_mathematical_properties(text, result)
        
        return result
    
    def _extract_text(self, rule, context: Dict[str, Any]) -> str:
        """Extract text from rule and context"""
        text_parts = []
        text_parts.extend(rule.pre_conditions)
        text_parts.extend(rule.post_conditions)
        text_parts.append(rule.name or "")
        
        if 'text' in context:
            text_parts.append(context['text'])
        
        return " ".join(text_parts)
    
    def _validate_expressions(self, text: str, result: ValidationResult):
        """Validate mathematical expressions using SymPy"""
        # Find mathematical expressions
        expressions = re.findall(r'([a-zA-Z_]\w*)\s*=\s*([^,\n]+)', text)
        
        for var, expr in expressions:
            try:
                # Parse with SymPy
                parsed_expr = sp.sympify(expr)
                
                # Check for undefined variables
                free_symbols = parsed_expr.free_symbols
                if len(free_symbols) > 10:  # Too many variables
                    result.add_warning(f"Expression {expr} has many undefined variables")
                
                # Check for mathematical validity
                if parsed_expr.has(sp.zoo) or parsed_expr.has(sp.nan):
                    result.add_error(f"Expression {expr} contains undefined values")
                
            except Exception as e:
                result.add_warning(f"Could not parse mathematical expression: {expr}")
    
    def _validate_logical_consistency(self, text: str, result: ValidationResult):
        """Validate logical consistency of mathematical statements"""
        # Check for contradictions
        if 'always' in text and 'never' in text:
            result.add_warning("Potential logical contradiction with 'always' and 'never'")
        
        # Check for tautologies
        if 'if and only if' in text:
            result.add_suggestion("Verify bidirectional implication is correct")
    
    def _validate_mathematical_properties(self, text: str, result: ValidationResult):
        """Validate mathematical properties and theorems"""
        # Check for well-known mathematical facts
        if 'prime number' in text:
            if '1 is prime' in text or '1 is a prime' in text:
                result.add_error("1 is not considered a prime number")
        
        if 'derivative' in text:
            if 'constant' in text and 'derivative' in text:
                if 'zero' not in text:
                    result.add_suggestion("Derivative of a constant is zero")
    
    def generate_tests(self, rule) -> List[Callable]:
        """Generate mathematics-specific tests"""
        tests = []
        
        def test_expression_validity(rule, context):
            text = self._extract_text(rule, context)
            expressions = re.findall(r'([a-zA-Z_]\w*)\s*=\s*([^,\n]+)', text)
            
            for var, expr in expressions:
                try:
                    sp.sympify(expr)
                except:
                    return False
            return True
        
        def test_mathematical_consistency(rule, context):
            text = self._extract_text(rule, context)
            # Simple consistency checks
            if '1 is prime' in text.lower():
                return False
            if 'divide by zero' in text.lower():
                return False
            return True
        
        tests.extend([test_expression_validity, test_mathematical_consistency])
        return tests
    
    def get_domain_keywords(self) -> Set[str]:
        """Get mathematics domain keywords"""
        return {
            'equation', 'function', 'derivative', 'integral', 'limit',
            'theorem', 'proof', 'lemma', 'corollary', 'axiom',
            'algebra', 'geometry', 'calculus', 'topology', 'analysis',
            'matrix', 'vector', 'scalar', 'tensor',
            'prime', 'composite', 'integer', 'rational', 'irrational',
            'continuous', 'differentiable', 'integrable',
            'convergent', 'divergent', 'series', 'sequence',
            'probability', 'statistics', 'distribution'
        }

class LogicValidator(DomainValidator):
    """Validator for logical reasoning rules"""

    def __init__(self):
        self.logical_operators = {
            'and', 'or', 'not', 'implies', 'if', 'then', 'else',
            'all', 'some', 'exists', 'forall', 'therefore'
        }

        self.logical_fallacies = {
            'ad hominem', 'straw man', 'false dichotomy', 'slippery slope',
            'circular reasoning', 'appeal to authority', 'hasty generalization'
        }

    def validate(self, rule, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult()
        result.level = ValidationLevel.DOMAIN_SPECIFIC

        text = self._extract_text(rule, context).lower()

        # Check for logical structure
        self._validate_logical_structure(text, result)

        # Check for logical fallacies
        self._validate_logical_fallacies(text, result)

        # Validate inference patterns
        self._validate_inference_patterns(text, result)

        return result

    def _extract_text(self, rule, context: Dict[str, Any]) -> str:
        text_parts = []
        text_parts.extend(rule.pre_conditions)
        text_parts.extend(rule.post_conditions)
        text_parts.append(rule.name or "")

        if 'text' in context:
            text_parts.append(context['text'])

        return " ".join(text_parts)

    def _validate_logical_structure(self, text: str, result: ValidationResult):
        """Validate logical structure of statements"""
        # Check for proper conditional structure
        if 'if' in text and 'then' not in text:
            result.add_warning("Conditional statement may be incomplete - missing 'then'")

        # Check for quantifier scope
        if 'all' in text and 'some' in text:
            result.add_suggestion("Check quantifier scope for potential ambiguity")

        # Check for negation clarity
        if text.count('not') > 2:
            result.add_warning("Multiple negations may cause confusion")

    def _validate_logical_fallacies(self, text: str, result: ValidationResult):
        """Check for common logical fallacies"""
        for fallacy in self.logical_fallacies:
            if fallacy in text:
                result.add_error(f"Potential logical fallacy detected: {fallacy}")

        # Check for circular reasoning patterns
        if 'because' in text:
            # Simple heuristic for circular reasoning
            parts = text.split('because')
            if len(parts) == 2:
                premise = parts[0].strip()
                conclusion = parts[1].strip()
                if premise in conclusion or conclusion in premise:
                    result.add_warning("Potential circular reasoning detected")

    def _validate_inference_patterns(self, text: str, result: ValidationResult):
        """Validate common inference patterns"""
        # Modus ponens: If P then Q, P, therefore Q
        if 'if' in text and 'then' in text and 'therefore' in text:
            result.add_suggestion("Verify modus ponens inference pattern is valid")

        # Universal instantiation
        if 'all' in text and 'therefore' in text:
            result.add_suggestion("Check universal instantiation is properly applied")

    def generate_tests(self, rule) -> List[Callable]:
        """Generate logic-specific tests"""
        tests = []

        def test_logical_consistency(rule, context):
            text = self._extract_text(rule, context).lower()

            # Check for obvious contradictions
            if 'always true' in text and 'always false' in text:
                return False
            if 'all' in text and 'none' in text:
                # Could be contradiction depending on context
                return 'except' in text or 'but' in text

            return True

        def test_fallacy_free(rule, context):
            text = self._extract_text(rule, context).lower()

            for fallacy in self.logical_fallacies:
                if fallacy in text:
                    return False

            return True

        tests.extend([test_logical_consistency, test_fallacy_free])
        return tests

    def get_domain_keywords(self) -> Set[str]:
        return self.logical_operators | {
            'premise', 'conclusion', 'argument', 'valid', 'sound',
            'necessary', 'sufficient', 'contradiction', 'tautology',
            'inference', 'deduction', 'induction', 'abduction'
        }

class ChemistryValidator(DomainValidator):
    """Validator for chemistry-related rules"""

    def __init__(self):
        self.periodic_elements = {
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca'
            # Add more as needed
        }

        self.chemical_properties = {
            'atomic_number', 'atomic_mass', 'electronegativity',
            'ionization_energy', 'electron_affinity', 'valence'
        }

    def validate(self, rule, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult()
        result.level = ValidationLevel.DOMAIN_SPECIFIC

        text = self._extract_text(rule, context)

        # Validate chemical formulas
        self._validate_chemical_formulas(text, result)

        # Validate chemical reactions
        self._validate_chemical_reactions(text, result)

        # Validate chemical properties
        self._validate_chemical_properties(text, result)

        return result

    def _extract_text(self, rule, context: Dict[str, Any]) -> str:
        text_parts = []
        text_parts.extend(rule.pre_conditions)
        text_parts.extend(rule.post_conditions)
        text_parts.append(rule.name or "")

        if 'text' in context:
            text_parts.append(context['text'])

        return " ".join(text_parts)

    def _validate_chemical_formulas(self, text: str, result: ValidationResult):
        """Validate chemical formulas"""
        # Find potential chemical formulas
        formulas = re.findall(r'\b[A-Z][a-z]?[0-9]*(?:[A-Z][a-z]?[0-9]*)*\b', text)

        for formula in formulas:
            # Check if elements exist
            elements = re.findall(r'[A-Z][a-z]?', formula)
            for element in elements:
                if element not in self.periodic_elements:
                    result.add_warning(f"Unknown chemical element: {element}")

    def _validate_chemical_reactions(self, text: str, result: ValidationResult):
        """Validate chemical reactions"""
        # Look for reaction arrows
        if '->' in text or '→' in text:
            result.add_suggestion("Verify chemical reaction is balanced")

        # Check for conservation of mass
        if 'reaction' in text and 'mass' in text:
            if 'lost' in text or 'gained' in text:
                result.add_warning("Check mass conservation in chemical reactions")

    def _validate_chemical_properties(self, text: str, result: ValidationResult):
        """Validate chemical properties"""
        for prop in self.chemical_properties:
            if prop in text:
                result.add_suggestion(f"Verify {prop} values are within expected ranges")

    def generate_tests(self, rule) -> List[Callable]:
        """Generate chemistry-specific tests"""
        tests = []

        def test_element_validity(rule, context):
            text = self._extract_text(rule, context)
            formulas = re.findall(r'\b[A-Z][a-z]?[0-9]*(?:[A-Z][a-z]?[0-9]*)*\b', text)

            for formula in formulas:
                elements = re.findall(r'[A-Z][a-z]?', formula)
                for element in elements:
                    if element not in self.periodic_elements:
                        return False
            return True

        tests.append(test_element_validity)
        return tests

    def get_domain_keywords(self) -> Set[str]:
        return {
            'atom', 'molecule', 'compound', 'element', 'reaction',
            'bond', 'ionic', 'covalent', 'metallic', 'hydrogen',
            'oxidation', 'reduction', 'catalyst', 'enzyme',
            'acid', 'base', 'pH', 'buffer', 'solution',
            'concentration', 'molarity', 'molality',
            'thermodynamics', 'kinetics', 'equilibrium'
        } | self.periodic_elements

class SemanticValidationEngine:
    """Main semantic validation engine"""

    def __init__(self):
        self.validators: Dict[str, DomainValidator] = {
            'physics': PhysicsValidator(),
            'mathematics': MathematicsValidator(),
            'logic': LogicValidator(),
            'chemistry': ChemistryValidator()
        }

        self.validation_history: List[Dict[str, Any]] = []

    def register_validator(self, domain: str, validator: DomainValidator):
        """Register a new domain validator"""
        self.validators[domain] = validator
        logger.info(f"Registered validator for domain: {domain}")

    def detect_domain(self, rule, context: Dict[str, Any]) -> List[str]:
        """Automatically detect relevant domains for a rule"""
        text = self._extract_text(rule, context).lower()
        detected_domains = []

        for domain, validator in self.validators.items():
            keywords = validator.get_domain_keywords()
            keyword_count = sum(1 for keyword in keywords if keyword.lower() in text)

            # If more than 2 domain keywords found, consider it relevant
            if keyword_count >= 2:
                detected_domains.append(domain)

        return detected_domains

    def _extract_text(self, rule, context: Dict[str, Any]) -> str:
        """Extract text from rule and context"""
        text_parts = []
        text_parts.extend(rule.pre_conditions)
        text_parts.extend(rule.post_conditions)
        text_parts.append(rule.name or "")

        if 'text' in context:
            text_parts.append(context['text'])
        if 'description' in context:
            text_parts.append(context['description'])

        return " ".join(text_parts)

    def validate_rule(self, rule, context: Dict[str, Any],
                     domains: Optional[List[str]] = None) -> Dict[str, ValidationResult]:
        """Validate a rule across multiple domains"""
        if domains is None:
            domains = self.detect_domain(rule, context)

        results = {}

        for domain in domains:
            if domain in self.validators:
                try:
                    result = self.validators[domain].validate(rule, context)
                    results[domain] = result
                    logger.debug(f"Validated rule {rule.id} in domain {domain}: "
                               f"passed={result.passed}, score={result.score}")
                except Exception as e:
                    logger.error(f"Error validating rule {rule.id} in domain {domain}: {e}")
                    error_result = ValidationResult()
                    error_result.add_error(f"Validation error: {str(e)}")
                    results[domain] = error_result

        # Store validation history
        self.validation_history.append({
            'rule_id': rule.id,
            'domains': domains,
            'results': {d: {'passed': r.passed, 'score': r.score,
                           'errors': len(r.errors), 'warnings': len(r.warnings)}
                       for d, r in results.items()},
            'timestamp': context.get('timestamp', 'unknown')
        })

        return results

    def generate_tests_for_rule(self, rule, domains: Optional[List[str]] = None) -> Dict[str, List[Callable]]:
        """Generate domain-specific tests for a rule"""
        if domains is None:
            domains = self.detect_domain(rule, {})

        tests = {}

        for domain in domains:
            if domain in self.validators:
                try:
                    domain_tests = self.validators[domain].generate_tests(rule)
                    tests[domain] = domain_tests
                    logger.info(f"Generated {len(domain_tests)} tests for rule {rule.id} in domain {domain}")
                except Exception as e:
                    logger.error(f"Error generating tests for rule {rule.id} in domain {domain}: {e}")
                    tests[domain] = []

        return tests

    def get_validation_summary(self, rule_id: Optional[str] = None) -> Dict[str, Any]:
        """Get validation summary statistics"""
        relevant_history = self.validation_history
        if rule_id:
            relevant_history = [h for h in self.validation_history if h['rule_id'] == rule_id]

        if not relevant_history:
            return {"total_validations": 0}

        total_validations = len(relevant_history)
        domain_stats = defaultdict(lambda: {'total': 0, 'passed': 0, 'avg_score': 0.0})

        for entry in relevant_history:
            for domain, result in entry['results'].items():
                domain_stats[domain]['total'] += 1
                if result['passed']:
                    domain_stats[domain]['passed'] += 1

        # Calculate averages
        for domain, stats in domain_stats.items():
            if stats['total'] > 0:
                stats['pass_rate'] = stats['passed'] / stats['total']

        return {
            'total_validations': total_validations,
            'domain_statistics': dict(domain_stats),
            'most_validated_domains': sorted(domain_stats.keys(),
                                           key=lambda d: domain_stats[d]['total'],
                                           reverse=True)[:5]
        }

# Utility functions
def create_semantic_tests_for_drm_rule(rule, validation_engine: SemanticValidationEngine) -> List[Callable]:
    """Create semantic tests for a DRM rule"""
    all_tests = []

    # Generate domain-specific tests
    domain_tests = validation_engine.generate_tests_for_rule(rule)

    for domain, tests in domain_tests.items():
        for test in tests:
            # Wrap test with domain information
            def wrapped_test(rule, context, original_test=test, domain_name=domain):
                try:
                    return original_test(rule, context)
                except Exception as e:
                    logger.error(f"Semantic test failed in domain {domain_name}: {e}")
                    return False

            wrapped_test.__name__ = f"semantic_{domain}_test"
            all_tests.append(wrapped_test)

    return all_tests

def integrate_semantic_validation_with_drm(drm_system, validation_engine: SemanticValidationEngine):
    """Integrate semantic validation with DRM system"""

    # Override the rule validation method
    original_validate_rule = drm_system.validate_rule

    def enhanced_validate_rule(rule_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Run original validation
        result = original_validate_rule(rule_id, context)

        if rule_id in drm_system.rules:
            rule = drm_system.rules[rule_id]

            # Run semantic validation
            semantic_results = validation_engine.validate_rule(rule, context)

            # Integrate results
            result['semantic_validation'] = {}
            overall_semantic_score = 1.0
            semantic_passed = True

            for domain, sem_result in semantic_results.items():
                result['semantic_validation'][domain] = {
                    'passed': sem_result.passed,
                    'score': sem_result.score,
                    'errors': sem_result.errors,
                    'warnings': sem_result.warnings,
                    'suggestions': sem_result.suggestions
                }

                overall_semantic_score *= sem_result.score
                semantic_passed = semantic_passed and sem_result.passed

            # Update overall decision based on semantic validation
            if not semantic_passed:
                result['quarantine'] = True
                result['reason'] = 'semantic_validation_failed'

            result['semantic_score'] = overall_semantic_score
            result['semantic_passed'] = semantic_passed

        return result

    # Replace the method
    drm_system.validate_rule = enhanced_validate_rule

    logger.info("Integrated semantic validation with DRM system")
