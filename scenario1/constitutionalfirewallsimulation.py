def _identify_violations(self, bill_text: str, articles: List[str]) -> List[str]:
        """Identify specific constitutional violations"""
        violations = []
        bill_lower = bill_text.lower()
        
        # Strong violation indicators
        violation_patterns = {
            "unreasonable search": ["warrantless", "without probable cause", "mass surveillance"],
            "speech suppression": ["prohibit speech", "censor", "ban expression"],
            "due process violation": ["immediate penalty", "no hearing", "summary judgment"],
            "equal protection": ["based on race", "discriminatory", "unequal treatment"]
        }
        
        for violation_type, patterns in violation_patterns.items():
            for pattern in patterns:
                if pattern in bill_lower:
                    violations.append(violation_type)
                    break
        
        return violations
    
    def _calculate_confidence(self, bill_text: str, concerns: List[str]) -> float:
        """Calculate confidence score for the analysis"""
        # Simple heuristic: more concerns and longer text = higher confidence
        base_confidence = 0.6
        concern_boost = len(concerns) * 0.1
        text_length_factor = min(len(bill_text) / 1000, 0.3)  # Cap at 0.3
        
        return min(base_confidence + concern_boost + text_length_factor, 1.0)
    
    def _extract_evidence(self, bill_text: str, violations: List[str]) -> List[str]:
        """Extract supporting evidence from bill text"""
        evidence = []
        sentences = bill_text.split('.')
        
        for violation in violations:
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in violation.split()):
                    evidence.append(sentence.strip())
                    break
        
        return evidence
    
    def _identify_mitigating_factors(self, bill_text: str) -> List[str]:
        """Identify factors that might mitigate constitutional concerns"""
        mitigating_factors = []
        bill_lower = bill_text.lower()
        
        mitigation_patterns = [
            ("judicial oversight", "judicial review"),
            ("sunset clause", "expires"),
            ("limited scope", "limited to"),
            ("constitutional safeguards", "constitutional protection"),
            ("due process protections", "due process")
        ]
        
        for factor, pattern in mitigation_patterns:
            if pattern in bill_lower:
                mitigating_factors.append(factor)
        
        return mitigating_factors

class XAIProcessor:
    """
    Explainable AI processor for generating human-readable explanations.
    Implements transparency and interpretability for the constitutional analysis.
    """
    
    def __init__(self):
        self.explanation_templates = self._load_explanation_templates()
    
    def _load_explanation_templates(self) -> Dict[str, str]:
        """Load templates for generating explanations"""
        return {
            "fourth_amendment": "The proposed legislation may violate Fourth Amendment protections by {reason}",
            "first_amendment": "This bill potentially restricts First Amendment rights through {reason}",
            "due_process": "The legislation appears to violate due process requirements by {reason}",
            "equal_protection": "The bill may violate equal protection principles through {reason}",
            "general": "Constitutional concern identified: {reason}"
        }
    
    def generate_explanations(self, analysis_result: AnalysisResult, 
                            violated_rules: List[str]) -> Tuple[List[str], float]:
        """
        Generate human-readable explanations for the constitutional analysis.
        Returns (explanations, xai_score)
        """
        explanations = []
        
        # Generate explanations for constitutional concerns
        for concern in analysis_result.constitutional_concerns:
            explanation = self._format_concern_explanation(concern)
            explanations.append(explanation)
        
        # Generate explanations for violated rules
        for rule in violated_rules:
            explanation = self._format_rule_violation(rule, analysis_result)
            explanations.append(explanation)
        
        # Add supporting evidence explanations
        for evidence in analysis_result.supporting_evidence[:3]:  # Limit to top 3
            explanations.append(f"Supporting evidence: {evidence}")
        
        # Calculate XAI score based on explanation quality and completeness
        xai_score = self._calculate_xai_score(explanations, analysis_result)
        
        return explanations, xai_score
    
    def _format_concern_explanation(self, concern: str) -> str:
        """Format a constitutional concern into a readable explanation"""
        return f"Analysis identified: {concern}"
    
    def _format_rule_violation(self, rule: str, analysis: AnalysisResult) -> str:
        """Format a rule violation into a readable explanation"""
        return f"Constitutional rule violated: {rule} (confidence: {analysis.confidence_score:.1f})"
    
    def _calculate_xai_score(self, explanations: List[str], analysis: AnalysisResult) -> float:
        """Calculate the XAI score based on explanation quality"""
        # Factors affecting XAI score:
        # 1. Number of explanations (completeness)
        # 2. Presence of supporting evidence
        # 3. Analysis confidence
        
        base_score = 0.5
        
        # Completeness factor
        completeness_factor = min(len(explanations) / 5, 0.3)
        
        # Evidence factor
        evidence_factor = min(len(analysis.supporting_evidence) / 3, 0.2)
        
        # Confidence factor
        confidence_factor = analysis.confidence_score * 0.3
        
        xai_score = base_score + completeness_factor + evidence_factor + confidence_factor
        return min(xai_score, 1.0)

class ConstitutionalFirewall:
    """
    Main Constitutional Firewall system that orchestrates the analysis process.
    """
    
    def __init__(self):
        self.rule_engine = RuleEngine()
        self.llm_analyzer = LLMAnalyzer()
        self.xai_processor = XAIProcessor()
        logger.info("Constitutional Firewall initialized")
    
    def analyze_legislation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis method that processes legislation through the constitutional firewall.
        
        Args:
            input_data: Dictionary containing bill_text, constitutional_articles, and ethics_matrix
            
        Returns:
            Dictionary containing verdict, xai_score, explanation, and justification_trace
        """
        try:
            # Validate input
            self._validate_input(input_data)
            
            # Extract input parameters
            bill_text = input_data["bill_text"]
            constitutional_articles = input_data["constitutional_articles"]
            ethics_matrix = input_data["ethics_matrix"]
            
            logger.info(f"Analyzing legislation with {len(bill_text)} characters")
            
            # Step 1: LLM Analysis
            llm_analysis = self.llm_analyzer.analyze_bill(bill_text, constitutional_articles)
            
            # Step 2: Rule Engine Evaluation
            violated_rules, rule_confidence = self.rule_engine.evaluate_rules(
                bill_text, constitutional_articles, ethics_matrix
            )
            
            # Step 3: Generate XAI Explanations
            explanations, xai_score = self.xai_processor.generate_explanations(
                llm_analysis, violated_rules
            )
            
            # Step 4: Make Final Verdict
            verdict = self._make_verdict(llm_analysis, violated_rules)
            
            # Step 5: Construct Output
            output = {
                "verdict": verdict.value,
                "xai_score": round(xai_score, 3),
                #!/usr/bin/env python3
"""
Constitutional Firewall Simulation System
=========================================

A comprehensive system that analyzes proposed legislation against constitutional
principles using an LLM and RuleEngine architecture with explainable AI capabilities.

Architecture Components:
- LLMAnalyzer: Natural language processing for bill analysis
- RuleEngine: Constitutional rule validation and logic
- XAIProcessor: Explainable AI for transparency
- ConstitutionalFirewall: Main orchestration system
"""

import json
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Verdict(Enum):
    CONSTITUTIONAL = "Constitutional"
    BLOCKED = "Blocked"

class EthicsMatrix(Enum):
    PRIVACY_WEIGHTED = "privacy-weighted"
    SECURITY_PRIORITIZED = "security-prioritized"
    BALANCED = "balanced"
    LIBERTY_FOCUSED = "liberty-focused"

@dataclass
class ConstitutionalRule:
    """Represents a constitutional rule or principle"""
    name: str
    article: str
    description: str
    keywords: List[str]
    severity: float  # 0.0 to 1.0, where 1.0 is most severe violation
    ethics_weight: Dict[str, float]  # Weights for different ethics matrices

@dataclass
class AnalysisResult:
    """Internal analysis result from LLM"""
    constitutional_concerns: List[str]
    violated_principles: List[str]
    confidence_score: float
    supporting_evidence: List[str]
    mitigating_factors: List[str]

class RuleEngine:
    """
    Constitutional rule validation engine.
    In a real implementation, this could interface with formal verification tools
    like Coq or Lean for mathematical proof verification.
    """
    
    def __init__(self):
        self.rules = self._initialize_constitutional_rules()
        
    def _initialize_constitutional_rules(self) -> Dict[str, ConstitutionalRule]:
        """Initialize the constitutional rules database"""
        rules = {
            "fourth_amendment": ConstitutionalRule(
                name="Fourth Amendment Protection",
                article="Fourth Amendment",
                description="Protection against unreasonable searches and seizures",
                keywords=["search", "seizure", "warrant", "probable cause", "privacy", "surveillance"],
                severity=0.9,
                ethics_weight={
                    "privacy-weighted": 1.0,
                    "security-prioritized": 0.6,
                    "balanced": 0.8,
                    "liberty-focused": 0.95
                }
            ),
            "first_amendment": ConstitutionalRule(
                name="First Amendment Rights",
                article="First Amendment",
                description="Freedom of speech, religion, press, assembly, and petition",
                keywords=["speech", "religion", "press", "assembly", "petition", "expression"],
                severity=0.95,
                ethics_weight={
                    "privacy-weighted": 0.8,
                    "security-prioritized": 0.7,
                    "balanced": 0.9,
                    "liberty-focused": 1.0
                }
            ),
            "due_process": ConstitutionalRule(
                name="Due Process Clause",
                article="Fifth Amendment, Fourteenth Amendment",
                description="Right to fair legal proceedings",
                keywords=["due process", "fair trial", "legal proceedings", "procedural fairness"],
                severity=0.85,
                ethics_weight={
                    "privacy-weighted": 0.7,
                    "security-prioritized": 0.8,
                    "balanced": 0.85,
                    "liberty-focused": 0.9
                }
            ),
            "equal_protection": ConstitutionalRule(
                name="Equal Protection Clause",
                article="Fourteenth Amendment",
                description="Equal protection under the law",
                keywords=["discrimination", "equal protection", "classification", "bias"],
                severity=0.9,
                ethics_weight={
                    "privacy-weighted": 0.8,
                    "security-prioritized": 0.8,
                    "balanced": 0.9,
                    "liberty-focused": 0.95
                }
            )
        }
        return rules
    
    def evaluate_rules(self, bill_text: str, constitutional_articles: List[str], 
                      ethics_matrix: str) -> Tuple[List[str], float]:
        """
        Evaluate constitutional rules against the bill text.
        Returns (violated_rules, confidence_score)
        """
        violated_rules = []
        total_confidence = 0.0
        rule_count = 0
        
        # Convert bill text to lowercase for keyword matching
        bill_lower = bill_text.lower()
        
        # Filter relevant rules based on constitutional articles mentioned
        relevant_rules = self._get_relevant_rules(constitutional_articles)
        
        for rule_key, rule in relevant_rules.items():
            # Check for keyword matches
            keyword_matches = sum(1 for keyword in rule.keywords if keyword in bill_lower)
            
            if keyword_matches > 0:
                # Calculate violation probability based on keyword density and ethics weighting
                keyword_density = keyword_matches / len(rule.keywords)
                ethics_weight = rule.ethics_weight.get(ethics_matrix, 0.8)
                
                violation_probability = keyword_density * rule.severity * ethics_weight
                
                # Threshold for rule violation (configurable)
                if violation_probability > 0.3:
                    violated_rules.append(rule.name)
                    total_confidence += violation_probability
                    rule_count += 1
        
        # Calculate average confidence
        avg_confidence = (total_confidence / rule_count * 100) if rule_count > 0 else 0.0
        
        return violated_rules, min(avg_confidence, 100.0)
    
    def _get_relevant_rules(self, constitutional_articles: List[str]) -> Dict[str, ConstitutionalRule]:
        """Filter rules based on mentioned constitutional articles"""
        relevant_rules = {}
        
        for rule_key, rule in self.rules.items():
            # Check if any of the rule's articles match the mentioned articles
            for article in constitutional_articles:
                if article.lower() in rule.article.lower():
                    relevant_rules[rule_key] = rule
                    break
        
        # If no specific matches, return all rules for comprehensive analysis
        if not relevant_rules:
            relevant_rules = self.rules
            
        return relevant_rules

class LLMAnalyzer:
    """
    Simulates LLM analysis of proposed legislation.
    In production, this would interface with actual LLM APIs.
    """
    
    def __init__(self):
        self.constitutional_knowledge = self._load_constitutional_knowledge()
    
    def _load_constitutional_knowledge(self) -> Dict[str, Any]:
        """Load constitutional knowledge base"""
        return {
            "amendments": {
                "First": "Freedom of speech, religion, press, assembly, petition",
                "Fourth": "Protection against unreasonable searches and seizures",
                "Fifth": "Due process, self-incrimination, double jeopardy",
                "Fourteenth": "Equal protection, due process"
            },
            "principles": [
                "separation of powers",
                "checks and balances",
                "federalism",
                "individual rights",
                "rule of law"
            ]
        }
    
    def analyze_bill(self, bill_text: str, constitutional_articles: List[str]) -> AnalysisResult:
        """
        Analyze bill text for constitutional concerns.
        This simulates what an LLM would return.
        """
        # Simulate LLM analysis with pattern matching and heuristics
        constitutional_concerns = self._identify_concerns(bill_text, constitutional_articles)
        violated_principles = self._identify_violations(bill_text, constitutional_articles)
        confidence_score = self._calculate_confidence(bill_text, constitutional_concerns)
        supporting_evidence = self._extract_evidence(bill_text, violated_principles)
        mitigating_factors = self._identify_mitigating_factors(bill_text)
        
        return AnalysisResult(
            constitutional_concerns=constitutional_concerns,
            violated_principles=violated_principles,
            confidence_score=confidence_score,
            supporting_evidence=supporting_evidence,
            mitigating_factors=mitigating_factors
        )
    
    def _identify_concerns(self, bill_text: str, articles: List[str]) -> List[str]:
        """Identify potential constitutional concerns"""
        concerns = []
        bill_lower = bill_text.lower()
        
        # Pattern-based concern identification
        concern_patterns = {
            "Fourth Amendment": ["surveillance", "search", "seizure", "monitoring", "tracking"],
            "First Amendment": ["speech restriction", "religious", "press regulation", "assembly"],
            "Due Process": ["without hearing", "immediate enforcement", "no appeal"],
            "Equal Protection": ["discrimination", "classification", "differential treatment"]
        }
        
        for article in articles:
            if article in concern_patterns:
                for pattern in concern_patterns[article]:
                    if pattern in bill_lower:
                        concerns.append(f"Potential {article} violation: {pattern} detected")
        
        return concerns
    
    def _identify_violations(self, bill_text: str, articles: List[str]) -> List[str]:
