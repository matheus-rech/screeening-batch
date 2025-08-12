import anthropic
from enum import Enum
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
import numpy as np

# ================== Conflict Types ==================
class ConflictType(Enum):
    CRITERIA_INTERPRETATION = "criteria_interpretation"
    METHODOLOGICAL_ASSESSMENT = "methodological_assessment"
    DATA_EXTRACTION = "data_extraction"
    QUALITY_RATING = "quality_rating"
    STATISTICAL_INTERPRETATION = "statistical_interpretation"

class ResolutionStrategy(Enum):
    EVIDENCE_BASED = "evidence_based"
    THIRD_REVIEWER = "third_reviewer"
    CONSERVATIVE = "conservative"  # When in doubt, include
    STRICT = "strict"  # When in doubt, exclude
    WEIGHTED_CONSENSUS = "weighted_consensus"
    EXPERT_RULES = "expert_rules"

# ================== Senior Evaluator Agent ==================
class SeniorEvaluatorAgent(ReviewComponent):
    """
    Senior evaluator with deep methodological expertise
    Acts as the ultimate arbiter for complex conflicts
    """
    
    def __init__(self):
        super().__init__(model="claude-opus-4-20250514")
        self.expertise_areas = [
            "statistical_methodology",
            "clinical_trial_design", 
            "systematic_review_methods",
            "risk_of_bias_assessment",
            "GRADE_methodology"
        ]
        
        self.evaluation_tools = [
            {
                "name": "analyze_conflict",
                "description": """Performs deep analysis of screening conflicts using methodological expertise.
                Identifies the type of conflict, underlying issues, and evidence quality.""",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "conflict_type": {
                            "type": "string",
                            "enum": ["criteria_interpretation", "methodological_assessment", 
                                    "data_extraction", "quality_rating", "statistical_interpretation"]
                        },
                        "primary_issue": {
                            "type": "string",
                            "description": "The main source of disagreement"
                        },
                        "evidence_strength": {
                            "type": "object",
                            "properties": {
                                "reviewer1_evidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "reviewer2_evidence": {"type": "number", "minimum": 0, "maximum": 1}
                            }
                        },
                        "methodological_concerns": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "resolution_confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        }
                    }
                }
            },
            {
                "name": "make_final_decision",
                "description": "Makes the final screening decision with detailed justification",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "decision": {
                            "type": "string",
                            "enum": ["include", "exclude", "need_full_text", "expert_consultation"]
                        },
                        "justification": {
                            "type": "string",
                            "description": "Detailed methodological justification"
                        },
                        "evidence_quotes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Direct quotes from abstract supporting decision"
                        },
                        "criteria_assessment": {
                            "type": "object",
                            "description": "Final assessment of each criterion"
                        },
                        "recommendations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Recommendations for similar future conflicts"
                        }
                    }
                }
            }
        ]
    
    async def evaluate_conflict(self, 
                                conflict_data: Dict,
                                strategy: ResolutionStrategy = ResolutionStrategy.EVIDENCE_BASED) -> Dict:
        """
        Comprehensive conflict evaluation with multiple strategies
        """
        
        # Use maximum thinking budget for complex evaluations
        thinking_budget = 20000 if strategy == ResolutionStrategy.EVIDENCE_BASED else 15000
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            thinking={
                "type": "enabled",
                "budget_tokens": thinking_budget
            },
            tools=self.evaluation_tools,
            messages=[{
                "role": "user",
                "content": f"""
                You are a senior systematic review methodologist with expertise in:
                {', '.join(self.expertise_areas)}
                
                Evaluate this screening conflict using {strategy.value} approach:
                
                PAPER DETAILS:
                {json.dumps(conflict_data['paper'], indent=2)}
                
                REVIEWER 1 ASSESSMENT:
                {json.dumps(conflict_data['reviewer1'], indent=2)}
                
                REVIEWER 2 ASSESSMENT:
                {json.dumps(conflict_data['reviewer2'], indent=2)}
                
                INCLUSION CRITERIA:
                {json.dumps(conflict_data['criteria'], indent=2)}
                
                RESOLUTION STRATEGY: {strategy.value}
                
                Apply the following principles:
                1. Evidence must clearly support all inclusion criteria
                2. Methodological quality indicators are critical
                3. When uncertain, consider the impact on review validity
                4. Document reasoning for training future decisions
                """
            }]
        )
        
        return self._parse_evaluation(response)

# ================== Multi-Agent Evaluator System ==================
class MultiAgentEvaluator:
    """
    Orchestrates multiple evaluator agents with different perspectives
    """
    
    def __init__(self):
        self.agents = {
            'senior_methodologist': SeniorEvaluatorAgent(),
            'clinical_expert': ClinicalExpertAgent(),
            'statistical_reviewer': StatisticalReviewerAgent(),
            'protocol_adherence': ProtocolAdherenceAgent()
        }
        self.voting_weights = {
            'senior_methodologist': 0.35,
            'clinical_expert': 0.25,
            'statistical_reviewer': 0.25,
            'protocol_adherence': 0.15
        }
    
    async def resolve_with_panel(self, conflict_data: Dict) -> Dict:
        """
        Have multiple specialized agents evaluate the conflict
        """
        
        evaluations = {}
        
        # Get evaluation from each agent
        for agent_name, agent in self.agents.items():
            evaluation = await agent.evaluate_conflict(
                conflict_data,
                self._get_agent_strategy(agent_name)
            )
            evaluations[agent_name] = evaluation
        
        # Synthesize final decision
        final_decision = await self._synthesize_decisions(evaluations, conflict_data)
        
        return final_decision
    
    def _get_agent_strategy(self, agent_name: str) -> ResolutionStrategy:
        """Assign resolution strategy based on agent role"""
        strategies = {
            'senior_methodologist': ResolutionStrategy.EVIDENCE_BASED,
            'clinical_expert': ResolutionStrategy.CONSERVATIVE,
            'statistical_reviewer': ResolutionStrategy.STRICT,
            'protocol_adherence': ResolutionStrategy.EXPERT_RULES
        }
        return strategies.get(agent_name, ResolutionStrategy.EVIDENCE_BASED)
    
    async def _synthesize_decisions(self, evaluations: Dict, conflict_data: Dict) -> Dict:
        """
        Synthesize multiple agent evaluations into final decision
        """
        
        # Calculate weighted vote
        decision_scores = {
            'include': 0,
            'exclude': 0,
            'need_full_text': 0
        }
        
        for agent_name, evaluation in evaluations.items():
            weight = self.voting_weights[agent_name]
            decision = evaluation['decision']
            confidence = evaluation.get('confidence', 0.5)
            
            decision_scores[decision] += weight * confidence
        
        # Determine final decision
        final_decision = max(decision_scores, key=decision_scores.get)
        consensus_strength = decision_scores[final_decision] / sum(self.voting_weights.values())
        
        return {
            'final_decision': final_decision,
            'consensus_strength': consensus_strength,
            'individual_evaluations': evaluations,
            'decision_scores': decision_scores,
            'requires_human_review': consensus_strength < 0.6
        }

# ================== Specialized Evaluator Agents ==================

class ClinicalExpertAgent(ReviewComponent):
    """Evaluates from clinical relevance perspective"""
    
    def __init__(self):
        super().__init__(model="claude-opus-4-20250514")
        
    async def evaluate_conflict(self, conflict_data: Dict, strategy: ResolutionStrategy) -> Dict:
        response = self.client.messages.create(
            model=self.model,
            thinking={"type": "enabled", "budget_tokens": 12000},
            messages=[{
                "role": "user",
                "content": f"""
                As a clinical expert, evaluate this screening conflict focusing on:
                1. Clinical relevance of the intervention
                2. Patient population appropriateness  
                3. Clinical meaningfulness of outcomes
                4. Real-world applicability
                
                Conflict data: {json.dumps(conflict_data, indent=2)}
                
                Prioritize clinical importance over methodological perfection.
                """
            }]
        )
        
        # Parse and return evaluation
        return self._parse_clinical_evaluation(response)

class StatisticalReviewerAgent(ReviewComponent):
    """Evaluates statistical and methodological rigor"""
    
    def __init__(self):
        super().__init__(model="claude-sonnet-4-20250514")
        
    async def evaluate_conflict(self, conflict_data: Dict, strategy: ResolutionStrategy) -> Dict:
        response = self.client.messages.create(
            model=self.model,
            thinking={"type": "enabled", "budget_tokens": 10000},
            messages=[{
                "role": "user",
                "content": f"""
                As a statistical methods expert, evaluate focusing on:
                1. Sample size adequacy
                2. Statistical analysis appropriateness
                3. Risk of bias indicators
                4. Power calculations
                5. Multiple comparisons handling
                
                Conflict data: {json.dumps(conflict_data, indent=2)}
                
                Be strict about methodological quality.
                """
            }]
        )
        
        return self._parse_statistical_evaluation(response)

class ProtocolAdherenceAgent(ReviewComponent):
    """Ensures strict adherence to pre-specified protocol"""
    
    def __init__(self):
        super().__init__(model="claude-opus-4-20250514")
        
    async def evaluate_conflict(self, conflict_data: Dict, strategy: ResolutionStrategy) -> Dict:
        response = self.client.messages.create(
            model=self.model,
            thinking={"type": "enabled", "budget_tokens": 8000},
            messages=[{
                "role": "user",
                "content": f"""
                Evaluate strict adherence to the registered protocol:
                - Do NOT make exceptions
                - Apply criteria exactly as written
                - Flag any protocol deviations
                
                Conflict data: {json.dumps(conflict_data, indent=2)}
                
                Decision must be binary based on protocol compliance.
                """
            }]
        )
        
        return self._parse_protocol_evaluation(response)

# ================== Learning Evaluator ==================

class LearningEvaluator(ReviewComponent):
    """
    Learns from previous conflict resolutions to improve future decisions
    """
    
    def __init__(self):
        super().__init__()
        self.resolution_history = []
        self.pattern_database = {}
        
    async def evaluate_with_learning(self, conflict_data: Dict) -> Dict:
        """
        Evaluate using historical patterns and learning
        """
        
        # Find similar past conflicts
        similar_conflicts = self._find_similar_conflicts(conflict_data)
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            thinking={"type": "enabled", "budget_tokens": 15000},
            messages=[{
                "role": "user",
                "content": f"""
                Evaluate this conflict considering similar past resolutions:
                
                CURRENT CONFLICT:
                {json.dumps(conflict_data, indent=2)}
                
                SIMILAR PAST CONFLICTS AND RESOLUTIONS:
                {json.dumps(similar_conflicts, indent=2)}
                
                Learn from patterns:
                1. What worked in similar cases?
                2. What mistakes were made?
                3. What would be consistent with past decisions?
                4. Should we deviate from past patterns? Why?
                
                Provide decision with learning insights.
                """
            }]
        )
        
        # Store resolution for future learning
        resolution = self._parse_resolution(response)
        self._update_pattern_database(conflict_data, resolution)
        
        return resolution
    
    def _find_similar_conflicts(self, conflict_data: Dict, n=5) -> List[Dict]:
        """
        Find n most similar past conflicts using embeddings or rules
        """
        # Simplified similarity based on conflict characteristics
        similarities = []
        
        for past_conflict in self.resolution_history:
            similarity = self._calculate_similarity(conflict_data, past_conflict)
            similarities.append((similarity, past_conflict))
        
        # Return top n similar conflicts
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [conf for _, conf in similarities[:n]]
    
    def _calculate_similarity(self, conflict1: Dict, conflict2: Dict) -> float:
        """
        Calculate similarity between two conflicts
        """
        similarity_score = 0.0
        
        # Check criteria overlap
        criteria1 = set(conflict1.get('criteria_flags', {}).keys())
        criteria2 = set(conflict2.get('criteria_flags', {}).keys())
        criteria_overlap = len(criteria1.intersection(criteria2)) / max(len(criteria1), len(criteria2), 1)
        similarity_score += criteria_overlap * 0.3
        
        # Check decision pattern
        if conflict1.get('reviewer1_decision') == conflict2.get('reviewer1_decision'):
            similarity_score += 0.2
        if conflict1.get('reviewer2_decision') == conflict2.get('reviewer2_decision'):
            similarity_score += 0.2
        
        # Check study type
        if conflict1.get('study_type') == conflict2.get('study_type'):
            similarity_score += 0.3
        
        return similarity_score
    
    def _update_pattern_database(self, conflict_data: Dict, resolution: Dict):
        """
        Update pattern database with new resolution
        """
        pattern_key = self._generate_pattern_key(conflict_data)
        
        if pattern_key not in self.pattern_database:
            self.pattern_database[pattern_key] = {
                'occurrences': 0,
                'resolutions': [],
                'success_rate': 0.0
            }
        
        self.pattern_database[pattern_key]['occurrences'] += 1
        self.pattern_database[pattern_key]['resolutions'].append(resolution)
        
        # Add to history
        self.resolution_history.append({
            'conflict': conflict_data,
            'resolution': resolution,
            'timestamp': datetime.now().isoformat()
        })
    
    def _generate_pattern_key(self, conflict_data: Dict) -> str:
        """
        Generate a pattern key for categorizing conflicts
        """
        components = [
            conflict_data.get('conflict_type', 'unknown'),
            conflict_data.get('study_type', 'unknown'),
            str(conflict_data.get('reviewer1_decision', '')),
            str(conflict_data.get('reviewer2_decision', ''))
        ]
        return '_'.join(components)

# ================== Escalation System ==================

class ConflictEscalationSystem:
    """
    Manages escalation path for difficult conflicts
    """
    
    def __init__(self):
        self.escalation_levels = {
            1: {'agent': 'auto_resolver', 'threshold': 0.8},
            2: {'agent': 'senior_evaluator', 'threshold': 0.7},
            3: {'agent': 'multi_agent_panel', 'threshold': 0.6},
            4: {'agent': 'human_expert', 'threshold': 0.0}
        }
        
        self.evaluators = {
            'auto_resolver': SimpleConflictResolver(),
            'senior_evaluator': SeniorEvaluatorAgent(),
            'multi_agent_panel': MultiAgentEvaluator(),
            'learning_evaluator': LearningEvaluator()
        }
    
    async def resolve_conflict(self, conflict_data: Dict) -> Dict:
        """
        Resolve conflict through escalation if needed
        """
        
        for level, config in self.escalation_levels.items():
            agent_name = config['agent']
            threshold = config['threshold']
            
            if agent_name == 'human_expert':
                # Flag for human review
                return {
                    'decision': 'requires_human_review',
                    'escalation_level': level,
                    'reason': 'Automated resolution confidence below threshold',
                    'conflict_data': conflict_data
                }
            
            # Get resolution from current level
            agent = self.evaluators[agent_name]
            resolution = await agent.evaluate_conflict(conflict_data)
            
            # Check if confidence meets threshold
            if resolution.get('confidence', 0) >= threshold:
                resolution['escalation_level'] = level
                resolution['resolver'] = agent_name
                return resolution
            
            # Otherwise, escalate to next level
            conflict_data['previous_attempts'] = conflict_data.get('previous_attempts', [])
            conflict_data['previous_attempts'].append({
                'level': level,
                'agent': agent_name,
                'resolution': resolution
            })
        
        return {
            'decision': 'failed_to_resolve',
            'conflict_data': conflict_data
        }

# ================== Usage Example ==================

async def demonstration():
    """
    Demonstrate the complete evaluator system
    """
    
    # Sample conflict data
    conflict_data = {
        'paper': {
            'id': 'PMC789012',
            'title': 'SGLT2 Inhibitors in Type 2 Diabetes: A Systematic Analysis',
            'abstract': 'Methods: We conducted a systematic review... Results: 15 studies were analyzed...'
        },
        'reviewer1': {
            'decision': 'include',
            'confidence': 0.7,
            'reasoning': 'Meets population and intervention criteria'
        },
        'reviewer2': {
            'decision': 'exclude', 
            'confidence': 0.8,
            'reasoning': 'Not a primary study, appears to be a review'
        },
        'criteria': {
            'population': 'T2DM patients',
            'intervention': 'SGLT2 inhibitors',
            'study_type': 'RCT'
        }
    }
    
    # Initialize escalation system
    escalation_system = ConflictEscalationSystem()
    
    # Resolve conflict
    resolution = await escalation_system.resolve_conflict(conflict_data)
    
    print(f"Resolution: {json.dumps(resolution, indent=2)}")
    
    # For complex conflicts, use multi-agent panel
    if resolution.get('escalation_level', 0) >= 3:
        multi_evaluator = MultiAgentEvaluator()
        panel_decision = await multi_evaluator.resolve_with_panel(conflict_data)
        print(f"Panel Decision: {json.dumps(panel_decision, indent=2)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstration())
