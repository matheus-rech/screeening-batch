import anthropic
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from enum import Enum
import asyncio
from dataclasses import dataclass

# ================== Data Models ==================
class ScreeningDecision(Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    MAYBE = "maybe"
    CONFLICT = "conflict"

@dataclass
class ScreeningResult:
    paper_id: str
    title: str
    abstract: str
    decision: ScreeningDecision
    confidence: float
    reasons: List[str]
    criteria_met: Dict[str, bool]
    timestamp: datetime
    reviewer_id: str

@dataclass
class InclusionCriteria:
    population: str
    intervention: str
    comparator: str
    outcome: str
    study_design: List[str]
    additional_criteria: Dict[str, str]

# ================== Base Component ==================
class ReviewComponent:
    def __init__(self, model="claude-opus-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.audit_trail = []
    
    def process_with_thinking(self, task, budget=10000):
        response = self.client.messages.create(
            model=self.model,
            thinking={"type": "enabled", "budget_tokens": budget},
            messages=task
        )
        
        # Log for audit trail
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'task': task,
            'response': response.content,
            'thinking': [block.thinking for block in response.content 
                        if hasattr(block, 'thinking')]
        })
        
        return response

# ================== Abstract Screener ==================
class AbstractScreener(ReviewComponent):
    def __init__(self):
        super().__init__()
        self.screening_tools = [
            {
                "name": "evaluate_abstract",
                "description": """Evaluates an abstract against inclusion/exclusion criteria.
                Returns structured assessment with confidence score and detailed reasoning.""",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "decision": {
                            "type": "string",
                            "enum": ["include", "exclude", "maybe"],
                            "description": "Screening decision based on criteria"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Confidence in the decision (0-1)"
                        },
                        "criteria_assessment": {
                            "type": "object",
                            "description": "Assessment of each inclusion criterion",
                            "properties": {
                                "population_met": {"type": "boolean"},
                                "intervention_met": {"type": "boolean"},
                                "comparator_met": {"type": "boolean"},
                                "outcome_met": {"type": "boolean"},
                                "study_design_met": {"type": "boolean"}
                            }
                        },
                        "exclusion_reasons": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific reasons for exclusion if applicable"
                        },
                        "key_findings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Key relevant findings from the abstract"
                        },
                        "concerns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Any concerns or ambiguities requiring human review"
                        }
                    },
                    "required": ["decision", "confidence", "criteria_assessment"]
                }
            }
        ]
    
    async def screen_single(self, paper: Dict, criteria: InclusionCriteria) -> ScreeningResult:
        """Screen a single abstract"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            thinking={"type": "enabled", "budget_tokens": 8000},
            tools=self.screening_tools,
            tool_choice={"type": "tool", "name": "evaluate_abstract"},
            messages=[{
                "role": "user",
                "content": f"""
                Screen this abstract against our inclusion criteria:
                
                PAPER:
                Title: {paper['title']}
                Abstract: {paper['abstract']}
                Year: {paper.get('year', 'Unknown')}
                
                INCLUSION CRITERIA:
                Population: {criteria.population}
                Intervention: {criteria.intervention}
                Comparator: {criteria.comparator}
                Outcome: {criteria.outcome}
                Study Design: {', '.join(criteria.study_design)}
                
                Additional Requirements:
                {json.dumps(criteria.additional_criteria, indent=2)}
                
                Carefully evaluate whether this study meets ALL criteria.
                If uncertain about any criterion, mark as "maybe" for human review.
                """
            }]
        )
        
        # Extract tool use result
        tool_result = next(
            (block for block in response.content if block.type == 'tool_use'),
            None
        )
        
        if tool_result:
            result_data = tool_result.input
            return ScreeningResult(
                paper_id=paper['id'],
                title=paper['title'],
                abstract=paper['abstract'],
                decision=ScreeningDecision(result_data['decision']),
                confidence=result_data['confidence'],
                reasons=result_data.get('exclusion_reasons', []),
                criteria_met=result_data['criteria_assessment'],
                timestamp=datetime.now(),
                reviewer_id="claude-1"
            )
    
    async def screen_batch(self, papers: List[Dict], criteria: InclusionCriteria, 
                          batch_size: int = 10) -> List[ScreeningResult]:
        """Screen multiple papers in batches"""
        results = []
        
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [
                self.screen_single(paper, criteria) 
                for paper in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            
            # Progress update
            print(f"Screened {min(i + batch_size, len(papers))}/{len(papers)} papers")
            
            # Add small delay to respect rate limits
            await asyncio.sleep(1)
        
        return results

# ================== Dual Screening Simulator ==================
class DualScreeningSimulator(ReviewComponent):
    """Simulates two independent reviewers for validation"""
    
    def __init__(self):
        super().__init__()
        self.reviewer_profiles = {
            "reviewer_1": {
                "strictness": 0.8,  # More conservative
                "focus": "methodology",
                "model": "claude-opus-4-20250514"
            },
            "reviewer_2": {
                "strictness": 0.6,  # More inclusive
                "focus": "clinical_relevance",
                "model": "claude-sonnet-4-20250514"
            }
        }
    
    async def dual_screen(self, paper: Dict, criteria: InclusionCriteria) -> Tuple[ScreeningResult, ScreeningResult]:
        """Have two different Claude instances screen the same paper"""
        
        results = []
        
        for reviewer_id, profile in self.reviewer_profiles.items():
            response = self.client.messages.create(
                model=profile["model"],
                thinking={"type": "enabled", "budget_tokens": 6000},
                messages=[{
                    "role": "user",
                    "content": f"""
                    You are Reviewer {reviewer_id.split('_')[1]} with a {profile['focus']} focus.
                    Strictness level: {profile['strictness']}
                    
                    Screen this paper:
                    {json.dumps(paper, indent=2)}
                    
                    Against criteria:
                    {json.dumps(criteria.__dict__, indent=2)}
                    
                    Be {'conservative' if profile['strictness'] > 0.7 else 'balanced'} in your assessment.
                    """
                }]
            )
            
            # Parse and create result
            # ... (similar to screen_single)
            results.append(result)
        
        return tuple(results)

# ================== Conflict Resolver ==================
class ConflictResolver(ReviewComponent):
    def __init__(self):
        super().__init__()
        self.resolution_tools = [
            {
                "name": "resolve_screening_conflict",
                "description": "Analyzes screening disagreements and provides resolution",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "final_decision": {
                            "type": "string",
                            "enum": ["include", "exclude", "need_full_text", "need_third_reviewer"]
                        },
                        "resolution_rationale": {
                            "type": "string",
                            "description": "Detailed explanation of the resolution"
                        },
                        "key_disagreement_points": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "evidence_from_abstract": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Direct quotes supporting the resolution"
                        },
                        "methodology_concerns": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "recommended_action": {
                            "type": "string",
                            "description": "Specific next steps"
                        }
                    }
                }
            }
        ]
    
    async def resolve_conflict(self, 
                               result1: ScreeningResult, 
                               result2: ScreeningResult,
                               criteria: InclusionCriteria,
                               require_consensus: bool = True) -> ScreeningResult:
        """Resolve conflicts between two screening decisions"""
        
        # Check if there's actually a conflict
        if result1.decision == result2.decision:
            # Agreement - return with higher confidence
            return result1 if result1.confidence >= result2.confidence else result2
        
        # Use extended thinking for complex conflict resolution
        response = self.client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=3000,
            thinking={
                "type": "enabled", 
                "budget_tokens": 15000  # More thinking for conflicts
            },
            tools=self.resolution_tools,
            tool_choice={"type": "tool", "name": "resolve_screening_conflict"},
            messages=[{
                "role": "user",
                "content": f"""
                Resolve screening conflict for paper: "{result1.title}"
                
                ABSTRACT:
                {result1.abstract}
                
                REVIEWER 1 ASSESSMENT:
                - Decision: {result1.decision.value}
                - Confidence: {result1.confidence}
                - Criteria met: {json.dumps(result1.criteria_met, indent=2)}
                - Reasons: {'; '.join(result1.reasons)}
                
                REVIEWER 2 ASSESSMENT:
                - Decision: {result2.decision.value}
                - Confidence: {result2.confidence}
                - Criteria met: {json.dumps(result2.criteria_met, indent=2)}
                - Reasons: {'; '.join(result2.reasons)}
                
                INCLUSION CRITERIA:
                {json.dumps(criteria.__dict__, indent=2)}
                
                Provide a thorough analysis considering:
                1. Which reviewer's interpretation is better supported by the abstract text
                2. Whether the disagreement stems from ambiguous wording
                3. If full-text review would resolve the conflict
                4. The methodological quality indicators in the abstract
                """
            }]
        )
        
        # Extract resolution
        tool_result = next(
            (block for block in response.content if block.type == 'tool_use'),
            None
        )
        
        if tool_result:
            resolution = tool_result.input
            
            # Create resolved result
            return ScreeningResult(
                paper_id=result1.paper_id,
                title=result1.title,
                abstract=result1.abstract,
                decision=ScreeningDecision(resolution['final_decision'].replace('need_', '')),
                confidence=(result1.confidence + result2.confidence) / 2,
                reasons=resolution['key_disagreement_points'],
                criteria_met={
                    # Merge criteria assessments conservatively
                    k: result1.criteria_met.get(k, False) and result2.criteria_met.get(k, False)
                    for k in result1.criteria_met
                },
                timestamp=datetime.now(),
                reviewer_id="conflict_resolver"
            )

# ================== Complete Screening Pipeline ==================
class ScreeningPipeline:
    def __init__(self, mode="single", confidence_threshold=0.8):
        self.mode = mode  # "single", "dual", or "consensus"
        self.confidence_threshold = confidence_threshold
        self.screener = AbstractScreener()
        self.dual_screener = DualScreeningSimulator()
        self.resolver = ConflictResolver()
        self.statistics = {
            'total_screened': 0,
            'included': 0,
            'excluded': 0,
            'conflicts': 0,
            'low_confidence': 0
        }
    
    async def run_screening(self, papers: List[Dict], criteria: InclusionCriteria) -> Dict:
        """Execute complete screening workflow"""
        
        results = []
        conflicts = []
        
        if self.mode == "single":
            # Single reviewer mode
            results = await self.screener.screen_batch(papers, criteria)
            
        elif self.mode == "dual":
            # Dual independent screening
            for paper in papers:
                result1, result2 = await self.dual_screener.dual_screen(paper, criteria)
                
                if result1.decision != result2.decision:
                    # Conflict detected
                    conflicts.append((result1, result2))
                    resolved = await self.resolver.resolve_conflict(
                        result1, result2, criteria
                    )
                    results.append(resolved)
                    self.statistics['conflicts'] += 1
                else:
                    # Agreement
                    results.append(result1)
        
        # Analyze results
        for result in results:
            if result.decision == ScreeningDecision.INCLUDE:
                self.statistics['included'] += 1
            elif result.decision == ScreeningDecision.EXCLUDE:
                self.statistics['excluded'] += 1
            
            if result.confidence < self.confidence_threshold:
                self.statistics['low_confidence'] += 1
        
        self.statistics['total_screened'] = len(papers)
        
        # Generate summary report
        report = self.generate_screening_report(results, conflicts)
        
        return {
            'results': results,
            'conflicts': conflicts,
            'statistics': self.statistics,
            'report': report,
            'flagged_for_review': [r for r in results if r.confidence < self.confidence_threshold]
        }
    
    def generate_screening_report(self, results: List[ScreeningResult], conflicts: List) -> str:
        """Generate comprehensive screening report"""
        return f"""
        SYSTEMATIC REVIEW SCREENING REPORT
        ===================================
        
        Total Papers Screened: {self.statistics['total_screened']}
        Included: {self.statistics['included']} ({self.statistics['included']/self.statistics['total_screened']*100:.1f}%)
        Excluded: {self.statistics['excluded']} ({self.statistics['excluded']/self.statistics['total_screened']*100:.1f}%)
        
        Quality Metrics:
        - Conflicts Resolved: {self.statistics['conflicts']}
        - Low Confidence Decisions: {self.statistics['low_confidence']}
        - Average Confidence: {sum(r.confidence for r in results)/len(results):.2f}
        
        Papers Requiring Manual Review: {self.statistics['low_confidence']}
        
        Top Exclusion Reasons:
        {self._analyze_exclusion_reasons(results)}
        
        Timestamp: {datetime.now().isoformat()}
        """
    
    def _analyze_exclusion_reasons(self, results: List[ScreeningResult]) -> str:
        """Analyze common exclusion reasons"""
        reasons_count = {}
        for result in results:
            if result.decision == ScreeningDecision.EXCLUDE:
                for reason in result.reasons:
                    reasons_count[reason] = reasons_count.get(reason, 0) + 1
        
        sorted_reasons = sorted(reasons_count.items(), key=lambda x: x[1], reverse=True)
        return '\n'.join([f"  - {reason}: {count}" for reason, count in sorted_reasons[:5]])

# ================== Usage Example ==================
async def main():
    # Define inclusion criteria
    criteria = InclusionCriteria(
        population="Adults with type 2 diabetes",
        intervention="SGLT2 inhibitors",
        comparator="Placebo or standard care",
        outcome="Cardiovascular outcomes",
        study_design=["RCT", "Randomized Controlled Trial"],
        additional_criteria={
            "minimum_followup": "12 weeks",
            "minimum_sample_size": "100",
            "language": "English"
        }
    )
    
    # Sample papers to screen
    papers = [
        {
            "id": "PMC123456",
            "title": "Effects of Empagliflozin on Cardiovascular Outcomes in Type 2 Diabetes",
            "abstract": "Background: SGLT2 inhibitors have shown promise... Methods: We conducted a randomized controlled trial with 500 adults with type 2 diabetes... Results: After 52 weeks, cardiovascular events were reduced by 38%...",
            "year": 2024
        },
        # ... more papers
    ]
    
    # Initialize pipeline
    pipeline = ScreeningPipeline(mode="dual", confidence_threshold=0.85)
    
    # Run screening
    results = await pipeline.run_screening(papers, criteria)
    
    # Output results
    print(results['report'])
    
    # Export for further processing
    with open('screening_results.json', 'w') as f:
        json.dump(
            {
                'results': [r.__dict__ for r in results['results']],
                'statistics': results['statistics']
            },
            f,
            indent=2,
            default=str
        )

if __name__ == "__main__":
    asyncio.run(main())
