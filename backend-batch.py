# backend_production.py
# Production-ready semi-automated screening system with human-in-the-loop conflict resolution.
# Uses Redis for persistent state management.

import anthropic
from anthropic import Anthropic
import asyncio
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import os
import logging
import re

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Dependencies Check ---
try:
    import redis
except ImportError:
    logger.error("The 'redis' library is not installed. Please install it with: pip install 'redis>=5.0'")
    # You might want to exit or handle this more gracefully
    exit(1)


# --- Redis State Manager ---
class RedisManager:
    """Handles persistent state for all screening jobs using Redis."""
    def __init__(self, host=os.getenv('REDIS_HOST', 'localhost'), port=int(os.getenv('REDIS_PORT', 6379))):
        try:
            # decode_responses=True makes Redis return strings instead of bytes
            self.client = redis.Redis(host=host, port=port, decode_responses=True)
            self.client.ping()
            logger.info(f"Successfully connected to Redis at {host}:{port}.")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Could not connect to Redis at {host}:{port}. Please ensure Redis is running. Error: {e}")
            self.client = None

    def create_job(self, session_id: str, total_papers: int, criteria: Dict) -> None:
        if not self.client: return
        job_key = f"job:{session_id}"
        job_data = {
            "status": "processing",
            "total_papers": total_papers,
            "processed_count": 0,
            "agreements": 0,
            "conflicts": 0,
            "created_at": datetime.now().isoformat(),
            "criteria": json.dumps(criteria) # Store criteria for later use
        }
        self.client.hset(job_key, mapping=job_data)
        # Use separate keys for lists to avoid large hash fields
        self.client.delete(f"results:{session_id}")
        self.client.delete(f"conflicts:{session_id}")


    def get_job(self, session_id: str) -> Optional[Dict]:
        if not self.client: return None
        job_key = f"job:{session_id}"
        job_data = self.client.hgetall(job_key)
        if not job_data:
            return None
        # Fetch list data
        job_data['agreements_count'] = self.client.llen(f"results:{session_id}")
        job_data['conflicts_count'] = self.client.llen(f"conflicts:{session_id}")
        return job_data

    def add_agreement(self, session_id: str, result: Dict):
        if not self.client: return
        self.client.rpush(f"results:{session_id}", json.dumps(result))
        self.client.hincrby(f"job:{session_id}", "processed_count")
        self.client.hincrby(f"job:{session_id}", "agreements")

    def add_conflict(self, session_id: str, conflict: Dict):
        if not self.client: return
        self.client.rpush(f"conflicts:{session_id}", json.dumps(conflict))
        self.client.hincrby(f"job:{session_id}", "processed_count")
        self.client.hincrby(f"job:{session_id}", "conflicts")

    def get_conflicts(self, session_id: str) -> List[Dict]:
        if not self.client: return []
        conflict_list = self.client.lrange(f"conflicts:{session_id}", 0, -1)
        return [json.loads(item) for item in conflict_list]

    def resolve_conflict(self, session_id: str, paper_id: str, human_decision: Dict):
        if not self.client: return False
        conflict_key = f"conflicts:{session_id}"
        conflicts = self.client.lrange(conflict_key, 0, -1)
        for i, item_str in enumerate(conflicts):
            item = json.loads(item_str)
            if item.get('paper_id') == paper_id:
                # Add human decision to the conflict object
                item['human_resolution'] = human_decision
                # Move from conflict queue to results queue
                self.client.lset(conflict_key, i, "__DELETED__") # Mark for deletion
                self.client.rpush(f"results:{session_id}", json.dumps(item))
        self.client.lrem(conflict_key, 1, "__DELETED__")
        return True

    def get_final_results(self, session_id: str) -> List[Dict]:
        if not self.client: return []
        results_list = self.client.lrange(f"results:{session_id}", 0, -1)
        return [json.loads(item) for item in results_list]

    def complete_job(self, session_id: str):
        if not self.client: return
        status = "pending_conflict_resolution" if self.client.llen(f"conflicts:{session_id}") > 0 else "completed"
        self.client.hset(f"job:{session_id}", "status", status)
        self.client.hset(f"job:{session_id}", "completed_at", datetime.now().isoformat())


# --- Pydantic Data Models ---
class Reference(BaseModel):
    id: str
    title: str
    abstract: str
    authors: Optional[str] = ""
    year: Optional[str] = ""
    journal: Optional[str] = ""

class InclusionCriteria(BaseModel):
    population: str
    intervention: str
    comparator: Optional[str] = ""
    outcome: str
    studyDesigns: List[str] = []

class ScreeningJobRequest(BaseModel):
    references: List[Reference]
    criteria: InclusionCriteria

class HumanResolutionRequest(BaseModel):
    paper_id: str
    decision: str # 'include', 'exclude', 'maybe'
    notes: Optional[str] = ""


# --- AI Screening Logic ---
class AIScreeningService:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Anthropic API key is required.")
        self.client = Anthropic(api_key=api_key)

    def _create_reviewer_prompt(self, ref: Reference, criteria: InclusionCriteria, reviewer_type: str) -> str:
        instructions = {
            "methodologist": "You are a strict methodologist. Focus on study design, risk of bias, and methodological rigor.",
            "clinician": "You are a clinical expert. Focus on clinical relevance and practical applicability."
        }
        return f"""
        {instructions[reviewer_type]}
        Screen this abstract against the inclusion criteria.
        PAPER:
        ID: {ref.id}
        Title: {ref.title}
        Abstract: {ref.abstract}
        CRITERIA:
        {json.dumps(criteria.dict())}
        Respond ONLY with a single JSON object:
        {{
            "paper_id": "{ref.id}",
            "decision": "include/exclude/maybe",
            "confidence": 0.0-1.0,
            "reasons": ["A list of short, specific reasons for your decision."]
        }}
        """

    async def screen_paper_dual_ai(self, reference: Reference, criteria: InclusionCriteria) -> Dict:
        """Performs dual screening and returns the combined result."""
        prompt1 = self._create_reviewer_prompt(reference, criteria, "methodologist")
        prompt2 = self._create_reviewer_prompt(reference, criteria, "clinician")

        # Use different models for more diverse perspectives
        model1 = "claude-3-opus-20240229"
        model2 = "claude-3-sonnet-20240229"

        review_tasks = [
            self.client.messages.create(model=model1, max_tokens=1024, messages=[{"role": "user", "content": prompt1}]),
            self.client.messages.create(model=model2, max_tokens=1024, messages=[{"role": "user", "content": prompt2}])
        ]
        responses = await asyncio.gather(*review_tasks, return_exceptions=True)

        def _safe_json_parse(text):
            try:
                # Use regex to find the JSON blob, making it robust to conversational fluff
                match = re.search(r'\{.*\}', text, re.DOTALL)
                return json.loads(match.group()) if match else {"error": "JSON object not found in response."}
            except (json.JSONDecodeError, AttributeError):
                return {"error": "Failed to parse JSON response.", "raw_response": text}

        review1 = _safe_json_parse(responses[0].content[0].text) if not isinstance(responses[0], Exception) else {"error": str(responses[0])}
        review2 = _safe_json_parse(responses[1].content[0].text) if not isinstance(responses[1], Exception) else {"error": str(responses[1])}
        
        # Add reviewer metadata
        review1['reviewer_type'] = 'methodologist'
        review2['reviewer_type'] = 'clinician'

        # Check for errors first
        if "error" in review1 or "error" in review2:
            return {"paper_id": reference.id, "status": "error", "reviewer1": review1, "reviewer2": review2}

        # Check for agreement
        if review1.get('decision') == review2.get('decision'):
            return {"paper_id": reference.id, "status": "agreement", "final_decision": review1['decision'], "reviewer1": review1, "reviewer2": review2}
        else:
            return {"paper_id": reference.id, "status": "conflict", "reviewer1": review1, "reviewer2": review2}


# --- FastAPI Application ---
app = FastAPI(title="AI-Accelerated Systematic Review Backend")

# --- Globals and Startup Event ---
redis_manager: Optional[RedisManager] = None
screening_service: Optional[AIScreeningService] = None

@app.on_event("startup")
async def startup_event():
    global redis_manager, screening_service
    redis_manager = RedisManager()
    if not redis_manager.client:
        logger.error("FATAL: Redis is not available. The application cannot function without it.")
        # In a real production environment, you might want the app to fail to start
        return

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        screening_service = AIScreeningService(api_key=api_key)
        logger.info("AI Screening Service initialized successfully.")
    else:
        logger.warning("ANTHROPIC_API_KEY environment variable not set. AI screening will fail.")

# --- Helper for Background Task ---
async def process_screening_job(session_id: str, references: List[Dict], criteria_dict: Dict):
    if not screening_service:
        logger.error(f"Job {session_id} cannot start: Screening service not initialized.")
        return

    references_models = [Reference(**r) for r in references]
    criteria_model = InclusionCriteria(**criteria_dict)
    
    # Process references in concurrent batches to speed things up
    batch_size = 10 # Number of papers to process concurrently
    for i in range(0, len(references_models), batch_size):
        batch = references_models[i:i+batch_size]
        tasks = [screening_service.screen_paper_dual_ai(ref, criteria_model) for ref in batch]
        results = await asyncio.gather(*tasks)

        for result in results:
            if result['status'] == 'agreement':
                redis_manager.add_agreement(session_id, result)
            elif result['status'] == 'conflict':
                # Add original reference data to the conflict object for the UI
                ref_data = next((r.dict() for r in batch if r.id == result['paper_id']), None)
                result['reference'] = ref_data
                redis_manager.add_conflict(session_id, result)
            else: # Error case
                logger.error(f"Error processing paper {result['paper_id']} in job {session_id}: {result}")
                # Optionally, add errors to a separate list in Redis
    
    redis_manager.complete_job(session_id)
    logger.info(f"Automated screening phase for job {session_id} is complete.")


# --- API Endpoints ---
@app.post("/api/jobs", status_code=202)
async def create_screening_job(request: ScreeningJobRequest, background_tasks: BackgroundTasks):
    """Creates and starts a new screening job."""
    if not redis_manager or not screening_service:
        raise HTTPException(status_code=503, detail="System not initialized. Check Redis/API Key.")
    
    session_id = str(uuid.uuid4())
    redis_manager.create_job(session_id, len(request.references), request.criteria.dict())
    
    # Pass serializable data to the background task
    background_tasks.add_task(
        process_screening_job,
        session_id,
        [r.dict() for r in request.references],
        request.criteria.dict()
    )

    return {"session_id": session_id, "status": "processing", "message": "Job accepted and is running in the background."}

@app.get("/api/jobs/{session_id}")
async def get_job_status(session_id: str):
    """Retrieves the status and progress of a screening job."""
    if not redis_manager:
        raise HTTPException(status_code=503, detail="Redis service unavailable.")
    
    job = redis_manager.get_job(session_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    
    return job

@app.get("/api/jobs/{session_id}/conflicts")
async def get_job_conflicts(session_id: str):
    """Retrieves the queue of papers needing human conflict resolution."""
    if not redis_manager:
        raise HTTPException(status_code=503, detail="Redis service unavailable.")
    
    conflicts = redis_manager.get_conflicts(session_id)
    return {"conflicts": conflicts}

@app.post("/api/jobs/{session_id}/resolve")
async def resolve_human_conflict(session_id: str, resolution: HumanResolutionRequest):
    """Submits a human's final decision for a conflicted paper."""
    if not redis_manager:
        raise HTTPException(status_code=503, detail="Redis service unavailable.")

    success = redis_manager.resolve_conflict(session_id, resolution.paper_id, resolution.dict())
    if not success:
        raise HTTPException(status_code=404, detail="Conflict with the specified paper_id not found in this job.")
    
    # Check if this was the last conflict
    remaining_conflicts = redis_manager.client.llen(f"conflicts:{session_id}")
    if remaining_conflicts == 0:
        job_status = redis_manager.client.hget(f"job:{session_id}", "status")
        if job_status == "pending_conflict_resolution":
             redis_manager.client.hset(f"job:{session_id}", "status", "completed")

    return {"message": "Resolution recorded successfully."}


@app.get("/api/jobs/{session_id}/results")
async def get_final_job_results(session_id: str):
    """Retrieves the final, collated results for a completed job."""
    if not redis_manager:
        raise HTTPException(status_code=503, detail="Redis service unavailable.")
    
    job_status = redis_manager.client.hget(f"job:{session_id}", "status")
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found.")
        
    if job_status not in ["completed", "pending_conflict_resolution"]:
         raise HTTPException(status_code=400, detail=f"Job is still in progress with status: {job_status}")

    results = redis_manager.get_final_results(session_id)
    return {"results": results}


# --- Root Endpoint for Health Check ---
@app.get("/")
async def root():
    return {"message": "AI-Accelerated Systematic Review Backend is running."}

# --- Main entry point for running the server ---
if __name__ == "__main__":
    import uvicorn
    # Example: uvicorn backend_production:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
