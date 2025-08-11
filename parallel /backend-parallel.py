from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import json
from datetime import datetime
import uuid
from collections import deque

# Import our screening components from earlier
from screening_workflow import (
    AbstractScreener,
    ConflictResolver,
    MultiAgentEvaluator,
    ScreeningPipeline,
    InclusionCriteria,
    ScreeningResult,
    ScreeningDecision
)

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Data Models ====================
class Reference(BaseModel):
    id: str
    title: str
    abstract: str
    authors: str
    year: str
    journal: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    status: str = "pending"

class CriteriaInput(BaseModel):
    population: str
    intervention: str
    comparator: Optional[str] = None
    outcome: str
    study_designs: List[str]
    additional_criteria: Dict[str, str]

class ScreeningRequest(BaseModel):
    reference: Reference
    criteria: CriteriaInput
    use_ai: bool = True
    require_dual_screening: bool = False

class BatchScreeningRequest(BaseModel):
    references: List[Reference]
    criteria: CriteriaInput
    batch_size: int = 10
    mode: str = "single"  # single, dual, consensus

# ==================== Queue Management ====================
class ScreeningQueueManager:
    def __init__(self):
        self.queues = {}  # session_id -> queue
        self.results = {}  # session_id -> results
        self.active_sessions = {}
        
    def create_session(self, references: List[Reference], criteria: CriteriaInput) -> str:
        session_id = str(uuid.uuid4())
        self.queues[session_id] = deque(references)
        self.results[session_id] = []
        self.active_sessions[session_id] = {
            'criteria': criteria,
            'created_at': datetime.now().isoformat(),
            'total_references': len(references),
            'processed': 0,
            'status': 'active'
        }
        return session_id
    
    def get_next(self, session_id: str) -> Optional[Reference]:
        if session_id in self.queues and self.queues[session_id]:
            return self.queues[session_id].popleft()
        return None
    
    def add_result(self, session_id: str, result: Dict):
        if session_id in self.results:
            self.results[session_id].append(result)
            self.active_sessions[session_id]['processed'] += 1
    
    def get_progress(self, session_id: str) -> Dict:
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        return {
            'total': session['total_references'],
            'processed': session['processed'],
            'remaining': len(self.queues.get(session_id, [])),
            'percentage': (session['processed'] / session['total_references'] * 100) 
                         if session['total_references'] > 0 else 0
        }

queue_manager = ScreeningQueueManager()

# ==================== API Endpoints ====================

@app.post("/api/import")
async def import_references(references: List[Reference], criteria: CriteriaInput):
    """Import references and create a screening session"""
    session_id = queue_manager.create_session(references, criteria)
    
    return {
        "session_id": session_id,
        "total_references": len(references),
        "message": f"Successfully imported {len(references)} references"
    }

@app.post("/api/screen/single")
async def screen_single_reference(request: ScreeningRequest):
    """Screen a single reference using AI"""
    
    # Convert to our internal format
    criteria = InclusionCriteria(
        population=request.criteria.population,
        intervention=request.criteria.intervention,
        comparator=request.criteria.comparator or "",
        outcome=request.criteria.outcome,
        study_design=request.criteria.study_designs,
        additional_criteria=request.criteria.additional_criteria
    )
    
    # Initialize screener
    screener = AbstractScreener()
    
    # Perform screening
    paper_data = {
        'id': request.reference.id,
        'title': request.reference.title,
        'abstract': request.reference.abstract,
        'year': request.reference.year
    }
    
    try:
        result = await screener.screen_single(paper_data, criteria)
        
        return {
            "decision": result.decision.value,
            "confidence": result.confidence,
            "reasons": result.reasons,
            "criteria_met": result.criteria_met,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/screen/batch")
async def screen_batch(request: BatchScreeningRequest, background_tasks: BackgroundTasks):
    """Start batch screening process"""
    
    session_id = queue_manager.create_session(request.references, request.criteria)
    
    # Start background processing
    background_tasks.add_task(
        process_batch_screening,
        session_id,
        request
    )
    
    return {
        "session_id": session_id,
        "message": "Batch screening started",
        "total_references": len(request.references)
    }

async def process_batch_screening(session_id: str, request: BatchScreeningRequest):
    """Background task for batch screening"""
    
    criteria = InclusionCriteria(
        population=request.criteria.population,
        intervention=request.criteria.intervention,
        comparator=request.criteria.comparator or "",
        outcome=request.criteria.outcome,
        study_design=request.criteria.study_designs,
        additional_criteria=request.criteria.additional_criteria
    )
    
    pipeline = ScreeningPipeline(mode=request.mode)
    
    papers = [
        {
            'id': ref.id,
            'title': ref.title,
            'abstract': ref.abstract,
            'year': ref.year
        }
        for ref in request.references
    ]
    
    # Process in batches
    for i in range(0, len(papers), request.batch_size):
        batch = papers[i:i + request.batch_size]
        results = await pipeline.screener.screen_batch(batch, criteria, batch_size=1)
        
        for result in results:
            queue_manager.add_result(session_id, {
                'paper_id': result.paper_id,
                'decision': result.decision.value,
                'confidence': result.confidence,
                'reasons': result.reasons
            })
        
        # Small delay between batches
        await asyncio.sleep(1)
    
    queue_manager.active_sessions[session_id]['status'] = 'completed'

@app.get("/api/screen/next/{session_id}")
async def get_next_reference(session_id: str):
    """Get next reference from queue"""
    
    reference = queue_manager.get_next(session_id)
    if not reference:
        return {"message": "No more references", "completed": True}
    
    return {
        "reference": reference,
        "progress": queue_manager.get_progress(session_id)
    }

@app.post("/api/conflict/resolve")
async def resolve_conflict(
    result1: Dict,
    result2: Dict,
    paper: Reference,
    criteria: CriteriaInput
):
    """Resolve screening conflict using multi-agent system"""
    
    conflict_data = {
        'paper': {
            'id': paper.id,
            'title': paper.title,
            'abstract': paper.abstract
        },
        'reviewer1': result1,
        'reviewer2': result2,
        'criteria': criteria.dict()
    }
    
    # Use multi-agent evaluator
    evaluator = MultiAgentEvaluator()
    resolution = await evaluator.resolve_with_panel(conflict_data)
    
    return resolution

@app.get("/api/session/{session_id}/progress")
async def get_session_progress(session_id: str):
    """Get screening session progress"""
    
    progress = queue_manager.get_progress(session_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return progress

@app.get("/api/session/{session_id}/results")
async def get_session_results(session_id: str):
    """Get all results for a session"""
    
    if session_id not in queue_manager.results:
        raise HTTPException(status_code=404, detail="Session not found")
    
    results = queue_manager.results[session_id]
    
    # Calculate statistics
    stats = {
        'total': len(results),
        'included': sum(1 for r in results if r['decision'] == 'include'),
        'excluded': sum(1 for r in results if r['decision'] == 'exclude'),
        'maybe': sum(1 for r in results if r['decision'] == 'maybe'),
        'avg_confidence': sum(r['confidence'] for r in results) / len(results) if results else 0
    }
    
    return {
        'results': results,
        'statistics': stats
    }

@app.post("/api/export/{session_id}")
async def export_results(session_id: str, format: str = "csv"):
    """Export screening results"""
    
    if session_id not in queue_manager.results:
        raise HTTPException(status_code=404, detail="Session not found")
    
    results = queue_manager.results[session_id]
    
    if format == "csv":
        csv_content = "paper_id,decision,confidence,reasons\n"
        for r in results:
            csv_content += f"{r['paper_id']},{r['decision']},{r['confidence']},{';'.join(r.get('reasons', []))}\n"
        
        return {
            "content": csv_content,
            "filename": f"screening_results_{session_id}.csv"
        }
    
    elif format == "json":
        return {
            "content": json.dumps(results, indent=2),
            "filename": f"screening_results_{session_id}.json"
        }
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")

@app.get("/api/stats/global")
async def get_global_statistics():
    """Get global usage statistics"""
    
    total_sessions = len(queue_manager.active_sessions)
    total_processed = sum(
        session['processed'] 
        for session in queue_manager.active_sessions.values()
    )
    
    active_sessions = sum(
        1 for session in queue_manager.active_sessions.values()
        if session['status'] == 'active'
    )
    
    return {
        'total_sessions': total_sessions,
        'active_sessions': active_sessions,
        'total_references_processed': total_processed,
        'server_status': 'operational'
    }

# ==================== WebSocket for Real-time Updates ====================
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    try:
        while True:
            # Send progress updates every 2 seconds
            progress = queue_manager.get_progress(session_id)
            if progress:
                await websocket.send_json({
                    "type": "progress",
                    "data": progress
                })
            
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        print(f"Client disconnected from session {session_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
