import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import export3d
import flysim
from simulate import simulate
from supervisor import start_job, jobs
from train_task import run_training

app = FastAPI(title="CloudFly engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

RUNS = {}  # in-memory for now; MongoDB/Tiger Data replace this later
_scene_cache = {}
_brain_cache = None


class SimRequest(BaseModel):
    query: str = "sugar"
    limit: int = 25
    dur_ms: float = 200
    hops: int = 2


class TrainRequest(BaseModel):
    pool_query: str = "olfactory"
    pool_size: int = 200
    neurons_per_word: int = 3
    dur_ms: float = 150.0
    hops: int = 2
    texts: Optional[List[str]] = None    # custom task later (Gemini/voice)
    labels: Optional[List[int]] = None


class SuperviseRequest(BaseModel):
    rounds: int = 5


@app.get("/brain")
def brain():
    """Full-brain point cloud (inactive context). Sent once per page load."""
    global _brain_cache
    if _brain_cache is None:
        _brain_cache = {"ghost": export3d._ghost()}
    return _brain_cache


@app.get("/scene")
def scene(query: str = "sugar", limit: int = 25, dur_ms: float = 200):
    key = (query, limit, dur_ms)
    if key not in _scene_cache:
        try:
            d = export3d.build_data(query, limit, dur_ms, heroes=0)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        d.pop("ghost", None)  # served separately by /brain
        _scene_cache[key] = d
    return _scene_cache[key]


@app.get("/neurons")
def find(query: str, limit: int = 25):
    found = flysim.find_neurons(query, limit=limit)["neurons"]
    return {"query": query, "count": len(found),
            "root_ids": [str(n["root_id"]) for n in found]}


@app.post("/simulate")
def run(req: SimRequest):
    try:
        result = simulate(req.query, req.limit, req.dur_ms, req.hops)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    run_id = uuid.uuid4().hex[:8]
    result["id"] = run_id
    RUNS[run_id] = result
    return result


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="run not found")
    return RUNS[run_id]


@app.post("/train")
def train(req: TrainRequest):
    try:
        return run_training(req.pool_query, req.pool_size, req.neurons_per_word,
                            req.dur_ms, req.hops, req.texts, req.labels)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/supervise")
def supervise_start(req: SuperviseRequest):
    return {"job_id": start_job(min(max(req.rounds, 1), 8))}


@app.get("/supervise/{job_id}")
def supervise_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown job")
    return job