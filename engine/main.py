import uuid
import json
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import export3d
import flysim
from gemini_util import gemini_call
from simulate import simulate
from supervisor import start_job, jobs
from train_task import run_training
from train_fly import start_fly_job, jobs as fly_jobs
from pipeline import start_pipeline, jobs as pipe_jobs, SAVE_DIR


@asynccontextmanager
async def lifespan(app):
    # load the connectome and neuron positions once, before any request
    flysim.find_neurons("sugar", limit=1)
    export3d._positions()
    export3d._ghost()
    yield


app = FastAPI(title="CloudFly engine", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

RUNS = {}  # in-memory for now; MongoDB/Tiger Data replace this later
_scene_cache = {}


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
    texts: Optional[List[str]] = None
    labels: Optional[List[int]] = None


class SuperviseRequest(BaseModel):
    rounds: int = 5


class TrainFlyRequest(BaseModel):
    dataset: str = "toy"      # "toy" or "news"
    epochs: int = 15
    pool_query: str = "olfactory"


class PipelineRequest(BaseModel):
    task: str = "urgent"      # "urgent" or "news"
    rounds: int = 4
    epochs: int = 20


VALID_QUERIES = ["sugar", "gustatory", "olfactory", "clock", "mushroom body",
                 "central complex", "descending", "motor", "Kenyon"]


class CommandRequest(BaseModel):
    text: str


def _job_view(job):
    """Job status without sending the whole log on every poll."""
    out = dict(job)
    if isinstance(out.get("log"), list):
        out["log"] = out["log"][-40:]
    return out


@app.get("/brain")
def brain():
    """Full-brain point cloud (inactive context). Sent once per page load."""
    return {"ghost": export3d._ghost()}


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
    return _job_view(job)


@app.post("/train_fly")
def train_fly_start(req: TrainFlyRequest):
    if req.dataset not in ("toy", "news"):
        raise HTTPException(status_code=422, detail="dataset must be 'toy' or 'news'")
    # one training job at a time: return the running one instead of starting another
    for jid, j in fly_jobs.items():
        if j.get("status") == "running":
            return {"job_id": jid, "already_running": True}
    return {"job_id": start_fly_job(dataset=req.dataset,
                                    epochs=min(max(req.epochs, 1), 60),
                                    pool_query=req.pool_query)}


@app.get("/train_fly/{job_id}")
def train_fly_status(job_id: str):
    job = fly_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown job")
    return _job_view(job)


@app.post("/pipeline")
def pipeline_start(req: PipelineRequest):
    if req.task not in ("urgent", "news"):
        raise HTTPException(status_code=422, detail="task must be 'urgent' or 'news'")
    for jid, j in pipe_jobs.items():
        if j.get("status") == "running":
            return {"job_id": jid, "already_running": True}
    return {"job_id": start_pipeline(req.task, min(max(req.rounds, 1), 6),
                                     min(max(req.epochs, 5), 40))}


@app.get("/pipeline/{job_id}")
def pipeline_status(job_id: str):
    job = pipe_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown job")
    return _job_view(job)


@app.get("/pipeline/saved/{task}")
def pipeline_saved(task: str):
    if task not in ("urgent", "news"):
        raise HTTPException(status_code=404, detail="unknown task")
    path = os.path.join(SAVE_DIR, f"{task}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="no saved result yet")
    with open(path) as f:
        return json.load(f)


@app.post("/command")
def command(req: CommandRequest):
    prompt = (
        "You control a fruit fly brain simulator. Convert the request into JSON with keys: "
        f"query (EXACTLY one of {VALID_QUERIES}; pick the closest fit, e.g. milk or candy "
        "maps to sugar, a smell maps to olfactory, memory maps to mushroom body), "
        "dur_ms (integer 50 to 500, default 200), and reply (one friendly sentence "
        "saying what you are doing). Reply with JSON only, no code fences.\n\n"
        f"Request: {req.text}"
    )
    raw = gemini_call(prompt).strip().removeprefix("```json").removesuffix("```").strip()
    try:
        cmd = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail=f"Could not parse: {raw[:200]}")
    if cmd.get("query") not in VALID_QUERIES:
        raise HTTPException(status_code=422, detail=f"Unknown query: {cmd.get('query')}")
    cmd["dur_ms"] = int(min(max(cmd.get("dur_ms", 200), 50), 500))
    return cmd