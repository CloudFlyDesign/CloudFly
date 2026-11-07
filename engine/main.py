import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import flysim
from simulate import simulate

app = FastAPI(title="CloudFly engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

RUNS = {}  # in-memory for now; MongoDB/Tiger Data replace this later


class SimRequest(BaseModel):
    query: str = "sugar"
    limit: int = 25
    dur_ms: float = 200
    hops: int = 2


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