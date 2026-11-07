import flysim

CODEX = "https://codex.flywire.ai/app/cell_details?data_version=783&root_id={}&dataset=fafb"

def simulate(query="sugar", limit=25, dur_ms=200, hops=2):
    found = flysim.find_neurons(query, limit=limit)["neurons"]
    if not found:
        raise ValueError(f"No neurons matched '{query}'")
    ids = [n["root_id"] for n in found]
    nodes, idx, W = flysim.build_subcircuit(ids, hops=hops)
    counts, st, si = flysim.run_lif(nodes, idx, W, ids, dur_ms=dur_ms, record=True)
    driven = {idx[i] for i in ids if i in idx}

    neurons = [{
        "row": r,
        "root_id": str(nodes[r]),  # string: these IDs exceed JavaScript's safe integer range
        "type": flysim.LABEL.get(nodes[r], "?"),
        "stimulated": r in driven,
        "spike_count": int(counts[r]),
        "codex_url": CODEX.format(nodes[r]),
    } for r in range(len(nodes)) if counts[r] > 0]

    return {
        "query": query,
        "dur_ms": dur_ms,
        "n_total": len(nodes),
        "spikes": [{"t": float(t), "row": int(r)} for t, r in zip(st, si)],
        "neurons": neurons,
    }

if __name__ == "__main__":
    import json, sys
    out = simulate(sys.argv[1] if len(sys.argv) > 1 else "sugar")
    print(json.dumps(out)[:600], "...")
    print(len(out["spikes"]), "spikes,", len(out["neurons"]), "neurons fired")