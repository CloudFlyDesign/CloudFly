import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";

const TASKS = [
  ["urgent", "Urgent vs not urgent messages (40 examples)"],
  ["news", "Hockey vs space posts (200 examples)"],
];

export default function ModelPanel({ api }) {
  const [task, setTask] = useState("urgent");
  const [jobId, setJobId] = useState(null);
  const [st, setSt] = useState(null);
  const [err, setErr] = useState("");
  const accRef = useRef(null);
  const lossRef = useRef(null);
  const cellRef = useRef(null);

  async function start() {
    setErr("");
    setSt(null);
    try {
      const r = await fetch(`${api}/pipeline`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task }),
      });
      if (!r.ok) throw new Error((await r.json()).detail || "could not start");
      setJobId((await r.json()).job_id);
    } catch (e) {
      setErr(String(e.message || e));
    }
  }

  useEffect(() => {
    if (!jobId) return;
    let stop = false;
    let timer;
    async function poll() {
      try {
        const j = await (await fetch(`${api}/pipeline/${jobId}`)).json();
        if (stop) return;
        setSt(j);
        if (j.status !== "running") return;
      } catch (e) {
        /* retry on next tick */
      }
      if (!stop) timer = setTimeout(poll, 2000);
    }
    poll();
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, [jobId, api]);

  // Load any previously saved result for the selected task
  useEffect(() => {
    fetch(`${api}/pipeline/saved/${task}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => {
        if (j) {
          setSt({
            status: "done",
            stage: "done",
            rounds: j.rounds,
            log: [],
            result: j.result,
            error: null,
          });
        }
      })
      .catch(() => {});
  }, [task, api]);

  const running = jobId && (!st || st.status === "running");
  const tr = st && st.result && st.result.train;

  useEffect(() => {
    if (!tr) return;
    Plotly.react(
      accRef.current,
      [
        {
          type: "bar",
          x: [
            "Guessing",
            "Input only (no fly)",
            "Untrained fly",
            "Trained fly",
          ],
          y: [
            tr.chance,
            tr.input_only_acc,
            tr.frozen_fly_acc,
            tr.trained_fly_acc,
          ].map((v) => v * 100),
          marker: { color: ["#888", "#1f77b4", "#6b7280", "#e8730c"] },
        },
      ],
      {
        title: `Test accuracy (${tr.n_test} held-out examples)`,
        yaxis: { title: "%", range: [0, 100] },
        height: 340,
      },
      { responsive: true },
    );
    Plotly.react(
      lossRef.current,
      [
        { y: tr.loss_curve, name: "train loss", mode: "lines" },
        { y: tr.val_curve, name: "val accuracy", mode: "lines", yaxis: "y2" },
      ],
      {
        title: "Training curves",
        xaxis: { title: "epoch" },
        yaxis: { title: "loss" },
        yaxis2: {
          title: "val acc",
          overlaying: "y",
          side: "right",
          range: [0, 1],
        },
        height: 320,
      },
      { responsive: true },
    );
    const cells = [...tr.cell_type_changes].reverse();
    Plotly.react(
      cellRef.current,
      [
        {
          type: "bar",
          orientation: "h",
          x: cells.map((c) => (c.mean_multiplier - 1) * 100),
          y: cells.map((c) => c.post_cell_type),
          marker: {
            color: cells.map((c) =>
              c.mean_multiplier >= 1 ? "#e8730c" : "#1f77b4",
            ),
          },
        },
      ],
      {
        title: "Cell types whose incoming connections changed most (%)",
        xaxis: { title: "mean change in strength (%)" },
        margin: { l: 140 },
        height: 360,
      },
      { responsive: true },
    );
  }, [tr]);

  const stageLabel = {
    coach: "Step 1 of 2: Gemini is tuning the input setup",
    train: "Step 2 of 2: training the circuit's connections",
    done: "Done",
  };
  const gain = tr ? (tr.trained_fly_acc - tr.frozen_fly_acc) * 100 : 0;
  const oneEx = tr ? 100 / tr.n_test : 0;

  return (
    <div
      style={{
        border: "1px solid #444",
        borderRadius: 8,
        padding: 16,
        margin: "16px 0",
      }}
    >
      <h2 style={{ marginTop: 0 }}>Train a model</h2>
      <p style={{ fontSize: 13, opacity: 0.75 }}>
        Pick a task. Gemini finds the best way to feed it into the fly circuit,
        then the real FlyWire connection strengths are trained for that task.
        Wiring and excitatory/inhibitory signs stay fixed.
      </p>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <select
          value={task}
          onChange={(e) => setTask(e.target.value)}
          disabled={running}
        >
          {TASKS.map(([k, label]) => (
            <option key={k} value={k}>
              {label}
            </option>
          ))}
        </select>
        <button onClick={start} disabled={running}>
          {running ? "Working..." : "Train a model"}
        </button>
      </div>
      {err && <p style={{ color: "crimson" }}>{err}</p>}
      {st && st.status === "error" && (
        <p style={{ color: "crimson" }}>{st.error}</p>
      )}

      {st && (
        <p style={{ marginTop: 12 }}>
          <b>{stageLabel[st.stage] || st.stage}</b>
        </p>
      )}

      {st && st.rounds.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table
            style={{ fontSize: 13, borderCollapse: "collapse", width: "100%" }}
          >
            <thead>
              <tr style={{ textAlign: "left" }}>
                <th>#</th>
                <th>input setup</th>
                <th>fly</th>
                <th>no fly</th>
                <th>shuffled</th>
                <th>Gemini said</th>
              </tr>
            </thead>
            <tbody>
              {st.rounds.map((r) => (
                <tr key={r.round} style={{ borderTop: "1px solid #444" }}>
                  <td>{r.round}</td>
                  <td>
                    {r.recipe.pool_query}, {r.recipe.neurons_per_word}/word,{" "}
                    {r.recipe.hops} hops
                  </td>
                  <td>{(r.fly_mean * 100).toFixed(1)}%</td>
                  <td>{(r.baseline_mean * 100).toFixed(1)}%</td>
                  <td>{(r.shuffled_mean * 100).toFixed(1)}%</td>
                  <td>{r.gemini_note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {st && st.status === "running" && st.log.length > 0 && (
        <pre style={{ fontSize: 12, maxHeight: 140, overflow: "auto" }}>
          {st.log.join("\n")}
        </pre>
      )}

      {tr && (
        <>
          <p>
            <b>Result:</b> training the connections moved test accuracy by{" "}
            {gain >= 0 ? "+" : ""}
            {gain.toFixed(1)} points over the untrained fly. One test example is{" "}
            {oneEx.toFixed(1)} points, so a gap smaller than about 5 points is
            noise.
            {tr.input_only_acc > tr.trained_fly_acc
              ? " A plain classifier on the raw input still scores higher than the fly."
              : ""}
          </p>
          <div ref={accRef} />
          <div ref={lossRef} />
          <div ref={cellRef} />
          <h3>Most changed connections</h3>
          <ul>
            {tr.top_changed_connections.slice(0, 8).map((e, i) => (
              <li key={i}>
                {e.pre} to {e.post}: x{e.multiplier}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
