import { useState, useRef, useEffect } from "react";
import Plotly from "plotly.js-dist-min";
import Brain3D from "./Brain3D";

const API = "http://localhost:8001";
const PRESETS = [
  ["Taste (sugar)", "sugar"],
  ["Smell", "olfactory"],
  ["Memory", "mushroom body"],
  ["Navigation", "central complex"],
  ["Body clock", "clock"],
  ["Movement commands", "descending"],
  ["Motor", "motor"],
];

export default function App() {
  const [query, setQuery] = useState("sugar");
  const [dur, setDur] = useState(200);
  const [data, setData] = useState(null);
  const [scene, setScene] = useState(null);
  const [ghost, setGhost] = useState(null);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [spin, setSpin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const rasterRef = useRef(null);
  const barRef = useRef(null);

  async function run(q = query) {
    setQuery(q);
    setLoading(true);
    setError("");
    setPlaying(false);
    try {
      const d = Number(dur);
      const [sim, sc] = await Promise.all([
        fetch(`${API}/simulate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: q, dur_ms: d }),
        }),
        fetch(`${API}/scene?query=${encodeURIComponent(q)}&dur_ms=${d}`),
      ]);
      if (!sim.ok)
        throw new Error((await sim.json()).detail || "request failed");
      setData(await sim.json());
      setScene(sc.ok ? await sc.json() : null);
      setT(0);
      setPlaying(true);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetch(`${API}/brain`)
      .then((r) => r.json())
      .then((j) => setGhost(j.ghost))
      .catch(() => {});
    run("sugar");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // playback clock, loops with a short hold at the end
  useEffect(() => {
    if (!playing || !data) return;
    const step = data.dur_ms / 100;
    let cur = 0;
    const id = setInterval(() => {
      cur += step;
      if (cur > data.dur_ms + 3 * step) cur = 0;
      setT(Math.min(cur, data.dur_ms));
    }, 40);
    return () => clearInterval(id);
  }, [playing, data]);

  // raster drawn once per run
  useEffect(() => {
    if (!data) return;
    const ordered = [...data.neurons].sort(
      (a, b) => b.stimulated - a.stimulated || b.spike_count - a.spike_count,
    );
    const yOf = Object.fromEntries(ordered.map((n, k) => [n.row, k]));
    const stim = new Set(
      data.neurons.filter((n) => n.stimulated).map((n) => n.row),
    );
    const trace = (isStim, color, name) => {
      const s = data.spikes.filter(
        (p) => stim.has(p.row) === isStim && p.row in yOf,
      );
      return {
        x: s.map((p) => p.t),
        y: s.map((p) => yOf[p.row]),
        mode: "markers",
        marker: { symbol: "line-ns-open", color, size: 7 },
        name,
      };
    };
    Plotly.react(
      rasterRef.current,
      [
        trace(true, "#e8730c", "stimulated (input)"),
        trace(false, "#1f77b4", "downstream"),
      ],
      {
        height: 420,
        title: `Stimulating '${data.query}' neurons`,
        xaxis: { title: "time (ms)", range: [0, data.dur_ms] },
        yaxis: { title: "neuron", autorange: "reversed" },
        margin: { t: 40 },
        shapes: [
          {
            type: "line",
            xref: "x",
            yref: "paper",
            x0: 0,
            x1: 0,
            y0: 0,
            y1: 1,
            line: { color: "#888", width: 2 },
          },
        ],
      },
      { responsive: true },
    );
  }, [data]);

  // only the time cursor moves during playback
  useEffect(() => {
    if (!data || !rasterRef.current || !rasterRef.current.data) return;
    Plotly.relayout(rasterRef.current, {
      "shapes[0].x0": t,
      "shapes[0].x1": t,
    });
  }, [t, data]);

  useEffect(() => {
    if (!data) return;
    const byType = {};
    data.neurons
      .filter((n) => !n.stimulated)
      .forEach((n) => (byType[n.type] = (byType[n.type] || 0) + n.spike_count));
    const top = Object.entries(byType)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);
    Plotly.react(
      barRef.current,
      [
        {
          type: "bar",
          orientation: "h",
          x: top.map((x) => x[1]).reverse(),
          y: top.map((x) => x[0]).reverse(),
        },
      ],
      {
        height: 380,
        title: "Top downstream cell types",
        xaxis: { title: "total spikes" },
        margin: { l: 140, t: 40 },
      },
      { responsive: true },
    );
  }, [data]);

  const responders = data
    ? data.neurons
        .filter((n) => !n.stimulated)
        .sort((a, b) => b.spike_count - a.spike_count)
        .slice(0, 8)
    : [];

  return (
    <div
      style={{
        maxWidth: 1000,
        margin: "0 auto",
        padding: 16,
        fontFamily: "sans-serif",
      }}
    >
      <h1>CloudFly: Poke a Brain</h1>

      <div
        style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}
      >
        {PRESETS.map(([label, q]) => (
          <button
            key={q}
            onClick={() => run(q)}
            disabled={loading}
            style={{ fontWeight: q === query ? "bold" : "normal" }}
          >
            {label}
          </button>
        ))}
      </div>
      <div
        style={{
          display: "flex",
          gap: 8,
          flexWrap: "wrap",
          marginBottom: 12,
          alignItems: "center",
        }}
      >
        <input value={query} onChange={(e) => setQuery(e.target.value)} />
        <input
          type="number"
          value={dur}
          onChange={(e) => setDur(e.target.value)}
          style={{ width: 80 }}
        />
        <span>ms</span>
        <button onClick={() => run()} disabled={loading}>
          {loading ? "Running..." : "Run"}
        </button>
        {data && (
          <button onClick={() => setPlaying((p) => !p)}>
            {playing ? "Pause" : "Play"}
          </button>
        )}
        <label>
          <input
            type="checkbox"
            checked={spin}
            onChange={(e) => setSpin(e.target.checked)}
          />{" "}
          spin
        </label>
      </div>
      {error && <p style={{ color: "crimson" }}>{error}</p>}

      <Brain3D ghost={ghost} scene={scene} t={t} spin={spin} />
      <p style={{ fontSize: 13, opacity: 0.7 }}>
        Orange = stimulated input, blue = downstream response. Drag to rotate,
        scroll to zoom.
      </p>

      {data && (
        <p>
          {data.neurons.length} neurons fired, {data.spikes.length} spikes.
        </p>
      )}
      <div ref={rasterRef} style={{ minHeight: 420 }} />
      <div ref={barRef} style={{ minHeight: 380 }} />
      {responders.length > 0 && (
        <>
          <h3>Strongest responders (links open the real neuron)</h3>
          <ul>
            {responders.map((n) => (
              <li key={n.row}>
                <a href={n.codex_url} target="_blank" rel="noreferrer">
                  {n.type}
                </a>{" "}
                ({n.spike_count} spikes)
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
