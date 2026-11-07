import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const COLORS = {
  input: new THREE.Color("#ff9a3c"),
  downstream: new THREE.Color("#4da3ff"),
  output: new THREE.Color("#7dffb0"),
};

function dotTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.4, "rgba(255,255,255,0.6)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

// brighten each neuron right after it spikes, then fade
function paint(st, t) {
  if (!st.neurons) return;
  const col = st.col;
  for (let i = 0; i < st.spikes.length; i++) {
    const ts = st.spikes[i];
    let last = -1;
    for (let j = 0; j < ts.length && ts[j] <= t; j++) last = ts[j];
    const glow = last < 0 ? 0 : Math.exp(-(t - last) / 18);
    const k = 0.1 + 0.9 * glow;
    const b = st.base[i];
    col[i * 3] = b.r * k;
    col[i * 3 + 1] = b.g * k;
    col[i * 3 + 2] = b.b * k;
  }
  st.neurons.geometry.attributes.color.needsUpdate = true;
}

export default function Brain3D({ ghost, scene, t, spin }) {
  const mountRef = useRef(null);
  const stateRef = useRef({});
  const tRef = useRef(0);
  tRef.current = t;

  // one-time setup
  useEffect(() => {
    const el = mountRef.current;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(el.clientWidth, el.clientHeight);
    el.appendChild(renderer.domElement);

    const cam = new THREE.PerspectiveCamera(50, el.clientWidth / el.clientHeight, 0.1, 2000);
    cam.position.set(0, 0, 230);
    const scn = new THREE.Scene();
    const group = new THREE.Group();
    group.scale.y = -1; // FlyWire y points down; change to 1 if the brain looks upside down
    scn.add(group);

    const controls = new OrbitControls(cam, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotateSpeed = 1.2;

    const st = { group, spin: true, dot: dotTexture() };
    stateRef.current = st;

    let raf;
    const loop = () => {
      raf = requestAnimationFrame(loop);
      controls.autoRotate = st.spin;
      controls.update();
      renderer.render(scn, cam);
    };
    loop();

    const ro = new ResizeObserver(() => {
      renderer.setSize(el.clientWidth, el.clientHeight);
      cam.aspect = el.clientWidth / el.clientHeight;
      cam.updateProjectionMatrix();
    });
    ro.observe(el);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
    };
  }, []);

  useEffect(() => {
    stateRef.current.spin = spin;
  }, [spin]);

  // faint full-brain outline
  useEffect(() => {
    const st = stateRef.current;
    if (!st.group || !ghost) return;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(ghost), 3));
    const mat = new THREE.PointsMaterial({
      size: 1.6, map: st.dot, color: 0x5a6b9a, transparent: true, opacity: 0.35,
      depthWrite: false, blending: THREE.AdditiveBlending,
    });
    const pts = new THREE.Points(geo, mat);
    st.group.add(pts);
    return () => {
      st.group.remove(pts);
      geo.dispose();
      mat.dispose();
    };
  }, [ghost]);

  // active neurons for the current run
  useEffect(() => {
    const st = stateRef.current;
    if (!st.group) return;
    if (st.neurons) {
      st.group.remove(st.neurons);
      st.neurons.geometry.dispose();
      st.neurons.material.dispose();
      st.neurons = null;
    }
    if (!scene || !scene.neurons.length) return;

    const n = scene.neurons.length;
    const pos = new Float32Array(n * 3);
    st.col = new Float32Array(n * 3);
    st.base = [];
    st.spikes = [];
    scene.neurons.forEach((nr, i) => {
      pos[i * 3] = nr.x;
      pos[i * 3 + 1] = nr.y;
      pos[i * 3 + 2] = nr.z;
      st.base.push(COLORS[nr.role] || COLORS.downstream);
      st.spikes.push(nr.t);
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(st.col, 3));
    const mat = new THREE.PointsMaterial({
      size: 4.5, map: st.dot, vertexColors: true, transparent: true,
      depthWrite: false, blending: THREE.AdditiveBlending,
    });
    st.neurons = new THREE.Points(geo, mat);
    st.group.add(st.neurons);
    paint(st, tRef.current);
  }, [scene]);

  // sync glow to the playback clock
  useEffect(() => {
    paint(stateRef.current, t);
  }, [t]);

  return (
    <div
      ref={mountRef}
      style={{ width: "100%", height: 520, background: "#000", borderRadius: 8, overflow: "hidden" }}
    />
  );
}