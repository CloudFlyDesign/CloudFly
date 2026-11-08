import { useFrame, useThree } from "@react-three/fiber";
import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import scene from "./brain-scene.json";
import { useTown } from "./store";

const DATA = scene as { ghost: number[] };

function dotTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  if (!g) return new THREE.Texture();
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.4, "rgba(255,255,255,0.55)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  const tex = new THREE.CanvasTexture(c);
  tex.needsUpdate = true;
  return tex;
}

export function NeuronStage() {
  const group = useRef<THREE.Group>(null);
  const points = useRef<THREE.Points>(null);
  const { gl } = useThree();
  const dot = useMemo(dotTexture, []);
  const ghostGeo = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(DATA.ghost, 3));
    return geo;
  }, []);
  const active = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    return { geo, colors: new Float32Array(0), key: "" };
  }, []);

  useLayoutEffect(() => {
    const prev = gl.toneMapping;
    gl.toneMapping = THREE.NoToneMapping;
    gl.setClearColor("#000000");
    return () => {
      gl.toneMapping = prev;
    };
  }, [gl]);

  useFrame((_, delta) => {
    const mesh = points.current;
    if (!mesh) return;
    const state = useTown.getState();
    const firing = state.firing && state.firing.flyId === state.speaker ? state.firing : null;
    const cells = firing?.cells ?? [];
    const key = firing ? `${firing.flyId}:${cells.map((cell) => cell.i).join(",")}` : "";
    if (active.key !== key) {
      active.key = key;
      const ghostCount = DATA.ghost.length / 3;
      const pos = new Float32Array(Math.max(cells.length, 1) * 3);
      active.colors = new Float32Array(Math.max(cells.length, 1) * 3);
      cells.forEach((cell, index) => {
        const slot = cell.i % ghostCount;
        pos[index * 3] = DATA.ghost[slot * 3] + (cell.i % 7) * 0.4;
        pos[index * 3 + 1] = DATA.ghost[slot * 3 + 1];
        pos[index * 3 + 2] = DATA.ghost[slot * 3 + 2] + (cell.i % 5) * 0.4;
      });
      active.geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      active.geo.setAttribute("color", new THREE.BufferAttribute(active.colors, 3));
      mesh.geometry = active.geo;
    }
    const peak = Math.max(...cells.map((cell) => cell.n), 1);
    const col = active.colors;
    const pulse = 0.55 + 0.45 * Math.sin(performance.now() / 280);
    for (let i = 0; i < cells.length; i += 1) {
      const hot = cells[i].n / peak;
      const k = (0.25 + 0.75 * hot) * pulse;
      col[i * 3] = (hot > 0.45 ? 1 : 0.25) * k;
      col[i * 3 + 1] = (hot > 0.45 ? 0.55 : 0.65) * k;
      col[i * 3 + 2] = (hot > 0.45 ? 0.2 : 1) * k;
    }
    const attr = mesh.geometry.getAttribute("color") as THREE.BufferAttribute;
    if (attr) attr.needsUpdate = true;
    if (group.current) group.current.rotation.y += delta * 0.15;
  });

  return (
    <>
      <color attach="background" args={["#000000"]} />
      <group ref={group} scale={[1, -1, 1]}>
        <points geometry={ghostGeo}>
          <pointsMaterial
            map={dot}
            color="#5a6b9a"
            size={1.7}
            transparent
            opacity={0.38}
            depthWrite={false}
            blending={THREE.AdditiveBlending}
            sizeAttenuation
          />
        </points>
        <points ref={points}>
          <bufferGeometry />
          <pointsMaterial
            map={dot}
            vertexColors
            size={16}
            sizeAttenuation={false}
          />
        </points>
      </group>
    </>
  );
}
