import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useRef } from "react";
import * as THREE from "three";
import { bindKeyboard, effectiveKeys, setOverride } from "./input";

const TURN = 2.4;
const MOVE = 6.2;
const BOUNDS = 21;

const BLOCKS: { x: number; z: number; hw: number; hd: number }[] = [
  { x: -14, z: -8, hw: 3.6, hd: 2.8 },
  { x: 14, z: -8, hw: 3.6, hd: 2.8 },
  { x: 0, z: -16.5, hw: 3.1, hd: 2.5 },
];

function blocked(x: number, z: number) {
  for (const b of BLOCKS) {
    if (Math.abs(x - b.x) < b.hw + 0.45 && Math.abs(z - b.z) < b.hd + 0.45) return true;
  }
  return false;
}

export function Walker() {
  const group = useRef<THREE.Group>(null);
  const yaw = useRef(0);
  const pos = useRef(new THREE.Vector3(0, 0, 16));
  const speed = useRef(0);
  const desired = useRef(new THREE.Vector3());
  const look = useRef(new THREE.Vector3());
  const { camera } = useThree();

  useEffect(() => {
    const unbind = bindKeyboard();
    const probe = {
      getYaw: () => yaw.current,
      getSpeed: () => speed.current,
      setKeys: (codes: string[]) => setOverride(codes),
    };
    window.__controlsTest = probe;
    return () => {
      unbind();
      setOverride(null);
      if (window.__controlsTest === probe) delete window.__controlsTest;
    };
  }, []);

  useFrame((_, delta) => {
    const dt = Math.min(delta, 0.05);
    const keys = effectiveKeys();
    let steer = 0;
    if (keys.includes("KeyA") || keys.includes("ArrowLeft")) steer += 1;
    if (keys.includes("KeyD") || keys.includes("ArrowRight")) steer -= 1;
    let throttle = 0;
    if (keys.includes("KeyW") || keys.includes("ArrowUp")) throttle += 1;
    if (keys.includes("KeyS") || keys.includes("ArrowDown")) throttle -= 1;

    const moving = throttle !== 0;
    const reverse = throttle >= 0 ? 1 : -1;
    yaw.current += steer * TURN * (moving ? 1 : 0.45) * reverse * dt;

    const vx = -Math.sin(yaw.current) * throttle * MOVE * dt;
    const vz = -Math.cos(yaw.current) * throttle * MOVE * dt;
    let nx = pos.current.x + vx;
    let nz = pos.current.z + vz;
    if (blocked(nx, pos.current.z)) nx = pos.current.x;
    if (blocked(pos.current.x, nz)) nz = pos.current.z;
    pos.current.x = THREE.MathUtils.clamp(nx, -BOUNDS, BOUNDS);
    pos.current.z = THREE.MathUtils.clamp(nz, -BOUNDS, BOUNDS);
    speed.current = throttle * MOVE;

    const g = group.current;
    if (g) {
      g.position.copy(pos.current);
      g.rotation.y = yaw.current + Math.PI;
    }

    const fx = -Math.sin(yaw.current);
    const fz = -Math.cos(yaw.current);
    desired.current.set(
      pos.current.x - fx * 7.2,
      4.1,
      pos.current.z - fz * 7.2,
    );
    camera.position.lerp(desired.current, 1 - Math.exp(-4.5 * dt));
    look.current.set(pos.current.x, 1.15, pos.current.z);
    camera.lookAt(look.current);
  });

  return (
    <group ref={group} position={[0, 0, 16]} raycast={() => null}>
      <mesh position={[0, 0.85, 0]} castShadow>
        <capsuleGeometry args={[0.28, 0.7, 4, 8]} />
        <meshStandardMaterial color="#c6a36a" />
      </mesh>
      <mesh position={[0, 1.55, 0]} castShadow>
        <sphereGeometry args={[0.22, 16, 12]} />
        <meshStandardMaterial color="#f3eee4" />
      </mesh>
      <mesh position={[0, 1.15, 0.18]}>
        <boxGeometry args={[0.16, 0.1, 0.08]} />
        <meshStandardMaterial color="#d4a017" emissive="#d4a017" emissiveIntensity={0.35} />
      </mesh>
    </group>
  );
}

declare global {
  interface Window {
    __controlsTest?: {
      getYaw: () => number;
      getSpeed: () => number;
      setKeys?: (codes: string[]) => void;
    };
  }
}
