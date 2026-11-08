import { useFrame } from "@react-three/fiber";
import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { NeuronStage } from "./Brain";
import { useNameplate } from "./labels";
import { SEED } from "./seed";
import { useTown } from "./store";
import { SLOTS, DESK, type Fly } from "./types";

const TREES: [number, number][] = [
  [-18, 8], [-16, 16], [-18, -2], [18, 6], [16, 16], [18, -4],
  [-12, 18], [12, 18], [-20, -14], [20, -14], [8, -18], [-8, -18],
  [-19, 12], [19, 11], [0, 19],
];

function Trees() {
  const ref = useRef<THREE.InstancedMesh>(null);
  const geo = useMemo(() => new THREE.ConeGeometry(0.85, 2.3, 6), []);
  const mat = useMemo(() => new THREE.MeshStandardMaterial({ color: "#7f9470" }), []);
  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();
    TREES.forEach((p, i) => {
      dummy.position.set(p[0], 1.15, p[1]);
      dummy.scale.setScalar(0.85 + (i % 4) * 0.12);
      dummy.rotation.y = i;
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
  }, []);
  return <instancedMesh ref={ref} args={[geo, mat, TREES.length]} castShadow />;
}

function Plate({
  text,
  position,
  gold = false,
}: {
  text: string;
  position: [number, number, number];
  gold?: boolean;
}) {
  const map = useNameplate(text, gold);
  if (!map) return null;
  return (
    <sprite position={position} scale={[2.1, 0.52, 1]}>
      <spriteMaterial map={map} transparent depthWrite={false} />
    </sprite>
  );
}

function House({
  position,
  roof,
  onClick,
  sign,
  yaw = 0,
  scale = 1,
}: {
  position: [number, number, number];
  roof: string;
  onClick?: () => void;
  sign: string;
  yaw?: number;
  scale?: number;
}) {
  return (
    <group
      position={position}
      rotation={[0, yaw, 0]}
      scale={scale}
      onClick={
        onClick
          ? (e) => {
              e.stopPropagation();
              onClick();
            }
          : undefined
      }
    >
      <mesh position={[0, 0.12, 0]} receiveShadow>
        <boxGeometry args={[4.8, 0.24, 3.8]} />
        <meshStandardMaterial color="#cbbba8" />
      </mesh>
      <mesh position={[0, 1.35, 0]} castShadow receiveShadow>
        <boxGeometry args={[4.2, 2.15, 3.2]} />
        <meshStandardMaterial color="#f7f3ea" />
      </mesh>
      <mesh position={[-1.15, 2.55, 0]} rotation={[0, 0, 0.42]} castShadow>
        <boxGeometry args={[2.5, 0.16, 3.45]} />
        <meshStandardMaterial color={roof} />
      </mesh>
      <mesh position={[1.15, 2.55, 0]} rotation={[0, 0, -0.42]} castShadow>
        <boxGeometry args={[2.5, 0.16, 3.45]} />
        <meshStandardMaterial color={roof} />
      </mesh>
      <mesh position={[1.55, 2.35, -0.9]} castShadow>
        <boxGeometry args={[0.38, 0.7, 0.38]} />
        <meshStandardMaterial color="#8d7464" />
      </mesh>
      <mesh position={[0, 0.55, 1.66]} castShadow>
        <boxGeometry args={[0.72, 1.15, 0.08]} />
        <meshStandardMaterial color="#6b4a36" />
      </mesh>
      <mesh position={[0, 0.22, 2.05]} receiveShadow>
        <boxGeometry args={[1.3, 0.16, 0.7]} />
        <meshStandardMaterial color="#d9cfc0" />
      </mesh>
      <mesh position={[-1.15, 1.45, 1.62]}>
        <boxGeometry args={[0.72, 0.62, 0.06]} />
        <meshStandardMaterial color="#d5e2dc" />
      </mesh>
      <mesh position={[1.15, 1.45, 1.62]}>
        <boxGeometry args={[0.72, 0.62, 0.06]} />
        <meshStandardMaterial color="#d5e2dc" />
      </mesh>
      <Plate text={sign} position={[0, 3.55, 0]} />
    </group>
  );
}

function FlyMesh({ fly, onClick }: { fly: Fly; onClick: () => void }) {
  const slot = SLOTS[fly.slot] ?? SLOTS[SLOTS.length - 1];
  const group = useRef<THREE.Group>(null);
  const wingL = useRef<THREE.Mesh>(null);
  const wingR = useRef<THREE.Mesh>(null);
  const facing = useRef<THREE.Group>(null);
  const phase = fly.id.length * 0.7;
  const pos = useRef(new THREE.Vector3(slot[0], 0.55, slot[1]));
  const goal = useRef(new THREE.Vector3(slot[0] + 4, 0, slot[1] + 2));
  const heading = useRef(phase);
  const cruise = 2.1 + (fly.slot % 3) * 0.35;
  useFrame((_, delta) => {
    const dt = Math.min(delta, 0.05);
    const now = performance.now() / 1000;
    const errand = useTown.getState().errand;
    const headingOut = errand?.flyId === fly.id;
    if (headingOut) {
      goal.current.set(errand.place === "garden" ? 8 : DESK[0], 0, errand.place === "garden" ? 7 : DESK[1]);
    }
    let dx = goal.current.x - pos.current.x;
    let dz = goal.current.z - pos.current.z;
    let dist = Math.hypot(dx, dz);
    if (!headingOut && dist < 0.55) {
      for (let n = 0; n < 8; n += 1) {
        const ang = Math.random() * Math.PI * 2;
        const reach = 6 + Math.random() * 8;
        const nx = THREE.MathUtils.clamp(Math.cos(ang) * reach + slot[0] * 0.2, -18, 18);
        const nz = THREE.MathUtils.clamp(Math.sin(ang) * reach + slot[1] * 0.15, -12, 17);
        if (!blocked(nx, nz)) {
          goal.current.set(nx, 0, nz);
          break;
        }
      }
      dx = goal.current.x - pos.current.x;
      dz = goal.current.z - pos.current.z;
      dist = Math.hypot(dx, dz);
    }
    const desired = Math.atan2(dx, dz);
    let turn = desired - heading.current;
    turn = Math.atan2(Math.sin(turn), Math.cos(turn));
    heading.current += turn * (1 - Math.exp(-3.4 * dt));
    const step = Math.min(dist, cruise) * dt;
    const nextX = pos.current.x + Math.sin(heading.current) * step;
    const nextZ = pos.current.z + Math.cos(heading.current) * step;
    if (!blocked(nextX, nextZ)) {
      pos.current.x = nextX;
      pos.current.z = nextZ;
    } else if (!headingOut) {
      goal.current.set(slot[0], 0, slot[1]);
    }
    pos.current.y = 0.62 + Math.sin(now * 2.4 + phase) * 0.1 + Math.min(0.9, dist * 0.06);
    if (group.current) group.current.position.copy(pos.current);
    if (facing.current) {
      facing.current.rotation.y = heading.current;
      facing.current.rotation.z = THREE.MathUtils.clamp(-turn * 0.35, -0.4, 0.4);
    }
    const flap = Math.sin(now * (16 + cruise * 4) + phase) * 0.75;
    if (wingL.current) wingL.current.rotation.z = 0.35 + flap;
    if (wingR.current) wingR.current.rotation.z = -0.35 - flap;
  });
  return (
    <group
      ref={group}
      position={[slot[0], 0.4, slot[1]]}
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
    >
      <group ref={facing} scale={1.35}>
        <mesh position={[0, 0, -0.26]} rotation={[Math.PI / 2, 0, 0]} castShadow>
          <capsuleGeometry args={[0.1, 0.32, 4, 12]} />
          <meshStandardMaterial color={fly.accent === "moss" ? "#d7d2b8" : "#e7d7b8"} />
        </mesh>
        {[ -0.38, -0.26, -0.14 ].map((z) => (
          <mesh key={z} position={[0, 0, z]} rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.102, 0.102, 0.035, 12]} />
            <meshStandardMaterial color="#2a241c" />
          </mesh>
        ))}
        <mesh position={[0, 0.03, 0.06]} castShadow>
          <sphereGeometry args={[0.125, 16, 12]} />
          <meshStandardMaterial color="#f3e6cc" />
        </mesh>
        {[-0.045, 0, 0.045].map((x) => (
          <mesh key={x} position={[x, 0.08, 0.06]}>
            <boxGeometry args={[0.012, 0.02, 0.18]} />
            <meshStandardMaterial color="#6d5a42" />
          </mesh>
        ))}
        <mesh position={[0, 0.02, 0.22]} castShadow>
          <sphereGeometry args={[0.08, 12, 10]} />
          <meshStandardMaterial color="#cbb892" />
        </mesh>
        {[-0.07, 0.07].map((x) => (
          <mesh key={x} position={[x, 0.035, 0.24]}>
            <sphereGeometry args={[0.072, 14, 12]} />
            <meshStandardMaterial color="#9d2b24" roughness={0.32} />
          </mesh>
        ))}
        {[-0.03, 0.03].map((x) => (
          <mesh key={`ant-${x}`} position={[x, 0.08, 0.3]} rotation={[1.1, 0, x > 0 ? -0.4 : 0.4]}>
            <cylinderGeometry args={[0.006, 0.006, 0.12, 6]} />
            <meshStandardMaterial color="#3a2c22" />
          </mesh>
        ))}
        <mesh ref={wingL} position={[-0.06, 0.14, -0.02]} rotation={[0.55, 0.35, 0.9]}>
          <circleGeometry args={[0.34, 18]} />
          <meshStandardMaterial color="#8e99a6" transparent opacity={0.32} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
        <mesh ref={wingR} position={[0.06, 0.14, -0.02]} rotation={[0.55, -0.35, -0.9]}>
          <circleGeometry args={[0.34, 18]} />
          <meshStandardMaterial color="#8e99a6" transparent opacity={0.32} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
        {[
          [-0.1, 0.08],
          [-0.12, -0.02],
          [-0.1, -0.12],
          [0.1, 0.08],
          [0.12, -0.02],
          [0.1, -0.12],
        ].map(([x, z]) => (
          <group key={`${x}-${z}`} position={[x, -0.02, z]} rotation={[0.4, 0, x > 0 ? -0.8 : 0.8]}>
            <mesh>
              <cylinderGeometry args={[0.008, 0.008, 0.16, 5]} />
              <meshStandardMaterial color="#5a4030" />
            </mesh>
            <mesh position={[x > 0 ? 0.05 : -0.05, -0.08, 0.02]} rotation={[0.6, 0, x > 0 ? 0.7 : -0.7]}>
              <cylinderGeometry args={[0.007, 0.007, 0.14, 5]} />
              <meshStandardMaterial color="#4a3428" />
            </mesh>
          </group>
        ))}
      </group>
      <Plate
        text={fly.coins > 0 ? `${fly.name}  ${fly.coins}` : fly.name}
        position={[0, 0.85, 0]}
        gold={fly.coins > 0}
      />
    </group>
  );
}

const BUILDINGS: {
  position: [number, number, number];
  roof: string;
  sign: string;
  place?: "school" | "bed" | "bank";
  yaw: number;
  scale: number;
}[] = [
  { position: [-14, 0, -8], roof: "#cbbba8", sign: "School", place: "school", yaw: 0.15, scale: 1 },
  { position: [14, 0, -8], roof: "#b7c3ae", sign: "Bunk", place: "bed", yaw: -0.2, scale: 1 },
  { position: [0, 0, -16.5], roof: "#d9cbb6", sign: "Till", place: "bank", yaw: 0, scale: 0.92 },
  { position: [-9, 0, 13], roof: "#c9b59a", sign: "Porch", yaw: 0.55, scale: 0.72 },
  { position: [10, 0, 13.5], roof: "#aeb89a", sign: "Shed", yaw: -0.35, scale: 0.64 },
  { position: [17, 0, 1.5], roof: "#d4c4ae", sign: "Loft", yaw: 1.15, scale: 0.7 },
  { position: [-17, 0, 3], roof: "#c4b49a", sign: "Nest", yaw: -0.85, scale: 0.68 },
  { position: [6, 0, -14], roof: "#b7c3ae", sign: "Nook", yaw: 0.4, scale: 0.58 },
];

function blocked(x: number, z: number) {
  for (const b of BUILDINGS) {
    const dx = x - b.position[0];
    const dz = z - b.position[2];
    const c = Math.cos(-b.yaw);
    const s = Math.sin(-b.yaw);
    const lx = dx * c - dz * s;
    const lz = dx * s + dz * c;
    if (Math.abs(lx) < 2.7 * b.scale && Math.abs(lz) < 2.2 * b.scale) return true;
  }
  return false;
}
function Desk() {
  return (
    <group position={[DESK[0], 0, DESK[1]]}>
      <mesh position={[-0.7, 0.38, -0.38]} castShadow>
        <boxGeometry args={[0.08, 0.76, 0.08]} />
        <meshStandardMaterial color="#6b4a36" />
      </mesh>
      <mesh position={[0.7, 0.38, -0.38]} castShadow>
        <boxGeometry args={[0.08, 0.76, 0.08]} />
        <meshStandardMaterial color="#6b4a36" />
      </mesh>
      <mesh position={[-0.7, 0.38, 0.38]} castShadow>
        <boxGeometry args={[0.08, 0.76, 0.08]} />
        <meshStandardMaterial color="#6b4a36" />
      </mesh>
      <mesh position={[0.7, 0.38, 0.38]} castShadow>
        <boxGeometry args={[0.08, 0.76, 0.08]} />
        <meshStandardMaterial color="#6b4a36" />
      </mesh>
      <mesh position={[0, 0.8, 0]} castShadow receiveShadow>
        <boxGeometry args={[1.8, 0.1, 1.05]} />
        <meshStandardMaterial color="#c4a574" />
      </mesh>
      <mesh position={[-0.35, 0.88, 0.05]} rotation={[-0.02, 0.2, 0]}>
        <boxGeometry args={[0.42, 0.02, 0.55]} />
        <meshStandardMaterial color="#f7f3ea" />
      </mesh>
      <mesh position={[0.42, 0.95, -0.15]}>
        <cylinderGeometry args={[0.04, 0.06, 0.22, 8]} />
        <meshStandardMaterial color="#8d7464" />
      </mesh>
      <mesh position={[0.42, 1.12, -0.15]}>
        <sphereGeometry args={[0.1, 10, 8]} />
        <meshStandardMaterial color="#f3eee4" emissive="#f3eee4" emissiveIntensity={0.25} />
      </mesh>
      <Plate text="Desk" position={[0, 1.45, 0]} />
    </group>
  );
}

function CoinStacks({ total }: { total: number }) {
  const piles = Math.min(8, Math.max(1, Math.round(total / 25)));
  const spots = useMemo(() => {
    return Array.from({ length: piles }, (_, i) => {
      const a = (i / piles) * Math.PI - Math.PI / 2;
      return [Math.sin(a) * 2.2, Math.cos(a) * 1.4] as const;
    });
  }, [piles]);
  return (
    <group position={[0, 0, -16.5]}>
      {spots.map((s, i) => (
        <mesh key={i} position={[s[0], 0.12, s[1] + 2.4]} rotation={[-Math.PI / 2, 0, 0]} castShadow>
          <cylinderGeometry args={[0.18, 0.18, 0.06, 12]} />
          <meshStandardMaterial color="#d4a017" metalness={0.45} roughness={0.35} />
        </mesh>
      ))}
    </group>
  );
}

export function World() {
  const openPlace = useTown((s) => s.openPlace);
  const selectFly = useTown((s) => s.selectFly);
  const extra = useTown((s) => s.extra);
  const mode = useTown((s) => s.camera);
  const flies = useMemo(() => [...SEED, ...extra], [extra]);
  const coins = flies.reduce((n, f) => n + f.coins, 0);
  if (mode === "brain") return <NeuronStage />;

  return (
    <>
      <color attach="background" args={["#d5decc"]} />
      <fog attach="fog" args={["#d5decc", 24, 70]} />
      <hemisphereLight args={["#f7f4ee", "#8ea584", 1.05]} />
      <directionalLight
        position={[12, 18, 8]}
        intensity={1.35}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
        shadow-camera-far={50}
        shadow-camera-left={-24}
        shadow-camera-right={24}
        shadow-camera-top={24}
        shadow-camera-bottom={-24}
      />
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <circleGeometry args={[28, 48]} />
        <meshStandardMaterial color="#b7c4a4" />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 2]} receiveShadow>
        <planeGeometry args={[3.2, 22]} />
        <meshStandardMaterial color="#e4dfd2" />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 6]} receiveShadow>
        <planeGeometry args={[20, 2.2]} />
        <meshStandardMaterial color="#e4dfd2" />
      </mesh>
      <Trees />
      {BUILDINGS.map((b) => (
        <House
          key={b.sign}
          position={b.position}
          roof={b.roof}
          sign={b.sign}
          yaw={b.yaw}
          scale={b.scale}
          onClick={b.place ? () => openPlace(b.place!) : undefined}
        />
      ))}
      <mesh position={[14, 0.35, -6.2]} castShadow>
        <boxGeometry args={[1.6, 0.45, 0.8]} />
        <meshStandardMaterial color="#f3eee4" />
      </mesh>
      <mesh position={[14, 0.62, -6.2]}>
        <boxGeometry args={[0.7, 0.18, 0.55]} />
        <meshStandardMaterial color="#3d6d8c" />
      </mesh>
      <mesh position={[-14, 1.3, -6.3]}>
        <boxGeometry args={[1.8, 1.1, 0.08]} />
        <meshStandardMaterial color="#1a211c" />
      </mesh>
      <CoinStacks total={coins} />
      <Desk />
      {flies.map((fly) => (
        <FlyMesh key={fly.id} fly={fly} onClick={() => selectFly(fly.id)} />
      ))}
    </>
  );
}
