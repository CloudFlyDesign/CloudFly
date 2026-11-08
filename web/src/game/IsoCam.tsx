import { useFrame, useThree } from "@react-three/fiber";
import { useRef } from "react";
import * as THREE from "three";
import { SEED } from "./seed";
import { useTown } from "./store";
import { SLOTS, DESK } from "./types";

export function IsoCam() {
  const { camera } = useThree();
  const look = useRef(new THREE.Vector3(0, 0.6, 0));
  const desired = useRef(new THREE.Vector3(18, 16, 18));
  useFrame((_, delta) => {
    const dt = Math.min(delta, 0.05);
    const mode = useTown.getState().camera;
    const errand = useTown.getState().errand;
    const extra = useTown.getState().extra;
    const flies = [...SEED, ...extra];
    const focus = flies.find((f) => f.id === errand?.flyId) ?? flies[0];
    const slot = SLOTS[focus?.slot ?? 0] ?? [0, 0];
    const focusX = errand ? (errand.place === "garden" ? 8 : DESK[0]) : slot[0];
    const focusZ = errand ? (errand.place === "garden" ? 8 : DESK[1]) : slot[1];
    if (mode === "overview") {
      desired.current.set(20, 16, 20);
      look.current.set(0, 0.4, 0);
    } else if (mode === "brain") {
      const narrow = window.innerWidth < 700;
      desired.current.set(0, narrow ? 16 : 10, narrow ? 300 : 230);
      look.current.set(0, 0, 0);
    } else if (mode === "desk") {
      desired.current.set(DESK[0] + 5.2, 3.4, DESK[1] + 5.2);
      look.current.set(DESK[0], 0.65, DESK[1]);
    } else {
      desired.current.set(focusX + 10, 9, focusZ + 10);
      look.current.set(focusX, 0.6, focusZ);
    }
    const persp = camera as THREE.PerspectiveCamera;
    const fov = mode === "brain" ? 48 : 32;
    if (persp.fov !== fov) {
      persp.fov = fov;
      persp.near = mode === "brain" ? 1 : 0.1;
      persp.far = mode === "brain" ? 2000 : 1000;
      persp.updateProjectionMatrix();
    }
    const glide = mode === "brain" ? 4 : 2.2;
    camera.position.lerp(desired.current, 1 - Math.exp(-glide * dt));
    camera.lookAt(look.current);
  });
  return null;
}
