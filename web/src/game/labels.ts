import { useMemo } from "react";
import * as THREE from "three";

export function useNameplate(text: string, gold = false) {
  return useMemo(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 128;
    const g = canvas.getContext("2d");
    if (!g) return null;
    g.clearRect(0, 0, 512, 128);
    g.fillStyle = "#1a211c";
    round(g, 16, 28, 480, 72, 36);
    g.fill();
    g.fillStyle = gold ? "#d4a017" : "#f3eee4";
    g.font = "600 48px Outfit, sans-serif";
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillText(text, 256, 64);
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }, [text, gold]);
}

function round(
  g: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  g.beginPath();
  g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r);
  g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r);
  g.arcTo(x, y, x + w, y, r);
  g.closePath();
}
