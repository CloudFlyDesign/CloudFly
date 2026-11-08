const held = new Set<string>();
let override: string[] | null = null;

export function press(code: string) {
  held.add(code);
}

export function release(code: string) {
  held.delete(code);
}

export function clearHeld() {
  held.clear();
}

export function setOverride(codes: string[] | null) {
  override = codes;
}

export function effectiveKeys(): string[] {
  return override ?? [...held];
}

export function bindKeyboard() {
  const down = (e: KeyboardEvent) => {
    const t = e.target as HTMLElement | null;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
    held.add(e.code);
  };
  const up = (e: KeyboardEvent) => held.delete(e.code);
  const blur = () => held.clear();
  window.addEventListener("keydown", down);
  window.addEventListener("keyup", up);
  window.addEventListener("blur", blur);
  document.addEventListener("visibilitychange", blur);
  return () => {
    window.removeEventListener("keydown", down);
    window.removeEventListener("keyup", up);
    window.removeEventListener("blur", blur);
    document.removeEventListener("visibilitychange", blur);
    held.clear();
  };
}
