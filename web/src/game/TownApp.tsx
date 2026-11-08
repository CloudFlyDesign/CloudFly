import { Canvas } from "@react-three/fiber";
import { useEffect, useState } from "react";
import { ChatDock } from "./ChatDock";
import { IsoCam } from "./IsoCam";
import { NeuronPanel } from "./NeuronPanel";
import { TownDock } from "./TownDock";
import { useTown } from "./store";
import { World } from "./World";

export default function TownApp() {
  const hydrate = useTown((s) => s.hydrate);
  const tab = useTown((s) => s.tab);
  const setTab = useTown((s) => s.setTab);
  const camera = useTown((s) => s.camera);
  const setCamera = useTown((s) => s.setCamera);
  const speaker = useTown((s) => s.speaker);
  const errand = useTown((s) => s.errand);
  const closePanel = useTown((s) => s.closePanel);
  const [sceneReady, setSceneReady] = useState(false);
  useEffect(() => {
    hydrate();
    setSceneReady(true);
  }, [hydrate]);

  return (
    <main className="flex h-dvh flex-col bg-sage text-ink">
      <header className="relative z-20 flex items-center justify-between gap-3 border-b border-line bg-foam px-4 py-2">
        <button
          type="button"
          className="flex items-center gap-2 font-display text-lg"
          onClick={() => {
            setTab("chat");
            setCamera("overview");
            closePanel();
          }}
        >
          <img src="/cloudfly-logo.png" alt="" className="h-8 w-8 object-contain" />
          CloudFly
        </button>
        <nav className="flex gap-1 rounded-full bg-sage p-1 text-sm">
          <button
            type="button"
            className={`rounded-full px-4 py-2 ${tab === "chat" && camera !== "brain" ? "bg-foam" : ""}`}
            onClick={() => {
              setTab("chat");
              setCamera("overview");
            }}
          >
            Chat
          </button>
          <button
            type="button"
            className={`rounded-full px-4 py-2 ${tab === "town" && camera !== "brain" ? "bg-foam" : ""}`}
            onClick={() => {
              setTab("town");
              setCamera("overview");
            }}
          >
            Town
          </button>
        </nav>
      </header>
      <div className="relative min-h-0 flex-1">
        {sceneReady ? (
          <Canvas
            shadows
            dpr={[1, 1.5]}
            camera={{ position: [20, 16, 20], fov: 32 }}
            onPointerMissed={() => closePanel()}
            className="absolute inset-0"
          >
            <World />
            <IsoCam />
          </Canvas>
        ) : (
          <div className="absolute inset-0 bg-sage" />
        )}
        {camera === "brain" ? null : (
          <div className="pointer-events-none absolute inset-0 flex flex-col sm:flex-row">
            {tab === "chat" ? <ChatDock /> : <TownDock />}
          </div>
        )}
        {camera === "brain" ? (
          <NeuronPanel speaker={speaker} note={errand ? errand.line : "Ask in Chat and this brain plays that fly."} />
        ) : null}
        {errand && camera !== "brain" ? (
          <p className="pointer-events-none absolute top-3 right-3 z-10 max-w-[16rem] rounded-2xl bg-foam/95 px-3 py-2 text-right text-xs text-ink">
            {errand.line}
          </p>
        ) : null}
        <div
          className={`pointer-events-none absolute z-20 ${
            camera === "brain" ? "top-2 right-2 left-2 flex justify-center" : "right-3 bottom-[48%] sm:bottom-3"
          }`}
        >
          <div className="pointer-events-auto flex max-w-full gap-1 overflow-x-auto rounded-full bg-foam/95 p-1 text-xs shadow-sm">
            {(["overview", "follow", "desk", "brain"] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                className={`rounded-full px-3 py-2 capitalize ${camera === mode ? "bg-sage" : ""}`}
                onClick={() => setCamera(mode)}
              >
                {mode === "brain" ? "Neurons" : mode}
              </button>
            ))}
          </div>
        </div>
      </div>
    </main>
  );
}
