import { Mic, Send } from "lucide-react";
import { useRef, useState } from "react";
import { hearLine } from "@/lib/hear";
import { classifyLine, traceLine } from "@/lib/classify";
import { sipRead } from "@/lib/exams";
import { bramSpeak } from "@/lib/gemini";
import { runTool } from "@/lib/tools";
import { phraseReply } from "@/lib/phrase";
import { speakLine } from "@/lib/speak";
import { routeTask } from "./route-task";
import { isScenario, SCENARIOS, stepMarrow } from "./marrow";
import { stepBram } from "./bram";
import { SEED } from "./seed";
import { useTown } from "./store";

let voice: HTMLAudioElement | null = null;

async function say(text: string, flyId: string, flyName: string) {
  try {
    const res = await speakLine({ data: { text, flyId, flyName } });
    if (res.ok) {
      voice?.pause();
      voice = new Audio(`data:audio/mpeg;base64,${res.audio}`);
      await voice.play();
      return;
    }
  } catch {
    /* fall through to the browser voice */
  }
  const synth = window.speechSynthesis;
  if (!synth) return;
  synth.cancel();
  synth.speak(new SpeechSynthesisUtterance(text));
}

export function ChatDock() {
  const messages = useTown((s) => s.messages);
  const pushMsg = useTown((s) => s.pushMsg);
  const setErrand = useTown((s) => s.setErrand);
  const setCamera = useTown((s) => s.setCamera);
  const setSpeaker = useTown((s) => s.setSpeaker);
  const setFiring = useTown((s) => s.setFiring);
  const extra = useTown((s) => s.extra);
  const marrowScenario = useTown((s) => s.marrowScenario);
  const marrowHunger = useTown((s) => s.marrowHunger);
  const setMarrow = useTown((s) => s.setMarrow);
  const bramScenario = useTown((s) => s.bramScenario);
  const bramHunger = useTown((s) => s.bramHunger);
  const setBram = useTown((s) => s.setBram);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [hearing, setHearing] = useState(false);
  const recRef = useRef<MediaRecorder | null>(null);

  async function toWav(blob: Blob) {
    const ctx = new AudioContext();
    try {
      const audio = await ctx.decodeAudioData(await blob.arrayBuffer());
      const channel = audio.getChannelData(0);
      const rate = 16000;
      const step = audio.sampleRate / rate;
      const count = Math.max(1, Math.floor(channel.length / step));
      const pcm = new Int16Array(count);
      for (let i = 0; i < count; i += 1) {
        const sample = Math.max(-1, Math.min(1, channel[Math.floor(i * step)] ?? 0));
        pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      }
      const header = new ArrayBuffer(44);
      const view = new DataView(header);
      const mark = (offset: number, text: string) => {
        for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
      };
      mark(0, "RIFF");
      view.setUint32(4, 36 + pcm.byteLength, true);
      mark(8, "WAVE");
      mark(12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, rate, true);
      view.setUint32(28, rate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      mark(36, "data");
      view.setUint32(40, pcm.byteLength, true);
      return new Blob([header, pcm], { type: "audio/wav" });
    } finally {
      await ctx.close();
    }
  }

  async function transcribe(blob: Blob) {
    const wav = await toWav(blob);
    const buf = new Uint8Array(await wav.arrayBuffer());
    let binary = "";
    for (let i = 0; i < buf.length; i += 0x8000) {
      binary += String.fromCharCode(...buf.subarray(i, i + 0x8000));
    }
    try {
      const heard = await hearLine({ data: { audio: btoa(binary), mime: "audio/wav" } });
      if (!heard.ok || !heard.text) {
        pushMsg({
          id: crypto.randomUUID(),
          role: "fly",
          flyName: "Town",
          text: heard.ok ? "I heard nothing. Try the microphone again." : heard.error,
        });
        return;
      }
      setText(heard.text);
      void send(heard.text);
    } catch (err) {
      pushMsg({
        id: crypto.randomUUID(),
        role: "fly",
        flyName: "Town",
        text: err instanceof Error ? err.message : "Could not turn that into text.",
      });
    }
  }

  async function listen() {
    if (recRef.current && recRef.current.state === "recording") {
      recRef.current.stop();
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      pushMsg({
        id: crypto.randomUUID(),
        role: "fly",
        flyName: "Town",
        text: "This browser has no microphone. Type the line instead.",
      });
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      rec.ondataavailable = (ev) => {
        if (ev.data.size) chunks.push(ev.data);
      };
      rec.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        setHearing(false);
        recRef.current = null;
        void transcribe(new Blob(chunks, { type: rec.mimeType || "audio/webm" }));
      };
      recRef.current = rec;
      setHearing(true);
      rec.start();
    } catch {
      setHearing(false);
      pushMsg({
        id: crypto.randomUUID(),
        role: "fly",
        flyName: "Town",
        text: "The microphone is blocked. Allow it, or type the line.",
      });
    }
  }

  async function send(raw: string) {
    const line = raw.trim();
    if (!line || busy) return;
    setText("");
    pushMsg({ id: crypto.randomUUID(), role: "user", text: line });
    const flies = [...SEED, ...extra];
    const command = line.match(/^\/scenario\s+([a-z_]+)/i);
    const scenario = command && isScenario(command[1]) ? command[1] : marrowScenario;
    if (command && !isScenario(command[1])) {
      pushMsg({
        id: crypto.randomUUID(),
        role: "fly",
        flyName: "Marrow",
        text: "Marrow knows neutral, starving, sugar_patch, toxic_hazard, and windy_chamber.",
      });
      return;
    }
    const routed = command
      ? {
          ...routeTask(/\bbram\b/i.test(line) ? "bram" : "marrow", flies),
          fly: flies.find((f) => f.id === (/\bbram\b/i.test(line) ? "bram" : "marrow")) ?? null,
        }
      : routeTask(line, flies);
    if (!routed.fly) {
      pushMsg({
        id: crypto.randomUUID(),
        role: "fly",
        flyName: "Town",
        text: "No fly here has that job. Open Town and raise one. A fly can only sort text into two labels.",
      });
      return;
    }
    setSpeaker(routed.fly.id);
    setErrand({ flyId: routed.fly.id, place: routed.place, line: `${routed.fly.name} is on its way.` });
    setBusy(true);
    let spoken = `${routed.fly.name} is back. ${routed.result}`;
    let mood = "";
    if (routed.fly.id === "bram") {
      const stepped = stepBram(command ? scenario : line, command ? scenario : bramScenario, bramHunger);
      setBram(stepped.scenario, stepped.hunger);
      mood = stepped.mood;
      spoken = stepped.line;
      setErrand({ flyId: "bram", place: routed.place, line: `${stepped.mood}.` });
      try {
        const said = await bramSpeak({
          data: {
            user: command ? `The room is now ${scenario}.` : line,
            scenario: stepped.scenario,
            mood: stepped.mood,
            hunger: stepped.hunger,
            taste: stepped.scores.taste,
            danger: stepped.scores.touch_danger,
          },
        });
        if (said.ok) spoken = said.text;
      } catch {
        /* keep a plain line, not only the numbers */
      }
      if (spoken === stepped.line) {
        const asked = line.replace(/^hey\s+bram[,:]?\s*/i, "").trim();
        spoken = /how were you made|how are you made|who made you/i.test(asked)
          ? "A frozen fruit-fly circuit fires on your sentence, then a model speaks in this mood. That is all I am."
          : stepped.mood.includes("STARVING")
            ? "I can't stay on your question. I need food."
            : stepped.mood.includes("PARANOID")
              ? "Don't come closer. Something here is wrong."
              : stepped.mood.includes("GUSTATORY")
                ? "This is wonderful. Ask me again while it lasts."
                : stepped.mood.includes("WIND")
                  ? "I'm flying. Say that once more."
                  : asked.length > 8
                    ? `I heard you. ${asked}`
                    : "I'm here. Ask me plainly.";
      }
      try {
        const traced = await traceLine({ data: { flyId: "bram", text: line } });
        if (traced.ok) setFiring({ flyId: "bram", active: traced.active, total: traced.total, cells: traced.cells });
      } catch {
        /* the reply still stands */
      }
    } else if (routed.fly.kind === "scenario") {
      const stepped = stepMarrow(command ? scenario : line, scenario, marrowHunger);
      setMarrow(stepped.scenario, stepped.hunger);
      mood = stepped.mood;
      spoken = stepped.line;
      try {
        const traced = await traceLine({ data: { flyId: routed.fly.id, text: line } });
        if (traced.ok) setFiring({ flyId: routed.fly.id, active: traced.active, total: traced.total, cells: traced.cells });
      } catch {
        /* Marrow still answers without a spike trace */
      }
    } else if (routed.fly.id === "sip") {
      spoken = sipRead(line);
    } else if (routed.fly.kind === "tool") {
      try {
        const traced = await traceLine({ data: { flyId: routed.fly.id, text: line } });
        if (traced.ok) setFiring({ flyId: routed.fly.id, active: traced.active, total: traced.total, cells: traced.cells });
      } catch {
        /* the tool can still answer */
      }
      try {
        const ran = await runTool({ data: { id: routed.fly.id, text: line } });
        spoken = ran.ok ? `${routed.fly.name} says ${ran.text}` : ran.error;
      } catch {
        spoken = `${routed.fly.name} could not reach its service.`;
      }
    } else if (routed.fly.labels.length >= 2) {
      try {
        const judged = await classifyLine({
          data: {
            flyId: routed.fly.id,
            text: line,
            labels: routed.fly.labels,
            keywords: routed.fly.keywords,
            examples: routed.fly.examples ?? [],
          },
        });
        if (judged.ok) {
          spoken = `${routed.fly.name} says ${judged.label}. ${judged.note}`;
          setFiring({ flyId: routed.fly.id, active: judged.activeNeurons, total: judged.totalSpikes, cells: judged.cells });
        }
      } catch {
        /* keep the local guess */
      }
    } else {
      try {
        const traced = await traceLine({ data: { flyId: routed.fly.id, text: line } });
        if (traced.ok) setFiring({ flyId: routed.fly.id, active: traced.active, total: traced.total, cells: traced.cells });
      } catch {
        /* this fly still answers, without a spike trace */
      }
    }
    if (routed.fly.kind !== "tool" && routed.fly.id !== "bram") {
      try {
        const phrased = await phraseReply({
          data: {
            user: line,
            flyName: routed.fly.name,
            job: routed.fly.job,
            result: spoken,
            honest: routed.fly.honest,
            mood,
          },
        });
        if (phrased.ok) spoken = phrased.text;
      } catch {
        /* keep the plain result */
      }
    }
    pushMsg({ id: crypto.randomUUID(), role: "fly", flyName: routed.fly.name, text: spoken });
    setCamera("brain");
    setErrand({ flyId: routed.fly.id, place: routed.place, line: spoken });
    setBusy(false);
    void say(spoken, routed.fly.id, routed.fly.name);
  }

  return (
    <aside className="pointer-events-auto relative mt-auto flex h-[46%] w-full max-w-none flex-col border-t border-line bg-foam/95 text-ink sm:mt-0 sm:h-full sm:max-w-sm sm:border-t-0 sm:border-r">
      <div className="border-b border-line px-4 py-3">
        <p className="text-xs tracking-wide text-muted">INBOX</p>
        <h2 className="font-display text-xl">Ask a fly</h2>
        <p className="mt-1 text-sm text-muted">Each fly answers in its own voice. Marrow changes with the room.</p>
        <div className="mt-2 flex flex-wrap gap-1">
          {SCENARIOS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`rounded-full border border-line px-2 py-1 text-xs ${marrowScenario === item.id ? "bg-ink text-foam" : "bg-foam"}`}
              onClick={() => void send(`/scenario ${item.id}`)}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>
      <ul className="min-h-0 flex-1 space-y-2 overflow-auto px-3 py-3">
        {messages.map((m) => (
          <li key={m.id} className={`rounded-2xl px-3 py-2 text-sm ${m.role === "user" ? "bg-sage" : "border border-line bg-foam"}`}>
            {m.flyName ? <p className="text-xs text-muted">{m.flyName}</p> : null}
            <p>{m.text}</p>
          </li>
        ))}
      </ul>
      <form
        className="flex gap-2 border-t border-line p-3"
        onSubmit={(e) => {
          e.preventDefault();
          void send(text);
        }}
      >
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={hearing ? "Listening… tap the mic to stop" : busy ? "The fly is answering…" : "Ask in one line"}
          disabled={busy}
          className="min-w-0 flex-1 rounded-full border border-line bg-foam px-3 py-2 text-sm"
        />
        <button type="button" aria-label="Speak" className="grid size-10 place-items-center rounded-full border border-line" onClick={listen}>
          <Mic className={`size-4 ${hearing ? "text-gold" : ""}`} />
        </button>
        <button type="submit" aria-label="Send" disabled={busy} className="grid size-10 place-items-center rounded-full bg-ink text-foam disabled:opacity-50">
          <Send className="size-4" />
        </button>
      </form>
    </aside>
  );
}
