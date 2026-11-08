import { create } from "zustand";
import { SEED } from "./seed";
import { SLOTS, type Fly } from "./types";
import type { Scenario } from "./marrow";

const KEY = "cloudfly-town-v1";

function loadExtra(): Fly[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Fly[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

type Panel =
  | { kind: "fly"; id: string }
  | { kind: "place"; place: "bed" | "school" | "bank" }
  | null;

type TownState = {
  extra: Fly[];
  hydrated: boolean;
  started: boolean;
  panel: Panel;
  minting: boolean;
  mintError: string;
  guessText: string;
  tab: "chat" | "town";
  camera: "overview" | "follow" | "desk" | "brain";
  speaker: string;
  errand: { flyId: string; place: "desk" | "garden"; line: string } | null;
  firing: { flyId: string; active: number; total: number; cells: { i: number; n: number }[] } | null;
  messages: { id: string; role: "user" | "fly"; text: string; flyName?: string }[];
  marrowScenario: Scenario;
  marrowHunger: number;
  bramScenario: Scenario;
  bramHunger: number;
  setMarrow: (scenario: Scenario, hunger: number) => void;
  setBram: (scenario: Scenario, hunger: number) => void;
  hydrate: () => void;
  start: () => void;
  selectFly: (id: string) => void;
  openPlace: (place: "bed" | "school" | "bank") => void;
  closePanel: () => void;
  setMinting: (v: boolean) => void;
  setMintError: (v: string) => void;
  setGuess: (v: string) => void;
  setTab: (tab: "chat" | "town") => void;
  setCamera: (camera: "overview" | "follow" | "desk" | "brain") => void;
  setSpeaker: (id: string) => void;
  patchFly: (id: string, patch: Partial<Fly>) => void;
  setErrand: (errand: TownState["errand"]) => void;
  setFiring: (firing: TownState["firing"]) => void;
  pushMsg: (msg: TownState["messages"][number]) => void;
  addFly: (fly: Fly) => void;
  flies: () => Fly[];
  nextSlot: () => number;
};

export const useTown = create<TownState>((set, get) => ({
  extra: [],
  hydrated: false,
  started: false,
  panel: null,
  minting: false,
  mintError: "",
  guessText: "",
  tab: "chat",
  camera: "overview",
  speaker: "sip",
  errand: null,
  firing: null,
  messages: [
    {
      id: "hello",
      role: "fly",
      flyName: "Town",
      text: "Say what you need. A fly only classifies. I will send the line to the one whose job it is.",
    },
  ],
  marrowScenario: "neutral",
  marrowHunger: 0.5,
  bramScenario: "neutral",
  bramHunger: 0.5,
  setMarrow: (marrowScenario, marrowHunger) => set({ marrowScenario, marrowHunger }),
  setBram: (bramScenario, bramHunger) => set({ bramScenario, bramHunger }),
  hydrate: () => set({ extra: loadExtra(), hydrated: true }),
  start: () => set({ started: true }),
  selectFly: (id) => set({ panel: { kind: "fly", id }, guessText: "", mintError: "" }),
  openPlace: (place) => set({ panel: { kind: "place", place }, mintError: "" }),
  closePanel: () => set({ panel: null, mintError: "" }),
  setMinting: (minting) => set({ minting }),
  setMintError: (mintError) => set({ mintError }),
  setGuess: (guessText) => set({ guessText }),
  setTab: (tab) => set({ tab }),
  setCamera: (camera) => set({ camera }),
  setSpeaker: (speaker) => set({ speaker }),
  patchFly: (id, patch) => {
    const extra = get().extra.map((fly) => (fly.id === id ? { ...fly, ...patch } : fly));
    set({ extra });
    try {
      localStorage.setItem(KEY, JSON.stringify(extra));
    } catch {
      /* ignore quota */
    }
  },
  setErrand: (errand) =>
    set((state) => ({
      errand,
      camera: state.camera === "brain" ? "brain" : errand ? "follow" : state.camera,
    })),
  setFiring: (firing) => set({ firing }),
  pushMsg: (msg) => set({ messages: [...get().messages, msg].slice(-40) }),
  addFly: (fly) => {
    const extra = [...get().extra, fly];
    set({ extra, panel: { kind: "fly", id: fly.id }, guessText: "" });
    try {
      localStorage.setItem(KEY, JSON.stringify(extra));
    } catch {
      /* ignore quota */
    }
  },
  flies: () => [...SEED, ...get().extra],
  nextSlot: () => {
    const used = new Set(get().flies().map((f) => f.slot));
    const free = SLOTS.findIndex((_, i) => !used.has(i));
    return free === -1 ? SLOTS.length - 1 : free;
  },
}));
