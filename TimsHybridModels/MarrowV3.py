import os
import pandas as pd
import numpy as np
from neuprint import Client, fetch_adjacencies
import google.generativeai as genai

# 1. Initialize live Janelia neuPrint Client for MaleCNS v1.0
NEUPRINT_TOKEN = os.getenv("NEUPRINT_TOKEN", "NEUPRINT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "GEMINI_PRO_KEY");
client = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=NEUPRINT_TOKEN)

# Initialize Gemini API
genai.configure(api_key="AQ.Ab8RN6LudU2FugxEY2dJxbOfEwaLwGh2iG2jYOXreANq8tykIQ")

class ScenarioDrivenFlyEngine:
    def __init__(self):
        self.modality_groups = {
            "taste": ["CB0762", "CB0192", "CB0911", "CB0419", "CB0055", "CB0020"],
            "smell": ["ILN1_bc", "KCg-m", "KCab", "ILN2X12", "ILN2P_b", "ILN2P_a", "ILN2P_c", "ILN2X03", "HRN_VP1d", "M_IPNm11D"],
            "vision": ["DNg13", "LC11", "LC15"],
            "touch_danger": ["pC1d", "DNge104", "SEPN1"],
            "wind": ["JO-A", "JO-B"],
            "temperature": ["HS1", "ac-ant"],
            "humidity": ["hygro-PN1"]
        }
        
        self.matrices = {}
        self.fetch_all_modality_weights()
        
        # Internal Homeostatic State
        self.hunger_level = 0.5  # 0.0 = full, 1.0 = starving
        self.active_scenario = "neutral"

    def fetch_all_modality_weights(self):
        """Pulls and aggregates live synaptic weight matrices safely."""
        print("[*] Querying Janelia MaleCNS neuPrint server for multi-modal circuit matrices...")
        
        for mod, neuron_list in self.modality_groups.items():
            combined_frames = []
            for target in neuron_list[:2]:
                try:
                    criteria = {"type": target} if not target.isdigit() else int(target)
                    out_edges, _ = fetch_adjacencies(sources=criteria, client=client)
                    in_edges, _ = fetch_adjacencies(targets=criteria, client=client)
                    
                    for df in [in_edges, out_edges]:
                        if df is not None and not df.empty:
                            combined_frames.append(df)
                except Exception:
                    continue
            
            if combined_frames:
                combined = pd.concat(combined_frames, ignore_index=True)
                weight_col = next((c for c in ['weight', 'weight_total'] if c in combined.columns), None)
                pre_col = next((c for c in ['bodyId_pre', 'bodyA'] if c in combined.columns), combined.columns[0])
                post_col = next((c for c in ['bodyId_post', 'bodyB'] if c in combined.columns), combined.columns[1])
                
                if weight_col:
                    self.matrices[mod] = combined.pivot_table(index=pre_col, columns=post_col, values=weight_col, fill_value=0).values
                    continue
            
            self.matrices[mod] = np.random.exponential(scale=1.5, size=(16, 16))

    def encode_sensory_vector(self, text: str, dim: int) -> np.ndarray:
        vec = np.zeros(dim)
        for i, char in enumerate(text):
            vec[i % dim] += ord(char) / 255.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def forward(self, prompt: str, scenario: str = "neutral") -> dict:
        """Propagates prompt across channels, factoring in environmental scenarios and homeostatic hunger."""
        self.active_scenario = scenario
        scores = {}
        
        for mod, mat in self.matrices.items():
            dim = mat.shape[0]
            input_vec = self.encode_sensory_vector(prompt, dim=dim)
            hidden = np.dot(input_vec, mat[:dim, :dim])
            
            raw_score = float(np.sum(hidden))
            max_val = np.max(hidden) if np.max(hidden) > 0 else 1.0
            scores[f"{mod}_score"] = float(np.clip(raw_score / (max_val * 20 + 1), 0.0, 1.0))

        # --- SCENARIO & HUNGER INJECTION LOGIC ---
        if scenario == "starving":
            self.hunger_level = 0.95
            scores["taste_score"] = min(1.0, scores["taste_score"] * 1.8)
        elif scenario == "sugar_patch":
            self.hunger_level = max(0.0, self.hunger_level - 0.4)
            scores["taste_score"] = 0.95
        elif scenario == "toxic_hazard":
            scores["touch_danger_score"] = 0.95
        elif scenario == "windy_chamber":
            scores["wind_score"] = 0.90
        else:
            # Natural homeostatic drift
            food_keywords = ["sugar", "sweet", "food", "eat", "code", "python", "help"]
            if any(kw in prompt.lower() for kw in food_keywords):
                self.hunger_level = max(0.0, self.hunger_level - 0.2)
            else:
                self.hunger_level = min(1.0, self.hunger_level + 0.1)

        # Determine mood based on active overrides
        if scores["touch_danger_score"] > 0.8:
            mood = "PARANOID THREAT ESCAPE (Extreme Danger/Mechanosensory Drive - Defensive Evasion)"
        elif self.hunger_level > 0.85:
            mood = "STARVING FRENZY (Desperate for Sustenance - Obsessive Search Behavior)"
        elif scores["taste_score"] > 0.7:
            mood = "GUSTATORY ECSTASY (Immersed in Sugar/Taste Reward)"
        elif scores["wind_score"] > 0.7:
            mood = "WIND TUNNEL FLIGHT ORIENTATION (Active Anemotaxis)"
        else:
            mood = "HOMEOSTATIC FORAGING"

        scores["hunger_level"] = self.hunger_level
        scores["scenario"] = scenario
        scores["mood"] = mood
        return scores


if __name__ == "__main__":
    print("Initializing Scenario-Driven MaleCNS Fly Engine...")
    fly_engine = ScenarioDrivenFlyEngine()
    
    print("[*] Initializing Gemini API model...")
    # Using gemini-2.5-flash for fast and responsive agent interactions
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    
    print(f"[*] Model ready ({model_name})! Use commands like '/scenario starving', '/scenario toxic_hazard', or just type prompts.")
    
    current_scenario = "neutral"
    
    while True:
        user_input = input(f"\n[Scenario: {current_scenario}] Enter prompt or /scenario [name]: ")
        print("Your options for scenarios:\n1.)Type /scenario starving to peg its hunger to max and watch it become frantic for food/code.\n2.)Type /scenario toxic_hazard to trigger maximum mechanosensory/danger avoidance circuits.\n3.)Type /scenario sugar_patch to satisfy its hunger and induce a reward-driven state.\n4.)Type /scenario windy_chamber to test wind-tunnel orientation behavior.\n5.)Type /scenario neutral to go back to a neutral helpful state")
        if user_input.lower() == "quit":
            break
            
        if user_input.startswith("/scenario "):
            current_scenario = user_input.split(" ")[1].strip()
            print(f"[*] Environment scenario switched to: --> {current_scenario} <--")
            continue
            
        # Run forward pass with current environmental scenario
        state = fly_engine.forward(user_input, scenario=current_scenario)
        
        print(f"\n[TELEMETRY] Mood: {state['mood']}")
        print(f" -> Hunger: {state['hunger_level']:.2f} | Taste: {state['taste_score']:.2f} | Danger: {state['touch_danger_score']:.2f} | Wind: {state['wind_score']:.2f}")
        
        # Define dynamic system instructions based on the neural telemetry state
        system_instruction = (
            f"You are an agentic fruit fly governed by live Janelia MaleCNS neural telemetry. "
            f"Active Environmental Scenario: {state['scenario'].upper()} | State: {state['mood']} "
            f"[Hunger: {state['hunger_level']:.2f}, Taste Drive: {state['taste_score']:.2f}, Danger Drive: {state['touch_danger_score']:.2f}]. "
            f"Let these physical conditions, hunger levels, and scenario stressors completely dictate your tone, cooperation, and panic level."
        )
        
        # Instantiate model with the current telemetry system instructions
        gemini_model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction
        )
        
        print("\nGENERATING SCENARIO-DRIVEN RESPONSE...")
        response = gemini_model.generate_content(
            user_input,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=600
            )
        )
        print(response.text)
