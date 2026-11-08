import os
import pandas as pd
import numpy as np
from neuprint import Client, fetch_adjacencies
from llama_cpp import Llama

# 1. Initialize live Janelia neuPrint Client for MaleCNS v1.0
NEUPRINT_TOKEN = os.getenv("NEUPRINT_TOKEN", "27e3fd572fe3d5314e5c90c8874a9ff35c4be88efb28e44d4fefd40ff4884aea")
client = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=NEUPRINT_TOKEN)

class HomeostaticMaleCNSFlyEngine:
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
        
        # Internal Homeostatic State: Hunger (0.0 = completely full, 1.0 = starving)
        self.hunger_level = 0.5

    def fetch_all_modality_weights(self):
        """Pulls and aggregates live synaptic weight matrices for each channel safely."""
        print("[*] Querying Janelia MaleCNS neuPrint server for multi-modal circuit matrices...")
        
        for mod, neuron_list in self.modality_groups.items():
            combined_frames = []
            print(f"    -> Fetching modality: [{mod}]...")
            
            for target in neuron_list[:2]:  # Limit query batch for speed
                try:
                    # Support both string types or body IDs
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
                
                # Automatically detect the weight column name safely
                weight_col = next((c for c in ['weight', 'weight_total'] if c in combined.columns), None)
                pre_col = next((c for c in ['bodyId_pre', 'bodyA'] if c in combined.columns), combined.columns[0])
                post_col = next((c for c in ['bodyId_post', 'bodyB'] if c in combined.columns), combined.columns[1])
                
                if weight_col:
                    mat = combined.pivot_table(index=pre_col, columns=post_col, values=weight_col, fill_value=0).values
                    self.matrices[mod] = mat
                    print(f"       Successfully loaded [{mod}] matrix of shape {mat.shape}")
                    continue
            
            # Fallback if specific live edges didn't resolve
            print(f"       Using biomimetic fallback matrix for [{mod}].")
            self.matrices[mod] = np.random.exponential(scale=1.5, size=(16, 16))

    def encode_sensory_vector(self, text: str, dim: int) -> np.ndarray:
        vec = np.zeros(dim)
        for i, char in enumerate(text):
            vec[i % dim] += ord(char) / 255.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def forward(self, prompt: str) -> dict:
        """Propagates prompt across channels and modulates taste/danger based on homeostatic hunger."""
        scores = {}
        
        for mod, mat in self.matrices.items():
            dim = mat.shape[0]
            input_vec = self.encode_sensory_vector(prompt, dim=dim)
            hidden = np.dot(input_vec, mat[:dim, :dim])
            
            raw_score = float(np.sum(hidden))
            max_val = np.max(hidden) if np.max(hidden) > 0 else 1.0
            scores[f"{mod}_score"] = float(np.clip(raw_score / (max_val * 20 + 1), 0.0, 1.0))

        # HOMEOSTATIC HUNGER MODULATION:
        # If the user prompt mentions food, sugar, or code/tasks (treated as digital sustenance), reduce hunger.
        # Otherwise, hunger creeps up over time.
        food_keywords = ["sugar", "sweet", "food", "eat", "code", "python", "help", "build"]
        is_nourishing = any(kw in prompt.lower() for kw in food_keywords)
        
        if is_nourishing:
            self.hunger_level = max(0.0, self.hunger_level - 0.25)
            # High hunger amplifies taste sensitivity biologically
            scores["taste_score"] = min(1.0, scores["taste_score"] * (1.0 + self.hunger_level))
        else:
            self.hunger_level = min(1.0, self.hunger_level + 0.15)

        # Behavioral state determination
        if self.hunger_level > 0.75:
            mood = "STARVING & DESPERATE (High Hunger Drive - Fixated entirely on finding sweet sustenance)"
        elif scores["touch_danger_score"] > 0.45 and self.hunger_level < 0.8:
            mood = "DEFENSIVE ESCAPE / THREAT PROTOCOL (High Mechanosensory/Danger Drive)"
        elif scores["taste_score"] > 0.4:
            mood = "GUSTATORY FIXATION (High Sugar/Taste Drive)"
        else:
            mood = "HOMEOSTATIC WANDERING"

        scores["hunger_level"] = self.hunger_level
        scores["mood"] = mood
        return scores


if __name__ == "__main__":
    print("Initializing Homeostatic Multi-Modal MaleCNS Fly Engine...")
    fly_engine = HomeostaticMaleCNSFlyEngine()
    
    print("[*] Loading Gemma 4 E2B GGUF model...")
    llm = Llama.from_pretrained(
        repo_id="unsloth/gemma-4-E2B-it-GGUF", 
        filename="gemma-4-E2B-it-Q4_K_M.gguf", 
        n_ctx=4096
    )
    print("[*] Model loaded successfully! Ready for agent loop.")
    
    while True:
        prompt = input("\nEnter your prompt here (or 'quit' to exit): ")
        if prompt.lower() == "quit":
            break
            
        state = fly_engine.forward(prompt)
        
        print(f"\n[NEUPRINT HOMEOSTATIC TELEMETRY]")
        print(f" -> Behavioral Mood: {state['mood']}")
        print(f" -> Hunger Level: {state['hunger_level']:.2f} (0.0 = Full, 1.0 = Starving)")
        print(f" -> Taste: {state['taste_score']:.2f} | Smell: {state['smell_score']:.2f} | Danger: {state['touch_danger_score']:.2f}")
        
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an agentic fruit fly whose cognitive core is governed by live multi-modal neural telemetry and homeostatic drives from Janelia MaleCNS. "
                    f"Current neural state: {state['mood']} "
                    f"[Internal Hunger State: {state['hunger_level']:.2f}, Taste Drive: {state['taste_score']:.2f}, Smell Drive: {state['smell_score']:.2f}, Danger/Touch: {state['touch_danger_score']:.2f}]. "
                    f"When your hunger level is high, you become erratic, demanding of sweet things or code solutions, and obsessed with finding food. Let this internal drive govern your persona."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        print("\nGENERATING HOMEOSTATIC AGENT RESPONSE...")
        output = llm.create_chat_completion(messages=messages, max_tokens=600, temperature=0.7)
        print(output['choices'][0]['message']['content'])