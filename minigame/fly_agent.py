import os
import pandas as pd
import numpy as np
from neuprint import Client, fetch_adjacencies
from llama_cpp import Llama

NEUPRINT_TOKEN = os.getenv("NEUPRINT_TOKEN", "27e3fd572fe3d5314e5c90c8874a9ff35c4be88efb28e44d4fefd40ff4884aea")
client = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=NEUPRINT_TOKEN)

class FlyAgent:
    def __init__(self, agent_id: str, color: str, llm_instance: Llama):
        self.agent_id = agent_id
        self.color = color or "#00ffff"
        self.llm = llm_instance
        
        self.hunger_level = float(np.random.uniform(0.3, 0.6))
        self.matrices = {}
        
        self.modality_groups = {
            "taste": ["CB0762", "CB0192", "CB0911"],
            "smell": ["KCg-m", "KCab", "ILN1_bc"],
            "vision": ["DNg13", "LC11"],
            "touch_danger": ["pC1d", "DNge104"],
            "wind": ["JO-A"],
            "temperature": ["HS1"],
            "humidity": ["hygro-PN1"]
        }
        self.load_circuit_weights()

    def load_circuit_weights(self):
        for mod, neuron_list in self.modality_groups.items():
            combined_frames = []
            for target in neuron_list[:1]:
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

    def _encode_vector(self, text: str, dim: int) -> np.ndarray:
        vec = np.zeros(dim)
        for i, char in enumerate(text):
            vec[i % dim] += ord(char) / 255.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def compute_telemetry(self, text_input: str, scenario: str, objects: list, fly_pos: tuple) -> dict:
        scores = {}
        for mod, mat in self.matrices.items():
            dim = mat.shape[0]
            input_vec = self._encode_vector(text_input, dim=dim)
            hidden = np.dot(input_vec, mat[:dim, :dim])
            
            raw_score = float(np.sum(hidden))
            max_val = np.max(hidden) if np.max(hidden) > 0 else 1.0
            scores[f"{mod}_score"] = float(np.clip(raw_score / (max_val * 20 + 1), 0.0, 1.0))

        # Check proximity to generative objects in the 3D world
        nearest_object_dist = float('inf')
        nearest_object_type = None
        for obj in objects:
            dist = np.sqrt((fly_pos[0] - obj["x"])**2 + (fly_pos[2] - obj["z"])**2)
            if dist < nearest_object_dist:
                nearest_object_dist = dist
                nearest_object_type = obj["type"]

        # Boost sensory states if close to objects
        if nearest_object_dist < 3.5:
            if "sugar" in nearest_object_type or "food" in nearest_object_type:
                scores["taste_score"] = 0.95
                self.hunger_level = max(0.0, self.hunger_level - 0.25)
            elif "toxic" in nearest_object_type or "danger" in nearest_object_type:
                scores["touch_danger_score"] = 0.98
            elif "fan" in nearest_object_type or "wind" in nearest_object_type:
                scores["wind_score"] = 0.90

        # Standard Homeostatic & Scenario Modifiers
        if scenario == "starving_chamber":
            self.hunger_level = min(1.0, self.hunger_level + 0.2)
        elif scenario == "sugar_spill":
            self.hunger_level = max(0.0, self.hunger_level - 0.3)
        else:
            self.hunger_level = min(1.0, self.hunger_level + 0.05)

        if scores["touch_danger_score"] > 0.8:
            mood = "DEFENSIVE THREAT ESCAPE"
        elif self.hunger_level > 0.8:
            mood = "STARVING FRENZY"
        elif scores["taste_score"] > 0.7:
            mood = "FEEDING ON OBJECT"
        else:
            mood = "HOMEOSTATIC FORAGING"

        scores["hunger_level"] = self.hunger_level
        scores["mood"] = mood
        scores["nearest_object"] = nearest_object_type
        scores["object_distance"] = nearest_object_dist
        return scores

    def step(self, scenario: str, colony_chatter: list[dict], objects: list, fly_pos: tuple) -> dict:
        recent_logs = colony_chatter[-3:] if colony_chatter else []
        chatter_text = "\n".join([f"Fly {m['agent_id']}: {m['text']}" for m in recent_logs])
        
        telemetry = self.compute_telemetry(chatter_text or scenario, scenario, objects, fly_pos)
        
        obj_context = f"Nearest Object: {telemetry['nearest_object']} at distance {telemetry['object_distance']:.1f}m." if telemetry['nearest_object'] else "No objects nearby."
        
        system_instruction = (
            f"You are Fly #{self.agent_id} in a 3D simulation. "
            f"Active Scenario: {scenario.upper()} | State: {telemetry['mood']} | {obj_context} "
            f"Respond in 1 short, snappy sentence to your environment and peers."
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Chatter:\n{chatter_text}\n\nReaction:" if chatter_text else "React to environment:"}
        ]

        response = self.llm.create_chat_completion(messages=messages, max_tokens=90, temperature=0.7)
        utterance = response['choices'][0]['message']['content'].strip()

        return {
            "agent_id": self.agent_id,
            "color": self.color,
            "hunger": self.hunger_level * 100.0,
            "mood": telemetry['mood'],
            "text": utterance
        }