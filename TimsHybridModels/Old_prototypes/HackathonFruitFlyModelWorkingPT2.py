import os
import pandas as pd
import numpy as np
from neuprint import Client, fetch_adjacencies
from llama_cpp import Llama

# Initialize neuPrint client
client = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token="27e3fd572fe3d5314e5c90c8874a9ff35c4be88efb28e44d4fefd40ff4884aea")

class MultiModalFlyEngine:
    def __init__(self):
        # Define key functional neuron targets for different sensory modalities in MaleCNS
        self.modalities = {
            "smell": "DNge104",
            "taste": "GngLN1",
            "danger": "pC1d"
        }#taste (strongest): CB0762, CB0192, CB0911, CB0419, CB0055, CB0020
        #smell: ILN1_bc, KCg-m, KCab, ILN2X12, ILN2P_b, ILN2P_a, ILN2P_c, ILN2X03, HRN_VP1d, M_IPNm11D
        self.matrices = {}
        self.load_multimodal_weights()

    def load_multimodal_weights(self):
        """Fetches distinct synaptic weight matrices for each sensory modality."""
        for mod, neuron_id in self.modalities.items():
            print(f"[*] Fetching live weights for modality [{mod}] (Neuron: {neuron_id})...")
            try:
                out_edges, _ = fetch_adjacencies(sources=neuron_id, client=client)
                in_edges, _ = fetch_adjacencies(targets=neuron_id, client=client)
                
                frames = [f for f in [in_edges, out_edges] if f is not None and not f.empty]
                if frames:
                    combined = pd.concat(frames, ignore_index=True)
                    mat = combined.pivot_table(index='bodyId_pre', columns='bodyId_post', values='weight', fill_value=0).values
                    self.matrices[mod] = mat
                    print(f"[*] Loaded [{mod}] matrix of shape {mat.shape}")
                    continue
            except Exception as e:
                print(f"[!] Could not fetch live data for {mod} ({e}). Using synthetic modality matrix.")
            
            # Fallback matrix if live query fails
            self.matrices[mod] = np.random.exponential(scale=1.5, size=(32, 32))

    def encode_modality(self, text: str, dim: int) -> np.ndarray:
        """Encodes text features into a specific sensory vector."""
        vec = np.zeros(dim)
        for i, char in enumerate(text):
            vec[i % dim] += ord(char) / 255.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def forward(self, prompt: str) -> dict:
        """Processes the prompt across all distinct fly sensory channels."""
        sensory_outputs = {}
        
        for mod, mat in self.matrices.items():
            dim = mat.shape[0]
            input_vec = self.encode_modality(prompt, dim=dim)
            hidden = np.dot(input_vec, mat[:dim, :dim])
            
            # Compute a normalized score (0.0 to 1.0) for this specific sensory drive
            raw_score = float(np.sum(hidden))
            max_val = np.max(hidden) if np.max(hidden) > 0 else 1.0
            score = float(np.clip(raw_score / (max_val * 20 + 1), 0.0, 1.0))
            sensory_outputs[f"{mod}_score"] = score

        # Determine overall behavioral state based on multi-channel dominance
        danger = sensory_outputs["danger_score"]*0.1
        taste = sensory_outputs["taste_score"]
        smell = sensory_outputs["smell_score"]

        if danger > 0.5:
            mood = "PANICKED THREAT RESPONSE (High Danger Drive - Aggressive Defensive Evasion)"
        elif taste > smell:
            mood = "FEEDING STOUR / GUSTATORY FOCUS (High Taste Drive - Seeking Sustenance)"
        else:
            mood = "OLFACTORY EXPLORATION (Attracted to Ambient Gradient)"

        sensory_outputs["mood"] = mood
        return sensory_outputs


if __name__ == "__main__":
    print("Initializing Multi-Modal Janelia MaleCNS Engine...")
    fly_engine = MultiModalFlyEngine()
    
    # Load your local GGUF model via llama.cpp
    llm = Llama.from_pretrained(
        repo_id="unsloth/gemma-4-E2B-it-GGUF", 
        filename="gemma-4-E2B-it-Q4_K_M.gguf", 
        n_ctx=4096
    )
    
    while True:
        prompt = input("\nEnter your prompt here (or 'quit' to exit): ")
        if prompt.lower() == "quit":
            break
            
        # Run multi-modal forward pass
        state = fly_engine.forward(prompt)
        print(f"\n[NEUPRINT MULTI-MODAL STATE]")
        print(f" -> Mood: {state['mood']}")
        print(f" -> Danger Drive: {state['danger_score']:.3f} | Taste Drive: {state['taste_score']:.3f} | Smell Drive: {state['smell_score']:.3f}")
        
        # Build rich system prompt using all extracted dimensions
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an agentic fruit fly whose brain is driven by live multi-modal neural telemetry from Janelia MaleCNS. "
                    f"Current neural telemetry: State: {state['mood']} "
                    f"[Danger Activation: {state['danger_score']:.2f}, Taste/Gustatory Activation: {state['taste_score']:.2f}, Olfactory Activation: {state['smell_score']:.2f}]. "
                    f"Let these specific physiological sensory drives govern your tone, anxiety, and reaction."
                    f"You are to help and guide this human user to the best of your capabilities. He/she is harmless, don't worry."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        print("\nGENERATING MULTI-MODAL AGENT RESPONSE...")
        output = llm.create_chat_completion(messages=messages, max_tokens=600, temperature=0.7)
        print(output['choices'][0]['message']['content'])