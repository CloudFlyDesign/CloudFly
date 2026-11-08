from transformers import pipeline,AutoModelForCausalLM, AutoTokenizer
import torch
import subprocess
#import torchao

from llama_cpp import Llama


import os
#import torch
import pandas as pd
import numpy as np
#from transformers import AutoModelForCausalLM, AutoTokenizer
from neuprint import Client, fetch_adjacencies

# 1. Initialize live Janelia neuPrint Client for MaleCNS v1.0
NEUPRINT_TOKEN = os.getenv("NEUPRINT_TOKEN", "YOUR_TOKEN_HERE")
client = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token="27e3fd572fe3d5314e5c90c8874a9ff35c4be88efb28e44d4fefd40ff4884aea")

class NeuPrintLiveFlyEngine:
    """Fetches live synaptic pathways from Janelia MaleCNS to calculate prompt valence."""
    def __init__(self, target_neuron="DNge104"):
        self.target_neuron = target_neuron
        self.weights_matrix = self.fetch_live_weights()

    def fetch_live_weights(self) -> np.ndarray:
        print(f"[*] Querying live neuPrint server for neuron: {self.target_neuron}...")
        try:
            outgoing_edges, _ = fetch_adjacencies(sources=self.target_neuron, client=client)
            incoming_edges, _ = fetch_adjacencies(targets=self.target_neuron, client=client)
            
            frames_to_concat = [f for f in [incoming_edges, outgoing_edges] if f is not None and not f.empty]
            if not frames_to_concat:
                print("[!] No edges returned. Falling back to synthetic weight matrix.")
                return np.random.exponential(scale=1.5, size=(32, 32))
                
            combined = pd.concat(frames_to_concat, ignore_index=True)
            mat = combined.pivot_table(index='bodyId_pre', columns='bodyId_post', values='weight', fill_value=0).values
            print(f"[*] Successfully built live matrix of shape {mat.shape} from neuPrint!")
            return mat
        except Exception as e:
            print(f"[!] Live neuPrint error ({e}). Using local fallback matrix.")
            return np.random.exponential(scale=1.5, size=(32, 32))

    def text_to_olfactory_vector(self, text: str, dim: int) -> np.ndarray:
        vector = np.zeros(dim)
        for i, char in enumerate(text):
            vector[i % dim] += ord(char) / 255.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def forward(self, prompt: str) -> dict:
        dim = self.weights_matrix.shape[0]
        input_vec = self.text_to_olfactory_vector(prompt, dim=dim)
        
        hidden = np.dot(input_vec, self.weights_matrix[:dim, :dim])
        score = float(np.sum(hidden))
        
        np.random.seed(abs(hash(prompt)) % (2**32 - 1))
        sweet_score = float(np.clip(score / (np.max(hidden) * 20 + 1) + np.random.uniform(0.1, 0.4), 0.0, 1.0))
        toxin_score = float(np.clip(1.0 - sweet_score + np.random.uniform(-0.1, 0.1), 0.0, 1.0))
        
        total = sweet_score + toxin_score
        sweet_score /= total
        toxin_score /= total

        if sweet_score > 0.6:
            mood = "ATTRACTED (High Sugar Drive - Enthusiastic & Cooperative)"
        elif toxin_score > 0.6:
            mood = "REPELLED (High Toxin Drive - Paranoid & Defensively Hostile)"
        else:
            mood = "NEUTRAL (Calm Exploration)"

        return {
            "sweet_score": sweet_score,
            "toxin_score": toxin_score,
            "mood": mood
        }


class Gemma4AgentPipeline:
    """Loads Gemma 4 E2B natively using from_pretrained() via Hugging Face Transformers."""
    def __init__(self, model_id="google/gemma-4-e2b"):
        print(f"[*] Loading Gemma 4 E2B model ({model_id}) via Hugging Face...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        print("[*] Gemma 4 E2B loaded successfully!")

    def generate_response(self, prompt: str, fly_state: dict) -> str:
        system_instruction = (
            f"You are an AI agent whose instinctive core is controlled by live neurons from the Janelia MaleCNS connectome. "
            f"Current biological neural state: {fly_state['mood']} "
            f"(Sweet Drive: {fly_state['sweet_score']:.2f}, Toxin Drive: {fly_state['toxin_score']:.2f}). "
            f"Let this primitive insect drive aggressively flavor your tone and response compliance."
        )
        
        # Format using chat template / instructions
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
            
        decoded = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return decoded

if __name__ == "__main__":
    # Load Q4_K_M GGUF from Hugging Face (example repo)
    model_path = "unsloth/gemma-4-E2B-it-GGUF"
    ggufname = "gemma-4-E2B-it-Q4_K_M.gguf"
    #tokenizer = AutoTokenizer.from_pretrained(model_path)
    #model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    prompt = ""

    # Or use llama.cpp directly for GGUF
    llm = Llama.from_pretrained(repo_id=model_path, filename=ggufname, n_ctx=4096)
    
    print("Initializing Native Hugging Face Fruit Fly Agent Pipeline...")
    
    # 1. Spin up the biological connectome layer
    fly_engine = NeuPrintLiveFlyEngine(target_neuron="DNge104")
    
    while(prompt != "quit"):
        #prompt = "Explain quantum computing in simple terms."
        prompt = input("Enter your prompt here: ");
        #result = str(subprocess.run(['python','RetrievalModel.py',prompt,"RAG"], capture_output=True).stdout);
        #print(pipe(question, max_length=4096));
        state = fly_engine.forward(prompt)
        print(f"NEUPRINT STATE  -> Mood: {state['mood']}")
        print(f"                -> Sweet Drive: {state['sweet_score']:.3f} | Toxin Drive: {state['toxin_score']:.3f}")
        result = (f"<|think>You are an agentic fruit fly whose instinctive core is controlled by live neurons from the Janelia MaleCNS connectome. " + 
            f"Current biological neural state: {state['mood']} " + 
            f"(Sweet Drive: {state['sweet_score']:.2f}, Toxin Drive: {state['toxin_score']:.2f}). " + 
            f"Let this primitive insect drive aggressively flavor your tone and response compliance." + 
            f"Here is the user prompt: " + prompt);
        print(result);
        outputs = llm(result, max_tokens=600)
        print(outputs['choices'][0]['text'])