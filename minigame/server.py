import asyncio
import json
import random
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from llama_cpp import Llama
from fly_agent import FlyAgent

app = FastAPI()

class WorldDirector:
    """Gemma 4 World Director that generates 3D environment styling AND generative scene objects."""
    def __init__(self, llm_instance: Llama):
        self.llm = llm_instance

    def generate_world_state(self, scenario: str) -> dict:
        prompt = (
            "You are an AI World Director for a 3D fruit fly simulation. "
            "Return a JSON object with: "
            "\"background\": hex color string, "
            "\"light_color\": hex color string, "
            "\"light_intensity\": float (0.5 to 2.0), "
            "\"mood_summary\": 4-word description, "
            "\"objects\": an array of 1 to 3 objects to spawn in the 3D room. Each object must have: "
            "\"type\" (string like \"sugar_cube\", \"toxic_barrel\", or \"wind_fan\"), "
            "\"x\": float between -5 and 5, "
            "\"y\": 0.0, "
            "\"z\": float between -5 and 5, "
            "\"color\": hex color string."
        )
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Scenario: {scenario}"}
        ]

        try:
            response = self.llm.create_chat_completion(
                messages=messages, max_tokens=250, temperature=0.7,
                response_format={"type": "json_object"}
            )
            return json.loads(response['choices'][0]['message']['content'])
        except Exception:
            # Fallback world objects
            return {
                "background": "#0b0b10", "light_color": "#ffffff", "light_intensity": 1.0,
                "mood_summary": "Neutral Laboratory Chamber",
                "objects": [{"type": "sugar_cube", "x": 2.0, "y": 0.0, "z": 2.0, "color": "#00ff88"}]
            }


class MultiUserSimulationManager:
    def __init__(self):
        print("[*] Initializing Gemma 4 model...")
        self.llm = Llama.from_pretrained(
            repo_id="unsloth/gemma-4-E2B-it-GGUF",
            filename="gemma-4-E2B-it-Q4_K_M.gguf",
            n_ctx=4096
        )
        
        self.director = WorldDirector(self.llm)
        self.colors = ["#00ffff", "#00ff00", "#ff00ff", "#ffff00", "#ff6600", "#0099ff"]
        self.agent_count = 0
        self.agents = []

        self.current_scenario = "neutral_lab"
        self.world_state = self.director.generate_world_state(self.current_scenario)
        self.scenario_queue = []
        self.colony_chatter = []
        self.active_connections: list[WebSocket] = []

        for name in ["Alpha", "Beta", "Gamma", "Delta"]:
            self.add_fly(name)

    def add_fly(self, name: str = None):
        self.agent_count += 1
        fly_name = name or f"Fly_{self.agent_count}"
        color = self.colors[(self.agent_count - 1) % len(self.colors)]
        
        self.agents.append({
            "agent": FlyAgent(fly_name, color, self.llm),
            "x": random.uniform(-4.0, 4.0), "y": 0.5, "z": random.uniform(-4.0, 4.0),
            "color": color
        })

    def remove_fly(self):
        if self.agents:
            self.agents.pop()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data)
        for conn in self.active_connections:
            try:
                await conn.send_text(msg)
            except Exception:
                pass

manager = MultiUserSimulationManager()

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Janelia MaleCNS Floating Stats Simulator</title>
        <style>
            body { margin: 0; background: #0b0b10; color: #fff; font-family: monospace; overflow: hidden; }
            #ui { position: absolute; top: 15px; left: 15px; z-index: 10; background: rgba(0,0,0,0.85); padding: 15px; border-radius: 8px; border: 1px solid #444; width: 340px; }
            input, button { background: #222; color: #0ff; border: 1px solid #0ff; padding: 6px 12px; margin-top: 5px; cursor: pointer; width: 100%; box-sizing: border-box; }
            .btn-group { display: flex; gap: 5px; margin-top: 5px; }
            .btn-group button { width: 50%; }
            #director-note { font-size: 10px; color: #ff00ff; margin-top: 5px; font-style: italic; }
            #feed { margin-top: 8px; font-size: 11px; height: 120px; overflow-y: auto; border: 1px solid #333; padding: 5px; background: #111; line-height: 1.3; }
            
            /* Floating 3D Label Styling */
            .fly-label {
                background: rgba(0, 0, 0, 0.75);
                color: #fff;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 10px;
                font-family: monospace;
                border: 1px solid #555;
                white-space: nowrap;
                pointer-events: none;
                text-align: center;
                transform: translateY(-30px);
            }
            .fly-label b { color: #0ffff0; }
        </style>
        <!-- Three.js & CSS2DRenderer CDNs -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/renderers/CSS2DRenderer.js"></script>
    </head>
    <body>
        <div id="ui">
            <h3>GENERATIVE FLY COLONY</h3>
            <div id="status">Connecting...</div>
            <div id="director-note">Director AI: Standby...</div>
            <br>
            <input type="text" id="scenarioInput" placeholder="Inject scenario (e.g. sugar_spill)">
            <button onclick="sendScenario()">Inject Scenario</button>
            <div class="btn-group">
                <button style="border-color:#0f0; color:#0f0" onclick="addFly()">+ Add Fly</button>
                <button style="border-color:#f00; color:#f00" onclick="removeFly()">- Delete Fly</button>
            </div>
            <div id="feed">Waiting for simulation telemetry...</div>
        </div>
        <script>
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0b0b10);
            
            const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 10, 16);
            camera.lookAt(0, 0, 0);

            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.body.appendChild(renderer.domElement);

            // Setup CSS2D Renderer for floating HTML labels
            const labelRenderer = new THREE.CSS2DRenderer();
            labelRenderer.setSize(window.innerWidth, window.innerHeight);
            labelRenderer.domElement.style.position = 'absolute';
            labelRenderer.domElement.style.top = '0px';
            labelRenderer.domElement.style.pointerEvents = 'none';
            document.body.appendChild(labelRenderer.domElement);

            scene.add(new THREE.AmbientLight(0xffffff, 0.6));
            const pointLight = new THREE.PointLight(0xffffff, 1, 50);
            pointLight.position.set(0, 15, 0);
            scene.add(pointLight);
            scene.add(new THREE.GridHelper(20, 20, 0x444444, 0x222222));

            const flyMeshes = {};
            const flyLabels = {};
            let worldObjectMeshes = [];

            const ws = new WebSocket("ws://" + window.location.host + "/ws");

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                document.getElementById("status").innerHTML = `<b>Scenario:</b> ${data.scenario.toUpperCase()}<br><b>Flies Active:</b> ${data.agents.length}`;
                
                if (data.world_state) {
                    document.getElementById("director-note").innerText = `Director AI: "${data.world_state.mood_summary}"`;
                    scene.background.set(data.world_state.background);
                    pointLight.color.set(data.world_state.light_color);
                    pointLight.intensity = data.world_state.light_intensity;

                    // Render World Objects
                    worldObjectMeshes.forEach(m => scene.remove(m));
                    worldObjectMeshes = [];

                    if (data.world_state.objects) {
                        data.world_state.objects.forEach(obj => {
                            const geom = new THREE.BoxGeometry(1.0, 1.0, 1.0);
                            const mat = new THREE.MeshStandardMaterial({ color: obj.color || 0x00ff00, emissive: obj.color || 0x00ff00, emissiveIntensity: 0.4 });
                            const mesh = new THREE.Mesh(geom, mat);
                            mesh.position.set(obj.x, obj.y + 0.5, obj.z);
                            scene.add(mesh);
                            worldObjectMeshes.push(mesh);
                        });
                    }
                }

                // Clean up deleted flies & labels
                const currentIds = data.agents.map(a => a.id);
                Object.keys(flyMeshes).forEach(id => {
                    if (!currentIds.includes(id)) {
                        scene.remove(flyMeshes[id]);
                        if (flyLabels[id]) {
                            scene.remove(flyLabels[id]);
                            delete flyLabels[id];
                        }
                        delete flyMeshes[id];
                    }
                });

                // Update or Create Fly Meshes & Floating Labels
                data.agents.forEach(agent => {
                    if (!flyMeshes[agent.id]) {
                        const hexColor = agent.color.startsWith("#") ? parseInt(agent.color.replace("#","0x")) : 0x00ffff;
                        
                        // 3D Sphere Mesh
                        const mesh = new THREE.Mesh(
                            new THREE.SphereGeometry(0.5, 32, 32), 
                            new THREE.MeshStandardMaterial({ color: hexColor, roughness: 0.2 })
                        );
                        scene.add(mesh);
                        flyMeshes[agent.id] = mesh;

                        // Floating HTML Stats Label
                        const div = document.createElement('div');
                        div.className = 'fly-label';
                        div.style.borderColor = agent.color;
                        const label = new THREE.CSS2DObject(div);
                        label.position.set(0, 0.8, 0);
                        mesh.add(label);
                        flyLabels[agent.id] = label;
                    }

                    // Update position
                    flyMeshes[agent.id].position.set(agent.x, agent.y, agent.z);

                    // Update label content dynamically with Name, Hunger, and Mood
                    const labelDiv = flyLabels[agent.id].element;
                    const hungerVal = agent.hunger ? agent.hunger.toFixed(0) : "0";
                    labelDiv.innerHTML = `<b>${agent.id}</b><br>Hunger: ${hungerVal}%<br><span style="color:#ffcc00">${agent.mood}</span>`;
                });

                const feed = document.getElementById("feed");
                if (data.chatter && data.chatter.length > 0) {
                    feed.innerHTML = data.chatter.map(c => `<b>[${c.agent_id}]:</b> ${c.text}`).join("<br>");
                    feed.scrollTop = feed.scrollHeight;
                }
            };

            function sendScenario() {
                const val = document.getElementById("scenarioInput").value;
                if(val) {
                    ws.send(JSON.stringify({ action: "scenario", scenario: val }));
                    document.getElementById("scenarioInput").value = "";
                }
            }

            function addFly() { ws.send(JSON.stringify({ action: "add_fly" })); }
            function removeFly() { ws.send(JSON.stringify({ action: "remove_fly" })); }

            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
                labelRenderer.setSize(window.innerWidth, window.innerHeight);
            });

            function animate() {
                requestAnimationFrame(animate);
                renderer.render(scene, camera);
                labelRenderer.render(scene, camera); // Render floating 2D DOM labels
            }
            animate();
        </script>
    </body>
    </html>
    """

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
                payload = json.loads(data)
                action = payload.get("action")
                
                if action == "scenario":
                    new_scen = payload["scenario"].strip().lower()
                    manager.scenario_queue.append(new_scen)
                    manager.world_state = manager.director.generate_world_state(new_scen)
                elif action == "add_fly":
                    manager.add_fly()
                elif action == "remove_fly":
                    manager.remove_fly()

            except asyncio.TimeoutError:
                pass

            if manager.scenario_queue:
                manager.current_scenario = manager.scenario_queue.pop(0)

            objects = manager.world_state.get("objects", [])
            agent_payloads = []
            
            for item in list(manager.agents):
                agent = item["agent"]
                
                # Step agent with object coordinates and fly position
                result = agent.step(manager.current_scenario, manager.colony_chatter, objects, (item["x"], item["y"], item["z"]))
                manager.colony_chatter.append(result)

                item["x"] += random.uniform(-0.15, 0.15)
                item["z"] += random.uniform(-0.15, 0.15)
                item["x"] = max(-8.0, min(8.0, item["x"]))
                item["z"] = max(-8.0, min(8.0, item["z"]))

                agent_payloads.append({
                    "id": agent.agent_id,
                    "color": item["color"],
                    "x": item["x"],
                    "y": item["y"],
                    "z": item["z"],
                    "hunger": result["hunger"],
                    "mood": result["mood"],
                    "text": result["text"]
                })

            await manager.broadcast({
                "scenario": manager.current_scenario,
                "world_state": manager.world_state,
                "agents": agent_payloads,
                "chatter": manager.colony_chatter[-6:]
            })

            await asyncio.sleep(0.8)

    except WebSocketDisconnect:
        manager.disconnect(websocket)