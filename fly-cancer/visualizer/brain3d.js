/* Flyputer-style Three.js brain viewer for cancer scenes. */
(function () {
  const IN = new THREE.Color(0xe07a3d);
  const DN = new THREE.Color(0x3db8a0);
  const GHOST = 0x3a4a44;

  let renderer, scene, camera, controls, points, glowAttr, spikes, ptr;
  let lines = null;
  let ghostPts = null;
  let animId = 0;
  let DUR = 120;
  let tNow = 0;
  let holdT = 0;
  let clock = null;
  let heroList = [];
  let currentData = null;

  function el(id) {
    return document.getElementById(id);
  }

  function clearScene() {
    if (animId) cancelAnimationFrame(animId);
    animId = 0;
    const host = el("brain3d");
    if (renderer) {
      renderer.dispose();
      if (renderer.domElement && renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    }
    host.innerHTML = "";
    renderer = scene = camera = controls = points = glowAttr = null;
    lines = ghostPts = null;
    spikes = null;
    heroList = [];
    currentData = null;
  }

  function boot(D) {
    clearScene();
    currentData = D;
    const host = el("brain3d");
    const N = D.neurons.length;
    el("brain3dSub").textContent =
      `${D.n_input} input → ${D.n_downstream} downstream · ${N} firing · ${D.label}`;

    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x0e1412, 0.0024);
    const w = host.clientWidth || 640;
    const h = host.clientHeight || 420;
    camera = new THREE.PerspectiveCamera(55, w / h, 1, 4000);
    camera.position.set(0, 30, 320);
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(w, h);
    host.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = el("spin3d").checked;
    controls.autoRotateSpeed = 0.55;
    controls.addEventListener("start", () => {
      if (controls.autoRotate) {
        controls.autoRotate = false;
        el("spin3d").checked = false;
      }
    });

    const pos = new Float32Array(N * 3);
    const col = new Float32Array(N * 3);
    const glow = new Float32Array(N);
    spikes = [];
    for (let i = 0; i < N; i++) {
      const n = D.neurons[i];
      pos[i * 3] = n.x;
      pos[i * 3 + 1] = n.y;
      pos[i * 3 + 2] = n.z;
      const c = n.role === "input" ? IN : DN;
      col[i * 3] = c.r;
      col[i * 3 + 1] = c.g;
      col[i * 3 + 2] = c.b;
      spikes.push(n.t || []);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    glowAttr = new THREE.BufferAttribute(glow, 1);
    geo.setAttribute("aGlow", glowAttr);

    const mat = new THREE.ShaderMaterial({
      uniforms: { uSize: { value: 26.0 } },
      transparent: true,
      depthTest: false,
      blending: THREE.AdditiveBlending,
      vertexColors: true,
      vertexShader: [
        "attribute float aGlow; varying float vGlow; varying vec3 vColor;",
        "uniform float uSize;",
        "void main(){",
        "  vGlow = aGlow; vColor = color;",
        "  vec4 mv = modelViewMatrix * vec4(position, 1.0);",
        "  gl_PointSize = uSize * (0.45 + 2.4*aGlow) * (300.0 / -mv.z);",
        "  gl_Position = projectionMatrix * mv;",
        "}",
      ].join("\n"),
      fragmentShader: [
        "varying float vGlow; varying vec3 vColor;",
        "void main(){",
        "  float d = length(gl_PointCoord - vec2(0.5));",
        "  if (d > 0.5) discard;",
        "  float a = smoothstep(0.5, 0.0, d);",
        "  vec3 c = vColor * (0.30 + 1.9*vGlow);",
        "  gl_FragColor = vec4(c, a * (0.32 + 0.68*vGlow));",
        "}",
      ].join("\n"),
    });
    points = new THREE.Points(geo, mat);
    scene.add(points);

    if (D.edges && D.edges.length) {
      const ep = new Float32Array(D.edges.length * 6);
      for (let e = 0; e < D.edges.length; e++) {
        const a = D.edges[e][0];
        const b = D.edges[e][1];
        ep[e * 6] = pos[a * 3];
        ep[e * 6 + 1] = pos[a * 3 + 1];
        ep[e * 6 + 2] = pos[a * 3 + 2];
        ep[e * 6 + 3] = pos[b * 3];
        ep[e * 6 + 4] = pos[b * 3 + 1];
        ep[e * 6 + 5] = pos[b * 3 + 2];
      }
      const lg = new THREE.BufferGeometry();
      lg.setAttribute("position", new THREE.BufferAttribute(ep, 3));
      lines = new THREE.LineSegments(
        lg,
        new THREE.LineBasicMaterial({
          color: 0x3db8a0,
          transparent: true,
          opacity: 0.06,
          depthTest: false,
          blending: THREE.AdditiveBlending,
        })
      );
      lines.visible = el("edges3d").checked;
      scene.add(lines);
    }

    if (D.ghost && D.ghost.length) {
      const gg = new THREE.BufferGeometry();
      gg.setAttribute(
        "position",
        new THREE.BufferAttribute(new Float32Array(D.ghost), 3)
      );
      ghostPts = new THREE.Points(
        gg,
        new THREE.PointsMaterial({
          color: GHOST,
          size: 1.5,
          sizeAttenuation: true,
          transparent: true,
          opacity: 0.22,
          depthTest: false,
          blending: THREE.AdditiveBlending,
        })
      );
      ghostPts.visible = el("ghost3d").checked;
      scene.add(ghostPts);
    }

    DUR = D.dur_ms || 120;
    ptr = new Int32Array(N);
    tNow = 0;
    holdT = 0;
    clock = new THREE.Clock();

    function frame() {
      animId = requestAnimationFrame(frame);
      const dt = Math.min(clock.getDelta(), 0.05);
      let looped = false;
      if (tNow < DUR) {
        tNow += dt * (DUR / 8.0);
        if (tNow > DUR) tNow = DUR;
      } else {
        holdT += dt;
        if (holdT > 0.6) {
          holdT = 0;
          tNow = 0;
          ptr.fill(0);
          looped = true;
        }
      }
      for (let i = 0; i < N; i++) {
        glow[i] *= 0.9;
        const s = spikes[i];
        while (ptr[i] < s.length && s[ptr[i]] <= tNow) {
          glow[i] = 1.0;
          ptr[i]++;
        }
      }
      geo.attributes.aGlow.needsUpdate = true;
      controls.update();
      renderer.render(scene, camera);
      if (looped) {
        /* keep looping */
      }
    }
    frame();

    const tip = el("tip3d");
    const ray = new THREE.Raycaster();
    ray.params.Points.threshold = 4;
    const mouse = new THREE.Vector2();
    renderer.domElement.addEventListener("pointermove", (ev) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
      ray.setFromCamera(mouse, camera);
      const hit = ray.intersectObject(points)[0];
      if (hit) {
        const n = D.neurons[hit.index];
        tip.style.display = "block";
        tip.style.left = ev.clientX + "px";
        tip.style.top = ev.clientY + "px";
        tip.textContent = `${n.type} — ${n.role}`;
      } else tip.style.display = "none";
    });
  }

  window.CancerBrain3D = {
    load: boot,
    clear: clearScene,
    setGhost(v) {
      if (ghostPts) ghostPts.visible = v;
    },
    setEdges(v) {
      if (lines) lines.visible = v;
    },
    setSpin(v) {
      if (controls) controls.autoRotate = v;
    },
    resize() {
      if (!renderer || !camera) return;
      const host = el("brain3d");
      const w = host.clientWidth || 640;
      const h = host.clientHeight || 420;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    },
  };

  el("ghost3d").onchange = (e) => window.CancerBrain3D.setGhost(e.target.checked);
  el("edges3d").onchange = (e) => window.CancerBrain3D.setEdges(e.target.checked);
  el("spin3d").onchange = (e) => window.CancerBrain3D.setSpin(e.target.checked);
  addEventListener("resize", () => window.CancerBrain3D.resize());
})();
