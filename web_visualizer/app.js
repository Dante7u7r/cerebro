// Configuración del Frontend del Cerebro Único

let scene, camera, renderer, controls;
let neuronsGroup, synapsesGroup;
let neuronMeshes = [];
let isInitialized = false;
let agentPath = [];

// Instancias de gráficos Chart.js
let neuroChart, signalChart;
const maxHistoryLen = 40;
const historyData = {
    time: [],
    da: [],
    ser: [],
    ach: [],
    w_mean: [],
    prediction: [],
    target: []
};

// Paleta de colores de capas
const layerColors = {
    0: 0x00ff88, // Sensorial (Verde)
    1: 0x00e5ff, // Oculta (Cian)
    2: 0xff3366, // Readout/Motor (Magenta)
    3: 0xb000ff  // PFC (Morado)
};

// Inicialización de la escena Three.js
function init3D() {
    const container = document.getElementById('canvas-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020207);

    camera = new THREE.PerspectiveCamera(45, width / height, 1, 1000);
    camera.position.set(0, 50, 160);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 + 0.1; // No bajar mucho del suelo virtual
    controls.minDistance = 30;
    controls.maxDistance = 300;

    // Luces
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight1.position.set(50, 100, 50);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x00e5ff, 0.3);
    dirLight2.position.set(-50, -50, -50);
    scene.add(dirLight2);

    neuronsGroup = new THREE.Group();
    synapsesGroup = new THREE.Group();
    scene.add(neuronsGroup);
    scene.add(synapsesGroup);

    // Ajustar el tamaño al cambiar la ventana
    window.addEventListener('resize', () => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });

    animate();
}

function animate() {
    requestAnimationFrame(animate);
    if (controls) controls.update();
    if (renderer && scene && camera) renderer.render(scene, camera);
}

// Inicializar Gráficos de Telemetría
function initCharts() {
    const ctxNeuro = document.getElementById('neuroChart').getContext('2d');
    const ctxSignal = document.getElementById('signalChart').getContext('2d');

    Chart.defaults.color = '#8888aa';
    Chart.defaults.font.family = "'Outfit', sans-serif";

    neuroChart = new Chart(ctxNeuro, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Dopamina', data: [], borderColor: '#ffcc00', borderWidth: 2, pointRadius: 0, fill: false, tension: 0.1 },
                { label: 'Serotonina', data: [], borderColor: '#b000ff', borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0.1 },
                { label: 'Acetilcolina', data: [], borderColor: '#00e5ff', borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0.1 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { boxWidth: 12, font: { size: 9 } } } },
            scales: {
                y: { min: -0.05, max: 1.05, grid: { color: 'rgba(255,255,255,0.04)' } },
                x: { grid: { display: false } }
            }
        }
    });

    signalChart = new Chart(ctxSignal, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Peso Medio (w)', data: [], borderColor: '#00ff88', borderWidth: 2, pointRadius: 0, fill: false, tension: 0.1 },
                { label: 'Salida Motor', data: [], borderColor: '#ff3366', borderWidth: 1.5, borderDash: [2, 2], pointRadius: 0, fill: false, tension: 0.1 },
                { label: 'Objetivo', data: [], borderColor: 'rgba(255,255,255,0.3)', borderWidth: 1, pointRadius: 0, fill: false, tension: 0.1 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { boxWidth: 12, font: { size: 9 } } } },
            scales: {
                y: { min: -1.2, max: 1.2, grid: { color: 'rgba(255,255,255,0.04)' } },
                x: { grid: { display: false } }
            }
        }
    });
}

// Re-crear u optimizar el renderizado de la red neuronal
function updateNetwork3D(data) {
    // 1. Inicialización en el primer fetch
    if (!isInitialized) {
        // Limpiar grupo
        while(neuronsGroup.children.length > 0) {
            neuronsGroup.remove(neuronsGroup.children[0]);
        }
        neuronMeshes = [];

        // Crear geometría de neurona común
        const sphereGeo = new THREE.SphereGeometry(2.2, 16, 16);

        for (let i = 0; i < data.neurons.length; i++) {
            const neuron = data.neurons[i];
            const color = layerColors[neuron.layer] || 0xffffff;
            
            // Material básico con emisivo para simular brillo
            const mat = new THREE.MeshStandardMaterial({
                color: color,
                emissive: color,
                emissiveIntensity: 0.15,
                roughness: 0.4,
                metalness: 0.1
            });

            const mesh = new THREE.Mesh(sphereGeo, mat);
            mesh.position.set(neuron.x, neuron.y, neuron.z);
            
            // Guardar metadatos útiles
            mesh.userData = { 
                id: i, 
                layer: neuron.layer, 
                baseColor: new THREE.Color(color) 
            };

            neuronsGroup.add(mesh);
            neuronMeshes.push(mesh);
        }
        isInitialized = true;
    }

    // 2. Actualizar estado de disparo de neuronas (glow temporal al disparar)
    for (let i = 0; i < neuronMeshes.length; i++) {
        const mesh = neuronMeshes[i];
        const state = data.neurons[i];
        
        // Si la neurona está disparando (firing_rate alto en la ventana reciente)
        if (state.firing > 5.0) {
            // Aumentar tamaño y brillar en blanco
            mesh.scale.set(1.6, 1.6, 1.6);
            mesh.material.emissive.setHex(0xffffff);
            mesh.material.emissiveIntensity = 1.0;
        } else {
            // Regresar al estado base
            mesh.scale.set(1.0, 1.0, 1.0);
            mesh.material.emissive.copy(mesh.userData.baseColor);
            mesh.material.emissiveIntensity = 0.15 + 0.3 * (state.energy / 1.0); // Brillar más si tiene más energía
        }
    }

    // 3. Dibujar sinapsis
    // Primero, remover sinapsis viejas del grupo
    while (synapsesGroup.children.length > 0) {
        const child = synapsesGroup.children[0];
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
        synapsesGroup.remove(child);
    }

    // Dibujar sinapsis activas (w > 0.08)
    const points = [];
    const colors = [];
    const synapses = data.synapses;

    for (let s = 0; s < synapses.length; s++) {
        const syn = synapses[s];
        if (syn.w > 0.08) {
            const preMesh = neuronMeshes[syn.pre];
            const postMesh = neuronMeshes[syn.post];
            if (preMesh && postMesh) {
                // Crear línea para cada conexión activa
                const p1 = preMesh.position;
                const p2 = postMesh.position;

                // Color dinámico según la capa de origen o si es inhibitoria
                let lineColor = new THREE.Color(0x7777aa);
                if (!syn.exc) {
                    lineColor = new THREE.Color(0xff3366); // Inhibitoria
                } else {
                    lineColor = new THREE.Color(layerColors[preMesh.userData.layer]);
                }

                const lineGeo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
                const lineMat = new THREE.LineBasicMaterial({
                    color: lineColor,
                    transparent: true,
                    opacity: Math.min(syn.w * 0.45, 0.65),
                    linewidth: 1 // Nota: WebGL en la mayoría de plataformas no soporta linewidth > 1
                });

                const line = new THREE.Line(lineGeo, lineMat);
                synapsesGroup.add(line);
            }
        }
    }
}

// Actualizar Dashboard y Telemetría General
function updateDashboard(data) {
    // 1. Textos e indicadores generales
    document.getElementById('val-synapses').innerText = data.synapses_active;
    document.getElementById('val-pruned').innerText = data.synapses_pruned;
    document.getElementById('val-created').innerText = data.synapses_created;
    document.getElementById('val-energy').innerText = (data.energy_mean * 100).toFixed(1) + '%';
    document.getElementById('val-time').innerText = (data.time / 1000).toFixed(1);
    document.getElementById('val-steps').innerText = data.step;

    // Actualizar badge de estado
    const stateBadge = document.getElementById('sim-state-badge');
    stateBadge.innerText = data.state;
    stateBadge.className = 'state-badge'; // reset
    if (data.state === 'AWAKE') {
        stateBadge.classList.add('state-awake');
    } else if (data.state === 'SLOW_WAVE_SLEEP') {
        stateBadge.classList.add('state-sws');
    } else {
        stateBadge.classList.add('state-rem');
    }

    // 2. Acumular histórico para gráficos
    const timeLabel = (data.time / 1000).toFixed(1) + 's';
    
    historyData.time.push(timeLabel);
    historyData.da.push(data.da);
    historyData.ser.push(data.ser);
    historyData.ach.push(data.ach);
    historyData.w_mean.push(data.w_mean);
    historyData.prediction.push(data.prediction);
    historyData.target.push(data.target);

    if (historyData.time.length > maxHistoryLen) {
        historyData.time.shift();
        historyData.da.shift();
        historyData.ser.shift();
        historyData.ach.shift();
        historyData.w_mean.shift();
        historyData.prediction.shift();
        historyData.target.shift();
    }

    // 3. Actualizar gráficos de Chart.js
    if (neuroChart) {
        neuroChart.data.labels = historyData.time;
        neuroChart.data.datasets[0].data = historyData.da;
        neuroChart.data.datasets[1].data = historyData.ser;
        neuroChart.data.datasets[2].data = historyData.ach;
        neuroChart.update('none'); // Update sin animación para velocidad
    }

    if (signalChart) {
        signalChart.data.labels = historyData.time;
        signalChart.data.datasets[0].data = historyData.w_mean;
        signalChart.data.datasets[1].data = historyData.prediction;
        signalChart.data.datasets[2].data = historyData.target;
        signalChart.update('none');
    }
}

// Renderizar Arena Toroidal 2D
function updateArena2D(data) {
    const canvas = document.getElementById('arenaCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    // Si la simulación reinició, limpiar el rastro
    if (data.step <= 1) {
        agentPath = [];
    }
    
    // Añadir posición actual al rastro
    agentPath.push({ x: data.agent_x, y: data.agent_y });
    if (agentPath.length > 80) {
        agentPath.shift();
    }
    
    // Actualizar telemetría textual
    document.getElementById('val-pos').innerText = `(${data.agent_x.toFixed(1)}, ${data.agent_y.toFixed(1)})`;
    document.getElementById('val-meals').innerText = data.meals_eaten;
    
    // Limpiar canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 1. Dibujar rejilla toroidal (cada 10 unidades de arena, límite es -40 a 40)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.lineWidth = 1;
    for (let gx = -30; gx <= 30; gx += 10) {
        const cx = ((gx + 40) / 80) * canvas.width;
        ctx.beginPath();
        ctx.moveTo(cx, 0);
        ctx.lineTo(cx, canvas.height);
        ctx.stroke();
    }
    for (let gy = -30; gy <= 30; gy += 10) {
        const cy = ((40 - gy) / 80) * canvas.height;
        ctx.beginPath();
        ctx.moveTo(0, cy);
        ctx.lineTo(canvas.width, cy);
        ctx.stroke();
    }
    
    // Ejes centrales (cruces)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.07)';
    ctx.beginPath();
    ctx.moveTo(canvas.width / 2, 0);
    ctx.lineTo(canvas.width / 2, canvas.height);
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
    
    // 2. Dibujar rastro difuminado (trayectoria)
    for (let i = 1; i < agentPath.length; i++) {
        const pt = agentPath[i];
        const prev = agentPath[i - 1];
        
        // Evitar líneas cruzando todo el mapa por salto toroidal
        const dx = Math.abs(pt.x - prev.x);
        const dy = Math.abs(pt.y - prev.y);
        if (dx > 40 || dy > 40) {
            continue;
        }
        
        const cx1 = ((prev.x + 40) / 80) * canvas.width;
        const cy1 = ((40 - prev.y) / 80) * canvas.height;
        const cx2 = ((pt.x + 40) / 80) * canvas.width;
        const cy2 = ((40 - pt.y) / 80) * canvas.height;
        
        const alpha = (i / agentPath.length) * 0.45;
        ctx.strokeStyle = `rgba(0, 229, 255, ${alpha})`;
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        ctx.moveTo(cx1, cy1);
        ctx.lineTo(cx2, cy2);
        ctx.stroke();
    }
    
    // 3. Dibujar comida (glowing neon green)
    const fx = ((data.food_x + 40) / 80) * canvas.width;
    const fy = ((40 - data.food_y) / 80) * canvas.height;
    
    ctx.save();
    ctx.shadowBlur = 15;
    ctx.shadowColor = '#00ff88';
    ctx.fillStyle = '#00ff88';
    ctx.beginPath();
    // Oscilación de radio para efecto latido
    const pulse = 6 + 2 * Math.sin(Date.now() / 150);
    ctx.arc(fx, fy, pulse, 0, 2 * Math.PI);
    ctx.fill();
    ctx.restore();
    
    // 4. Dibujar agente (glowing cyan circle + vector de dirección)
    const ax = ((data.agent_x + 40) / 80) * canvas.width;
    const ay = ((40 - data.agent_y) / 80) * canvas.height;
    
    ctx.save();
    ctx.shadowBlur = 12;
    const isBlind = data.modo_ciego === true;
    ctx.shadowColor = isBlind ? '#ffaa00' : '#00e5ff';
    ctx.fillStyle = isBlind ? '#ffaa00' : '#00e5ff';
    ctx.beginPath();
    ctx.arc(ax, ay, 6, 0, 2 * Math.PI);
    ctx.fill();
    
    // Vector dirección (línea blanca)
    const hlen = 14;
    const hx = ax + hlen * Math.cos(data.agent_theta);
    const hy = ay - hlen * Math.sin(data.agent_theta); // resta porque canvas Y aumenta hacia abajo
    
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(hx, hy);
    ctx.stroke();
    ctx.restore();
}

// Ciclo de Fetching de datos
async function fetchData() {
    try {
        const response = await fetch('/sim_state.json');
        if (!response.ok) throw new Error('Network response error');
        const data = await response.json();
        
        // Actualizar vistas
        updateNetwork3D(data);
        updateDashboard(data);
        updateArena2D(data);
    } catch (err) {
        console.warn('Esperando datos de simulación activa...', err.message);
    }
}

// Inicialización de la aplicación
document.addEventListener('DOMContentLoaded', () => {
    init3D();
    initCharts();
    
    // Polling cada 200ms
    setInterval(fetchData, 200);
});
