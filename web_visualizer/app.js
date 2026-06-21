// Configuración del Frontend del Cerebro Único

let scene, camera, renderer, controls;
let neuronsGroup, synapsesGroup;
let neuronMeshes = [];
let isInitialized = false;

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

// Ciclo de Fetching de datos
async function fetchData() {
    try {
        const response = await fetch('/sim_state.json');
        if (!response.ok) throw new Error('Network response error');
        const data = await response.json();
        
        // Actualizar vistas
        updateNetwork3D(data);
        updateDashboard(data);
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
