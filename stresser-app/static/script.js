let debounceTimer = null;

function onSliderChange(value) {
    updateDisplay(parseInt(value));

    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        setStress(parseInt(value));
    }, 200);
}

function updateDisplay(level) {
    const display = document.getElementById('stressDisplay');
    display.textContent = level + '%';

    display.className = 'stress-display';
    if (level <= 30) display.classList.add('low');
    else if (level <= 70) display.classList.add('medium');
    else display.classList.add('high');
}

async function setStress(level) {
    const status = document.getElementById('status');
    try {
        const res = await fetch('/api/stress', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ level: level })
        });

        const data = await res.json();

        if (res.ok) {
            document.getElementById('cpuWorkers').textContent = data.cpu_workers;
            document.getElementById('currentLevel').textContent = data.stress_level;
            status.textContent = data.message;
            status.className = 'status ok';
        } else {
            status.textContent = (data.error || 'Erro desconhecido');
            status.className = 'status error';
        }
    } catch (err) {
        status.textContent = ' Falha na conexao';
        status.className = 'status error';
    }
}

async function loadState() {
    try {
        const res = await fetch('/api/state');
        const data = await res.json();

        document.getElementById('stressSlider').value = data.stress_level;
        document.getElementById('cpuWorkers').textContent = data.cpu_workers;
        document.getElementById('currentLevel').textContent = data.stress_level;
        updateDisplay(data.stress_level);

        document.getElementById('status').textContent = 'Conectado';
        document.getElementById('status').className = 'status ok';
    } catch (err) {
        document.getElementById('status').textContent = 'Falha ao conectar';
        document.getElementById('status').className = 'status error';
    }
}

loadState();
setInterval(loadState, 2000);
