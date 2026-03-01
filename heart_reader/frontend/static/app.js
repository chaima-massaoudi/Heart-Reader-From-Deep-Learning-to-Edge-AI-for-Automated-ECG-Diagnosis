/* ─────────────────────────────────────────────────────────────────
   Heart Reader — Frontend JavaScript
   ───────────────────────────────────────────────────────────────── */

const API = '';
const ECG_COLOR = '#00df80';
const LEAD_COLORS = [
    '#00df80','#3b82f6','#e74c5e','#f59e0b',
    '#a855f7','#06b6d4','#f97316','#ec4899',
    '#14b8a6','#8b5cf6','#64748b','#84cc16',
];
const CLASS_COLORS = {
    NORM: '#22c55e', MI: '#ef4444', STTC: '#f59e0b', CD: '#3b82f6', HYP: '#a855f7'
};

const LEAD_GROUPS = {
    all:   [0,1,2,3,4,5,6,7,8,9,10,11],
    limb:  [0,1,2,3,4,5],
    chest: [6,7,8,9,10,11],
};

let currentSignal = null;
let leadNames = [];
let ecgChartInstances = [];
let probChartInstance = null;

/* ── Init ─────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
    loadModelInfo();
    setupUpload();
    setupLeadSelector();
    setupSaveButtons();
    document.getElementById('btnRandomSample').addEventListener('click', loadRandomSample);
});

/* ── Model Info ───────────────────────────────────────── */
async function loadModelInfo() {
    try {
        const resp = await fetch(`${API}/api/model-info`);
        const info = await resp.json();
        leadNames = info.lead_names || [];

        const body = document.getElementById('modelInfoBody');
        const params = info.num_parameters;
        const paramsStr = params > 1e6 ? (params/1e6).toFixed(2)+'M' : (params/1e3).toFixed(1)+'K';

        let html = `
            <div class="info-row"><span class="label">Architecture</span><span class="value">Fusion InceptionTime1D</span></div>
            <div class="info-row"><span class="label">Parameters</span><span class="value">${paramsStr}</span></div>
            <div class="info-row"><span class="label">Input</span><span class="value">${info.input_shape.leads}-lead, ${info.input_shape.duration_sec}s @ ${info.input_shape.sampling_rate}Hz</span></div>
            <div class="info-row"><span class="label">Features</span><span class="value">${info.num_features > 0 ? info.num_features+' PTB-XL+' : 'Signal only'}</span></div>
            <div class="info-row"><span class="label">Classes</span><span class="value">${info.classes.join(', ')}</span></div>
        `;

        if (info.performance) {
            const p = info.performance;
            if (p.test_macro_auc) html += `<div class="info-row"><span class="label">Test AUC</span><span class="value" style="color:var(--ecg-green)">${p.test_macro_auc.toFixed(4)}</span></div>`;
            if (p.test_f1)       html += `<div class="info-row"><span class="label">Test F1</span><span class="value">${p.test_f1.toFixed(4)}</span></div>`;
        }
        body.innerHTML = html;

        if (info.performance) showPerformanceCard(info.performance);

        document.getElementById('sampleCount').textContent = '';
        const hResp = await fetch(`${API}/api/health`);
        const hData = await hResp.json();
        if (hData.test_samples_available)
            document.getElementById('sampleCount').textContent = `${hData.test_samples_available} test samples`;

        document.getElementById('statusBadge').innerHTML = '<i class="bi bi-circle-fill me-1" style="font-size:0.5rem;"></i> Online';
        document.getElementById('statusBadge').className = 'badge bg-success';

    } catch(err) {
        console.error('Model info error:', err);
        document.getElementById('statusBadge').innerHTML = '<i class="bi bi-circle-fill me-1" style="font-size:0.5rem;"></i> Offline';
        document.getElementById('statusBadge').className = 'badge bg-danger';
    }
}

function showPerformanceCard(perf) {
    const card = document.getElementById('perfCard');
    const content = document.getElementById('perfContent');
    card.style.display = 'block';

    let html = '<div class="row g-2 mb-3">';
    if (perf.test_macro_auc !== undefined)
        html += `<div class="col-4"><div class="perf-metric"><div class="number">${perf.test_macro_auc.toFixed(3)}</div><div class="desc">Macro AUC</div></div></div>`;
    if (perf.test_f1 !== undefined)
        html += `<div class="col-4"><div class="perf-metric"><div class="number">${perf.test_f1.toFixed(3)}</div><div class="desc">Macro F1</div></div></div>`;
    // Accuracy if available
    if (perf.test_accuracy !== undefined)
        html += `<div class="col-4"><div class="perf-metric"><div class="number">${perf.test_accuracy.toFixed(3)}</div><div class="desc">Accuracy</div></div></div>`;
    else
        html += `<div class="col-4"><div class="perf-metric"><div class="number">5</div><div class="desc">Classes</div></div></div>`;
    html += '</div>';

    if (perf.per_class_auc) {
        html += '<div style="font-size:0.7rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Per-Class AUC</div>';
        for (const [cls, val] of Object.entries(perf.per_class_auc)) {
            const pct = (val * 100).toFixed(1);
            html += `<div class="auc-row">
                <span class="cls">${cls}</span>
                <div class="track"><div class="fill" style="width:${pct}%;"></div></div>
                <span class="val">${val.toFixed(3)}</span>
            </div>`;
        }
    }
    content.innerHTML = html;
}

/* ── Upload ───────────────────────────────────────────── */
function setupUpload() {
    const zone = document.getElementById('uploadZone');
    const input = document.getElementById('fileInput');

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
        e.preventDefault(); zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
    });
    input.addEventListener('change', () => { if (input.files.length) uploadFile(input.files[0]); });
    document.getElementById('btnClearFile').addEventListener('click', () => {
        document.getElementById('fileInfo').style.display = 'none';
        input.value = '';
    });
}

async function uploadFile(file) {
    showLoading('Analyzing uploaded ECG...');
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileInfo').style.display = 'block';

    try {
        const formData = new FormData();
        formData.append('file', file);
        const resp = await fetch(`${API}/api/predict`, { method: 'POST', body: formData });
        if (!resp.ok) { const err = await resp.json(); throw new Error(err.detail || 'Upload failed'); }
        const data = await resp.json();

        currentSignal = data.signal;
        leadNames = data.lead_names || leadNames;
        renderECG(currentSignal, 'all');
        renderDiagnosis(data.predictions);

        document.getElementById('gtSection').style.display = 'none';
        showToast(`Analyzed ${file.name}`, 'success');
    } catch(err) {
        showToast('Error: ' + err.message, 'danger');
    } finally { hideLoading(); }
}

/* ── Random Sample ────────────────────────────────────── */
async function loadRandomSample() {
    showLoading('Loading random sample...');
    try {
        const resp = await fetch(`${API}/api/random-sample`);
        if (!resp.ok) throw new Error('Failed to load sample');
        const data = await resp.json();

        currentSignal = data.signal;
        leadNames = data.lead_names || leadNames;
        renderECG(currentSignal, 'all');
        renderDiagnosis(data.predictions);
        renderGroundTruth(data.ground_truth);

        showToast(`Loaded test sample #${data.sample_index}`, 'success');
    } catch(err) {
        showToast('Error: ' + err.message, 'danger');
    } finally { hideLoading(); }
}

/* ── ECG Rendering ────────────────────────────────────── */
function renderECG(signal, group) {
    const container = document.getElementById('ecgCharts');
    const placeholder = document.getElementById('ecgPlaceholder');
    placeholder.style.display = 'none';
    container.style.display = 'grid';
    document.getElementById('btnSaveECG').style.display = '';

    ecgChartInstances.forEach(c => c.destroy());
    ecgChartInstances = [];
    container.innerHTML = '';

    const indices = LEAD_GROUPS[group] || LEAD_GROUPS.all;
    const numSamples = signal.length;
    const timeAxis = Array.from({length: numSamples}, (_,i) => (i/100).toFixed(2));

    indices.forEach(idx => {
        const wrapper = document.createElement('div');
        wrapper.className = 'ecg-lead-wrapper';

        const label = document.createElement('div');
        label.className = 'ecg-lead-label';
        label.textContent = leadNames[idx] || `Lead ${idx+1}`;
        wrapper.appendChild(label);

        const timeLabel = document.createElement('div');
        timeLabel.className = 'ecg-time-label';
        timeLabel.textContent = '10s · 100Hz';
        wrapper.appendChild(timeLabel);

        const canvas = document.createElement('canvas');
        wrapper.appendChild(canvas);
        container.appendChild(wrapper);

        const leadData = signal.map(row => row[idx]);
        const color = LEAD_COLORS[idx % LEAD_COLORS.length];

        const chart = new Chart(canvas, {
            type: 'line',
            data: {
                labels: timeAxis,
                datasets: [{
                    data: leadData,
                    borderColor: color,
                    borderWidth: 1.2,
                    pointRadius: 0,
                    fill: false,
                    tension: 0,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 600, easing: 'easeOutCubic' },
                plugins: { legend: { display: false }, tooltip: { enabled: false } },
                scales: {
                    x: {
                        display: true,
                        grid: { color: 'rgba(255,255,255,0.03)', drawTicks: false },
                        ticks: { display: false },
                        border: { display: false },
                    },
                    y: {
                        display: true,
                        grid: { color: 'rgba(255,255,255,0.03)', drawTicks: false },
                        ticks: { display: false },
                        border: { display: false },
                    }
                },
            }
        });
        ecgChartInstances.push(chart);
    });
}

/* ── Lead Selector ────────────────────────────────────── */
function setupLeadSelector() {
    document.querySelectorAll('#leadSelector button').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#leadSelector button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (currentSignal) renderECG(currentSignal, btn.dataset.leads);
        });
    });
}

/* ── Diagnosis Rendering ──────────────────────────────── */
function renderDiagnosis(predictions) {
    const placeholder = document.getElementById('diagnosisPlaceholder');
    const content = document.getElementById('diagnosisContent');
    const status = document.getElementById('resultStatus');
    const probCard = document.getElementById('probCard');

    placeholder.style.display = 'none';
    content.style.display = 'block';
    probCard.style.display = 'block';

    const classes = Object.keys(predictions);
    const probs = classes.map(c => predictions[c].probability);
    const predicted = classes.map(c => predictions[c].predicted);
    const anyPositive = predicted.some(p => p);

    // Status badge
    status.textContent = anyPositive ? 'Abnormality Detected' : 'Normal';
    status.className = `badge ${anyPositive ? 'bg-danger' : 'bg-success'}`;

    // Banner
    const banner = document.getElementById('diagnosisBanner');
    if (anyPositive) {
        const posClasses = classes.filter((_,i) => predicted[i]);
        banner.className = 'diagnosis-banner abnormal';
        banner.innerHTML = `<i class="bi bi-exclamation-triangle-fill"></i>
            <div>
                <div>Abnormality Detected</div>
                <div style="font-size:0.75rem;font-weight:400;opacity:0.8;margin-top:2px;">
                    ${posClasses.join(', ')} — further review recommended
                </div>
            </div>`;
    } else {
        banner.className = 'diagnosis-banner normal';
        banner.innerHTML = `<i class="bi bi-check-circle-fill"></i>
            <div>
                <div>Normal ECG</div>
                <div style="font-size:0.75rem;font-weight:400;opacity:0.8;margin-top:2px;">No significant abnormalities detected</div>
            </div>`;
    }

    // Prediction bars
    const barsDiv = document.getElementById('predictionBars');
    barsDiv.innerHTML = classes.map((cls, i) => {
        const prob = predictions[cls].probability;
        const pct = (prob * 100).toFixed(1);
        const isPred = predictions[cls].predicted;
        const clr = CLASS_COLORS[cls] || '#888';
        return `<div class="pred-row">
            <span class="status-icon">${isPred ? '<i class="bi bi-exclamation-circle-fill" style="color:var(--danger)"></i>' : '<i class="bi bi-check-circle-fill" style="color:var(--success)"></i>'}</span>
            <span class="cls-name">${cls}</span>
            <div class="bar-track">
                <div class="bar-fill ${isPred ? 'positive' : 'negative'}" style="width:${pct}%;background:${isPred ? `linear-gradient(90deg, ${clr}, ${clr}88)` : ''}"></div>
            </div>
            <span class="prob-value" style="color:${isPred ? clr : 'var(--text-dim)'}">${pct}%</span>
        </div>`;
    }).join('');

    // Probability chart
    renderProbChart(classes, probs, predicted);
}

function renderProbChart(classes, probs, predicted) {
    if (probChartInstance) probChartInstance.destroy();
    const ctx = document.getElementById('probChart').getContext('2d');

    probChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: classes,
            datasets: [{
                data: probs.map(p => (p*100).toFixed(1)),
                backgroundColor: classes.map((c,i) =>
                    predicted[i] ? (CLASS_COLORS[c] || '#e74c5e') + 'BB' : 'rgba(255,255,255,0.06)'
                ),
                borderColor: classes.map((c,i) =>
                    predicted[i] ? CLASS_COLORS[c] || '#e74c5e' : 'rgba(255,255,255,0.12)'
                ),
                borderWidth: 1.5,
                borderRadius: 6,
                borderSkipped: false,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 800, easing: 'easeOutQuart' },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1a1d28',
                    borderColor: '#353a50',
                    borderWidth: 1,
                    titleColor: '#e4e5eb',
                    bodyColor: '#e4e5eb',
                    callbacks: { label: ctx => `${ctx.parsed.x}%` }
                }
            },
            scales: {
                x: {
                    min: 0, max: 100,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#6b7194', callback: v => v+'%', font: { size: 11 } },
                    border: { display: false },
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#e4e5eb', font: { weight: '700', size: 12 } },
                    border: { display: false },
                }
            }
        }
    });
}

function renderGroundTruth(gtClasses) {
    const section = document.getElementById('gtSection');
    const content = document.getElementById('groundTruth');
    section.style.display = 'block';

    const allClasses = ['NORM','MI','STTC','CD','HYP'];
    content.innerHTML = allClasses.map(cls => {
        const isPos = gtClasses.includes(cls);
        return `<span class="gt-badge ${isPos ? 'positive' : 'negative'}">
            ${isPos ? '<i class="bi bi-check-circle-fill me-1"></i>' : ''}${cls}
        </span>`;
    }).join('');
}

/* ── Save / Download Graphs ───────────────────────────── */
function setupSaveButtons() {
    document.getElementById('btnSaveECG').addEventListener('click', () => saveCardAsImage('ecgCard', 'ecg_waveform.png'));
    document.getElementById('btnSaveProbs').addEventListener('click', () => saveCardAsImage('probCard', 'class_probabilities.png'));
    document.getElementById('btnSavePerf').addEventListener('click', () => saveCardAsImage('perfCard', 'model_performance.png'));
}

function saveCardAsImage(cardId, filename) {
    const card = document.getElementById(cardId);
    if (!card) return;

    showToast('Generating image...', 'secondary');

    html2canvas(card, {
        backgroundColor: '#12151f',
        scale: 2,
        useCORS: true,
        logging: false,
    }).then(canvas => {
        const link = document.createElement('a');
        link.download = filename;
        link.href = canvas.toDataURL('image/png');
        link.click();
        showToast(`Saved ${filename}`, 'success');
    }).catch(err => {
        console.error('Save error:', err);
        showToast('Failed to save image', 'danger');
    });
}

/* ── Utilities ────────────────────────────────────────── */
function showLoading(text) {
    document.getElementById('loadingText').textContent = text || 'Loading...';
    document.getElementById('loadingOverlay').style.display = 'flex';
}
function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}
function showToast(message, type) {
    const toast = document.getElementById('toastNotif');
    const body = document.getElementById('toastBody');
    body.textContent = message;
    toast.className = `toast align-items-center border-0 text-bg-${type || 'secondary'}`;
    new bootstrap.Toast(toast, { delay: 3000 }).show();
}
