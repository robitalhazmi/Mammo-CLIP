const state = {
    model: null,
    label: null,
    setting: null,
    prompts: [],
    imageFile: null,
    sampleImage: null,
    isLoading: false,
    results: null,
    paperMetrics: null,
    availableModels: null,
};

const DEFAULT_PROMPTS = {
    mass: ['no mass', 'mass'],
    calcification: ['no suspicious calcification', 'suspicious calcification'],
    malignancy: ['benign', 'malignant']
};

const METRICS_LABELS = {
    'zs_100':  'Zero-shot (100%)',
    'lp_10':   'Linear Probe (10%)',
    'lp_50':   'Linear Probe (50%)',
    'lp_100':  'Linear Probe (100%)'
};

// Map frontend setting to metrics key prefix
const SETTING_TO_METRIC_KEY = {
    'zero_shot': 'zs_100',
    'linear_probe': 'lp_100'
};

let DOM = {};

function cacheDom() {
    DOM = {
        deviceStatus: document.getElementById('deviceStatus'),
        deviceText: document.getElementById('deviceText'),
        modelInputs: document.querySelectorAll('input[name="model"]'),
        labelInputs: document.querySelectorAll('input[name="label"]'),
        settingInputs: document.querySelectorAll('input[name="setting"]'),
        linearProbeCard: document.getElementById('linearProbeCard'),
        promptsSection: document.getElementById('promptsSection'),
        promptInputsContainer: document.getElementById('promptInputsContainer'),
        resetPromptsBtn: document.getElementById('resetPromptsBtn'),
        sampleImagesSelect: document.getElementById('sampleImages'),
        uploadZone: document.getElementById('uploadZone'),
        fileInput: document.getElementById('fileInput'),
        previewContainer: document.getElementById('previewContainer'),
        imagePreviewSmall: document.getElementById('imagePreviewSmall'),
        clearImageBtn: document.getElementById('clearImageBtn'),
        runBtn: document.getElementById('runBtn'),
        runSpinner: document.getElementById('runSpinner'),
        breadcrumb: document.getElementById('breadcrumb'),
        emptyState: document.getElementById('emptyState'),
        resultsView: document.getElementById('resultsView'),
        predictionSummary: document.getElementById('predictionSummary'),
        originalImage: document.getElementById('originalImage'),
        heatmapImage: document.getElementById('heatmapImage'),

        metadataFooter: document.getElementById('metadataFooter'),
    };
}

function initApp() {
    cacheDom();
    setupEventListeners();
    
    // Initialize state from inputs
    const modelInput = document.querySelector('input[name="model"]');
    if (modelInput) {
        modelInput.checked = true;
        selectModel(modelInput.value);
    }
    
    const checkedLabel = document.querySelector('input[name="label"]:checked');
    if (checkedLabel) selectLabel(checkedLabel.value);
    
    const checkedSetting = document.querySelector('input[name="setting"]:checked');
    if (checkedSetting) selectSetting(checkedSetting.value);
    
    fetchHealthStatus();
    fetchSampleImages();

    updateUI();
}

function setupEventListeners() {
    DOM.modelInputs.forEach(input => {
        input.addEventListener('change', (e) => selectModel(e.target.value));
    });
    DOM.labelInputs.forEach(input => {
        input.addEventListener('change', (e) => selectLabel(e.target.value));
    });
    DOM.settingInputs.forEach(input => {
        input.addEventListener('change', (e) => selectSetting(e.target.value));
    });
    DOM.resetPromptsBtn.addEventListener('click', resetPrompts);
    
    // Image upload
    DOM.uploadZone.addEventListener('click', () => DOM.fileInput.click());
    DOM.uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        DOM.uploadZone.classList.add('dragover');
    });
    DOM.uploadZone.addEventListener('dragleave', () => {
        DOM.uploadZone.classList.remove('dragover');
    });
    DOM.uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        DOM.uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleImageUpload(e.dataTransfer.files[0]);
        }
    });
    DOM.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleImageUpload(e.target.files[0]);
    });
    
    DOM.sampleImagesSelect.addEventListener('change', (e) => selectSampleImage(e.target.value));
    DOM.clearImageBtn.addEventListener('click', clearImage);
    DOM.runBtn.addEventListener('click', runPrediction);
}

async function fetchHealthStatus() {
    try {
        const res = await fetch('/api/health');
        if (res.ok) {
            const data = await res.json();
            const deviceLabel = data.cuda_available 
                ? `GPU: ${data.gpu_name || data.device}` 
                : `CPU: ${data.device}`;
            DOM.deviceText.textContent = deviceLabel;
            DOM.deviceStatus.querySelector('.status-indicator').classList.add('ready');
        }
    } catch (e) {
        DOM.deviceText.textContent = 'System Offline';
    }
}

async function fetchSampleImages() {
    try {
        const res = await fetch('/api/sample-images');
        if (res.ok) {
            const samples = await res.json();
            samples.forEach(s => {
                const opt = document.createElement('option');
                opt.value = s.filename;
                opt.textContent = s.filename;
                DOM.sampleImagesSelect.appendChild(opt);
            });
        }
    } catch (e) {
        console.error('Failed to load sample images', e);
    }
}



function selectModel(model) {
    state.model = model;
    updateSettingAvailability();
    updateUI();
}

function selectLabel(label) {
    state.label = label;
    if (state.setting === 'zero_shot') {
        resetPrompts();
    }
    updateUI();
}

function selectSetting(setting) {
    state.setting = setting;
    if (setting === 'zero_shot') {
        DOM.promptsSection.classList.remove('hidden');
        resetPrompts();
    } else {
        DOM.promptsSection.classList.add('hidden');
    }
    updateUI();
}

function updateSettingAvailability() {
    DOM.linearProbeCard.classList.remove('disabled');
    DOM.settingInputs.forEach(input => {
        input.disabled = false;
    });
}

function resetPrompts() {
    if (!state.label) return;
    state.prompts = [...DEFAULT_PROMPTS[state.label]];
    renderPromptInputs();
}

function renderPromptInputs() {
    DOM.promptInputsContainer.innerHTML = '';
    state.prompts.forEach((p, idx) => {
        const div = document.createElement('div');
        div.className = 'prompt-input-group';
        div.innerHTML = `
            <label class="input-label">Prompt ${idx + 1}</label>
            <input type="text" class="input-field prompt-input" value="${p}" data-idx="${idx}">
        `;
        DOM.promptInputsContainer.appendChild(div);
    });
    document.querySelectorAll('.prompt-input').forEach(input => {
        input.addEventListener('input', (e) => {
            state.prompts[parseInt(e.target.dataset.idx)] = e.target.value;
        });
    });
}

function handleImageUpload(file) {
    if (!file.type.match('image.*') && !file.name.toLowerCase().endsWith('.dicom')) return;
    state.imageFile = file;
    state.sampleImage = null;
    DOM.sampleImagesSelect.value = '';
    showImagePreview(URL.createObjectURL(file));
    updateUI();
}

function selectSampleImage(filename) {
    if (!filename) { clearImage(); return; }
    state.sampleImage = filename;
    state.imageFile = null;
    showImagePreview(`/static/sample_images/${filename}`);
    updateUI();
}

function showImagePreview(url) {
    DOM.imagePreviewSmall.src = url;
    DOM.uploadZone.classList.add('hidden');
    DOM.previewContainer.classList.remove('hidden');
}

function clearImage() {
    state.imageFile = null;
    state.sampleImage = null;
    DOM.fileInput.value = '';
    DOM.sampleImagesSelect.value = '';
    DOM.imagePreviewSmall.src = '';
    DOM.uploadZone.classList.remove('hidden');
    DOM.previewContainer.classList.add('hidden');
    updateUI();
}

function updateUI() {
    // Breadcrumb
    const m = state.model ? 'Prompt-Guided MV VLM' : 'Model';
    const l = state.label ? state.label.charAt(0).toUpperCase() + state.label.slice(1) : 'Label';
    const settingNames = { zero_shot: 'Zero-shot', linear_probe: 'Linear Probe' };
    const s = state.setting ? settingNames[state.setting] : 'Setting';

    DOM.breadcrumb.innerHTML = `
        <span class="${state.model ? 'active' : ''}">${m}</span>
        <span class="separator">›</span>
        <span class="${state.label ? 'active' : ''}">${l}</span>
        <span class="separator">›</span>
        <span class="${state.setting ? 'active' : ''}">${s}</span>
    `;

    // Run button
    const canRun = state.model && state.label && state.setting 
        && (state.imageFile || state.sampleImage) && !state.isLoading;
    DOM.runBtn.disabled = !canRun;
    DOM.runBtn.classList.toggle('pulse', canRun);
}

async function runPrediction() {
    state.isLoading = true;
    updateUI();
    
    const btnText = DOM.runBtn.querySelector('.btn-text');
    btnText.textContent = 'Processing...';
    DOM.runSpinner.classList.remove('hidden');
    DOM.emptyState.classList.add('hidden');
    DOM.resultsView.classList.add('hidden');

    try {
        const formData = new FormData();
        formData.append('model', state.model);
        formData.append('label', state.label);
        formData.append('setting', state.setting);
        
        if (state.setting === 'zero_shot' && state.prompts.length > 0) {
            formData.append('prompts', JSON.stringify(state.prompts));
        }

        if (state.imageFile) {
            formData.append('image', state.imageFile);
        } else if (state.sampleImage) {
            formData.append('sample_image', state.sampleImage);
        }

        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(err.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error('Prediction failed:', error);
        showError(error.message);
    } finally {
        state.isLoading = false;
        btnText.textContent = 'Run Analysis';
        DOM.runSpinner.classList.add('hidden');
        updateUI();
    }
}

function showError(message) {
    DOM.predictionSummary.innerHTML = `
        <div class="error-message">
            <span class="error-icon">⚠️</span>
            <p>${message}</p>
        </div>
    `;
    DOM.resultsView.classList.remove('hidden');
}

function displayResults(data) {
    DOM.resultsView.classList.remove('hidden');
    
    const pred = data.prediction;
    
    // Prediction summary with probability bars
    let html = `<div class="prediction-large">${pred.predicted_class}</div>`;
    html += `<div class="confidence-badge">Confidence: ${(pred.confidence * 100).toFixed(1)}%</div>`;
    
    pred.classes.forEach((cls, idx) => {
        const prob = pred.probabilities[idx];
        const percent = (prob * 100).toFixed(1);
        const isActive = idx === pred.prediction_idx;
        html += `
            <div class="progress-container ${isActive ? 'active' : ''}">
                <div class="progress-header">
                    <span>${cls}</span>
                    <span class="mono">${percent}%</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-fill" style="width: 0%" data-width="${percent}"></div>
                </div>
            </div>
        `;
    });
    DOM.predictionSummary.innerHTML = html;
    
    // Animate progress bars
    requestAnimationFrame(() => {
        document.querySelectorAll('.progress-fill[data-width]').forEach(bar => {
            bar.style.width = bar.dataset.width + '%';
        });
    });
    
    // Images
    const vis = data.visualization;
    if (vis.original) {
        DOM.originalImage.src = `data:image/png;base64,${vis.original}`;
    }
    if (vis.overlay) {
        DOM.heatmapImage.src = `data:image/png;base64,${vis.overlay}`;
        DOM.heatmapImage.style.filter = 'none';
    } else if (vis.original) {
        DOM.heatmapImage.src = `data:image/png;base64,${vis.original}`;
        DOM.heatmapImage.style.filter = 'sepia(1) hue-rotate(-50deg) saturate(3)';
    }
    
    
    // Footer metadata
    DOM.metadataFooter.innerHTML = `
        <span>Inference: ${data.inference_time}s</span> |
        <span>Model: ${data.model_name}</span> |
        <span>Device: ${data.device}</span>
    `;
}


document.addEventListener('DOMContentLoaded', initApp);
