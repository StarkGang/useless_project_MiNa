/**
 * THE UNNECESSARY FM — Spotify-Inspired Modern Audio Studio
 * 100% Pure DSP Procedural Music Generation Logic
 * Team MiNa • Midhun K M & Nayana P • TinkerHub Useless Projects 3.0
 */

(() => {
  // API URL Configuration:
  // 1. Checks window.ENV.BACKEND_URL (injected from ENV variable BACKEND_URL via /env.js)
  // 2. Checks window.BACKEND_URL or window.API_BASE_URL
  // 3. Checks <meta name="backend-url"> or <meta name="api-base-url">
  // 4. Checks URL query param (?backend=... or ?api=...)
  // 5. Checks localStorage('BACKEND_URL' or 'API_BASE_URL')
  // If not set, defaults to "" (current origin / relative path).
  const getApiBaseUrl = () => {
    try {
      if (typeof window !== 'undefined' && window.ENV && window.ENV.BACKEND_URL) {
        return window.ENV.BACKEND_URL.trim().replace(/\/+$/, '');
      }
      if (typeof window !== 'undefined' && (window.BACKEND_URL || window.API_BASE_URL)) {
        return (window.BACKEND_URL || window.API_BASE_URL).trim().replace(/\/+$/, '');
      }
      const meta = document.querySelector('meta[name="backend-url"], meta[name="api-base-url"]');
      if (meta && meta.content && meta.content.trim()) {
        return meta.content.trim().replace(/\/+$/, '');
      }
      const urlParams = new URLSearchParams(window.location.search);
      const queryApi = urlParams.get('backend') || urlParams.get('api');
      if (queryApi) return queryApi.trim().replace(/\/+$/, '');
      const stored = localStorage.getItem('BACKEND_URL') || localStorage.getItem('API_BASE_URL');
      if (stored && stored.trim()) return stored.trim().replace(/\/+$/, '');
    } catch (e) {}
    return '';
  };

  const API_BASE = getApiBaseUrl();
  const buildUrl = (path) => {
    if (!path) return '';
    if (path.startsWith('http://') || path.startsWith('https://')) return path;
    const cleanPath = path.startsWith('/') ? path : `/${path}`;
    return API_BASE ? `${API_BASE}${cleanPath}` : cleanPath;
  };

  // DOM Elements - Source & Input
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('audioFileInput');
  const dropZoneContent = document.getElementById('dropZoneContent');
  const selectedFilePill = document.getElementById('selectedFilePill');
  const selectedFileName = document.getElementById('selectedFileName');
  const selectedFileSize = document.getElementById('selectedFileSize');
  const btnRemoveFile = document.getElementById('btnRemoveFile');

  // Microphone Recording Elements
  const btnRecord = document.getElementById('btnRecord');
  const recordBtnText = document.getElementById('recordBtnText');
  const recTimer = document.getElementById('recTimer');
  const recTimeDisplay = document.getElementById('recTimeDisplay');
  const liveMicCanvas = document.getElementById('liveMicCanvas');

  // Composition Directives Elements
  const beatSelect = document.getElementById('beatSelect');
  const energySelect = document.getElementById('energySelect');
  const seedInput = document.getElementById('seedInput');
  const btnRandomizeSeed = document.getElementById('btnRandomizeSeed');
  const btnGenerate = document.getElementById('btnGenerate');

  // Generation & Status UI Elements
  const emptyState = document.getElementById('emptyState');
  const processingState = document.getElementById('processingState');
  const resultsDisplay = document.getElementById('resultsDisplay');
  const progressPercent = document.getElementById('progressPercent');
  const progressStage = document.getElementById('progressStage');
  const progressBarFill = document.getElementById('progressBarFill');

  // Candidate tabs & metadata banner
  const candidateTabs = document.getElementById('candidateTabs');
  const metaGenre = document.getElementById('metaGenre');
  const metaTempo = document.getElementById('metaTempo');
  const metaScale = document.getElementById('metaScale');
  const metaDuration = document.getElementById('metaDuration');
  const metaSourceRatio = document.getElementById('metaSourceRatio');
  const metaSynthRatio = document.getElementById('metaSynthRatio');
  const metaScore = document.getElementById('metaScore');
  const metaSeed = document.getElementById('metaSeed');

  // Waveform displays & controls
  const srcNameDisplay = document.getElementById('srcNameDisplay');
  const srcTimeDisplay = document.getElementById('srcTimeDisplay');
  const srcWaveformCanvas = document.getElementById('srcWaveformCanvas');
  const srcPlayhead = document.getElementById('srcPlayhead');
  const srcCanvasWrapper = document.getElementById('srcCanvasWrapper');
  const btnPlaySource = document.getElementById('btnPlaySource');
  const srcPlayIcon = document.getElementById('srcPlayIcon');
  const srcPlayText = document.getElementById('srcPlayText');

  const resFormDisplay = document.getElementById('resFormDisplay');
  const resTimeDisplay = document.getElementById('resTimeDisplay');
  const resWaveformCanvas = document.getElementById('resWaveformCanvas');
  const resPlayhead = document.getElementById('resPlayhead');
  const resCanvasWrapper = document.getElementById('resCanvasWrapper');
  const btnPlayResult = document.getElementById('btnPlayResult');
  const resPlayIcon = document.getElementById('resPlayIcon');
  const resPlayText = document.getElementById('resPlayText');
  const btnDownload = document.getElementById('btnDownload');

  // Acoustic DNA Panel Elements
  const catPill = document.getElementById('catPill');
  const meterRhythm = document.getElementById('meterRhythm');
  const valRhythm = document.getElementById('valRhythm');
  const meterHarmonic = document.getElementById('meterHarmonic');
  const valHarmonic = document.getElementById('valHarmonic');
  const meterTexture = document.getElementById('meterTexture');
  const valTexture = document.getElementById('valTexture');
  const meterEnergy = document.getElementById('meterEnergy');
  const valEnergy = document.getElementById('valEnergy');

  const dnaDetectedPitch = document.getElementById('dnaDetectedPitch');
  const dnaActiveStems = document.getElementById('dnaActiveStems');
  const dnaPaletteSlices = document.getElementById('dnaPaletteSlices');
  const dnaClassReason = document.getElementById('dnaClassReason');
  const dnaFormName = document.getElementById('dnaFormName');
  const btnRegenerateNew = document.getElementById('btnRegenerateNew');

  // Hidden Audio Players & Notifications
  const sourceAudioPlayer = document.getElementById('sourceAudioPlayer');
  const resultAudioPlayer = document.getElementById('resultAudioPlayer');
  const toastContainer = document.getElementById('toastContainer');

  // Persistent Spotify Bottom Player Elements
  const bottomPlayer = document.getElementById('bottomPlayer');
  const playerTrackTitle = document.getElementById('playerTrackTitle');
  const playerTrackSubtitle = document.getElementById('playerTrackSubtitle');
  const btnPlayerFav = document.getElementById('btnPlayerFav');
  const playerBtnShuffle = document.getElementById('playerBtnShuffle');
  const playerBtnPrev = document.getElementById('playerBtnPrev');
  const playerBtnPlay = document.getElementById('playerBtnPlay');
  const playerPlaySvg = document.getElementById('playerPlaySvg');
  const playerBtnNext = document.getElementById('playerBtnNext');
  const playerBtnLoop = document.getElementById('playerBtnLoop');
  const playerCurrentTime = document.getElementById('playerCurrentTime');
  const playerTotalDuration = document.getElementById('playerTotalDuration');
  const playerSliderTrack = document.getElementById('playerSliderTrack');
  const playerSliderFill = document.getElementById('playerSliderFill');
  const playerSliderThumb = document.getElementById('playerSliderThumb');
  const playerBtnTogglePanel = document.getElementById('playerBtnTogglePanel');
  const btnCloseRightPanel = document.getElementById('btnCloseRightPanel');
  const playerBtnDownload = document.getElementById('playerBtnDownload');
  const playerBtnVolume = document.getElementById('playerBtnVolume');
  const volumeSliderTrack = document.getElementById('volumeSliderTrack');
  const volumeSliderFill = document.getElementById('volumeSliderFill');
  const volumeSvg = document.getElementById('volumeSvg');

  // Right Panel & Shell Elements
  const spotifyShell = document.querySelector('.spotify-shell');
  const rightPanelTrackTitle = document.getElementById('rightPanelTrackTitle');
  const rightPanelTrackSub = document.getElementById('rightPanelTrackSub');
  const btnScrollToDNA = document.getElementById('btnScrollToDNA');

  // Library & Search Elements
  const libCurrentTrack = document.getElementById('libCurrentTrack');
  const libTrackTitle = document.getElementById('libTrackTitle');
  const libTrackMeta = document.getElementById('libTrackMeta');
  const libSourceTrack = document.getElementById('libSourceTrack');
  const libSourceTitle = document.getElementById('libSourceTitle');
  const quickFilterInput = document.getElementById('quickFilterInput');

  // State
  let currentFile = null;
  let currentJobId = null;
  let currentResultData = null;
  let activeCandidateIndex = 0;
  let isRecording = false;
  let mediaRecorder = null;
  let audioChunks = [];
  let recordStartTime = 0;
  let recordTimerInterval = null;
  let micAudioContext = null;
  let micAnalyser = null;
  let micStream = null;
  let micAnimFrame = null;
  let isLooping = false;
  let isMuted = false;
  let previousVolume = 0.8;
  let currentVolume = 0.8;
  let activePlayingTarget = 'result'; // 'result' or 'source'

  // Set default audio volume
  sourceAudioPlayer.volume = currentVolume;
  resultAudioPlayer.volume = currentVolume;

  // Web Audio Context for Waveform Decoding
  let decodeAudioCtx = null;
  function getDecodeAudioContext() {
    if (!decodeAudioCtx) {
      decodeAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    return decodeAudioCtx;
  }

  // Toast Notifications
  function showToast(message, type = 'info', title = '') {
    if (!toastContainer) {
      alert(message);
      return;
    }
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const defaultTitle = type === 'error' ? 'Notice' : (type === 'success' ? 'Success' : 'The Unnecessary FM');
    toast.innerHTML = `
      <div style="flex: 1;">
        <div class="toast-title">${title || defaultTitle}</div>
        <div class="toast-message">${escapeHtml(message)}</div>
      </div>
      <button type="button" class="toast-close" title="Close">&times;</button>
    `;
    toast.querySelector('.toast-close').addEventListener('click', () => {
      toast.remove();
    });
    toastContainer.appendChild(toast);
    setTimeout(() => {
      if (toast.parentElement) toast.remove();
    }, 5000);
  }

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, s => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    }[s]));
  }

  // 1. File Upload & Drag-and-Drop
  if (dropZone) {
    dropZone.addEventListener('click', (e) => {
      if (e.target !== btnRemoveFile && !btnRemoveFile.contains(e.target)) {
        fileInput.click();
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelected(e.target.files[0]);
      }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');
      });
    });

    dropZone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelected(e.dataTransfer.files[0]);
      }
    });
  }

  if (btnRemoveFile) {
    btnRemoveFile.addEventListener('click', (e) => {
      e.stopPropagation();
      resetFileInput();
    });
  }

  function handleFileSelected(file) {
    currentFile = file;
    if (selectedFileName) selectedFileName.textContent = file.name;
    if (selectedFileSize) selectedFileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
    if (dropZoneContent) dropZoneContent.classList.add('hidden');
    if (selectedFilePill) selectedFilePill.classList.remove('hidden');
    if (btnGenerate) btnGenerate.disabled = false;

    // Update Player & Library state
    if (playerTrackTitle) playerTrackTitle.textContent = file.name;
    if (playerTrackSubtitle) playerTrackSubtitle.textContent = 'Uploaded Audio • Ready to Compose';
    if (libSourceTitle) libSourceTitle.textContent = file.name;

    showToast(`Loaded "${file.name}". Ready to compose ~60s music!`, 'success', 'AUDIO READY');
  }

  function resetFileInput() {
    currentFile = null;
    fileInput.value = '';
    if (selectedFilePill) selectedFilePill.classList.add('hidden');
    if (dropZoneContent) dropZoneContent.classList.remove('hidden');
    if (!currentJobId && btnGenerate) {
      btnGenerate.disabled = true;
    }
  }

  // 2. Microphone Recording (5 - 60s)
  if (btnRecord) {
    btnRecord.addEventListener('click', async () => {
      if (!isRecording) {
        startMicRecording();
      } else {
        stopMicRecording();
      }
    });
  }

  async function startMicRecording() {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      mediaRecorder = new MediaRecorder(micStream);
      audioChunks = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunks.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        const recordedFile = new File([audioBlob], `noise_recording_${Date.now()}.webm`, { type: 'audio/webm' });
        handleFileSelected(recordedFile);
        cleanupMicStream();
      };

      mediaRecorder.start(100);
      isRecording = true;
      btnRecord.classList.add('recording');
      if (recordBtnText) recordBtnText.textContent = 'Stop Recording';
      if (recTimer) recTimer.classList.remove('hidden');
      if (liveMicCanvas) liveMicCanvas.classList.remove('hidden');

      recordStartTime = Date.now();
      updateRecordTimer();
      recordTimerInterval = setInterval(updateRecordTimer, 500);

      // Start live visualizer
      startLiveMicVisualizer(micStream);

      // Auto stop at 60s
      setTimeout(() => {
        if (isRecording) {
          stopMicRecording();
        }
      }, 60000);

    } catch (err) {
      showToast('Microphone access was denied or not available: ' + err.message, 'error', 'MIC ACCESS ERROR');
    }
  }

  function stopMicRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
    }
    isRecording = false;
    if (btnRecord) btnRecord.classList.remove('recording');
    if (recordBtnText) recordBtnText.textContent = 'Record Mic (5–60s)';
    if (recTimer) recTimer.classList.add('hidden');
    if (liveMicCanvas) liveMicCanvas.classList.add('hidden');
    if (recordTimerInterval) {
      clearInterval(recordTimerInterval);
      recordTimerInterval = null;
    }
  }

  function updateRecordTimer() {
    const elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
    const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const s = String(elapsed % 60).padStart(2, '0');
    if (recTimeDisplay) recTimeDisplay.textContent = `${m}:${s}`;
  }

  function startLiveMicVisualizer(stream) {
    if (!liveMicCanvas) return;
    micAudioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = micAudioContext.createMediaStreamSource(stream);
    micAnalyser = micAudioContext.createAnalyser();
    micAnalyser.fftSize = 64;
    source.connect(micAnalyser);

    const canvasCtx = liveMicCanvas.getContext('2d');
    const dataArray = new Uint8Array(micAnalyser.frequencyBinCount);

    function draw() {
      if (!isRecording) return;
      micAnimFrame = requestAnimationFrame(draw);
      micAnalyser.getByteFrequencyData(dataArray);

      canvasCtx.clearRect(0, 0, liveMicCanvas.width, liveMicCanvas.height);
      const barWidth = (liveMicCanvas.width / dataArray.length) * 1.5;
      let x = 0;

      for (let i = 0; i < dataArray.length; i++) {
        const barHeight = (dataArray[i] / 255) * (liveMicCanvas.height - 4);
        canvasCtx.fillStyle = '#1ed760';
        canvasCtx.fillRect(x, liveMicCanvas.height - barHeight, Math.max(2, barWidth - 2), barHeight);
        x += barWidth;
      }
    }
    draw();
  }

  function cleanupMicStream() {
    if (micStream) {
      micStream.getTracks().forEach(t => t.stop());
      micStream = null;
    }
    if (micAudioContext) {
      micAudioContext.close();
      micAudioContext = null;
    }
    if (micAnimFrame) {
      cancelAnimationFrame(micAnimFrame);
      micAnimFrame = null;
    }
  }

  // 3. Seed Helpers
  if (btnRandomizeSeed) {
    btnRandomizeSeed.addEventListener('click', () => {
      seedInput.value = Math.floor(Math.random() * 9000000000) + 1000000000;
    });
  }

  // 4. Quick Preset Shelves
  document.querySelectorAll('.preset-card').forEach(card => {
    card.addEventListener('click', () => {
      const g = card.dataset.genre;
      const e = card.dataset.energy;
      if (g && beatSelect) beatSelect.value = g;
      if (e && energySelect) energySelect.value = e;
      const title = card.querySelector('h3') ? card.querySelector('h3').textContent : g;
      showToast(`Selected style: ${title}. Choose an audio file and hit Create!`, 'info', 'STYLE PRESET');

      // Scroll to studio
      const studioSec = document.getElementById('studioSection');
      if (studioSec) {
        studioSec.scrollIntoView({ behavior: 'smooth' });
      }
    });
  });

  // Quick filter search
  if (quickFilterInput) {
    quickFilterInput.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase().trim();
      document.querySelectorAll('.preset-card').forEach(card => {
        const text = card.textContent.toLowerCase();
        card.style.display = (!q || text.includes(q)) ? 'flex' : 'none';
      });
    });
  }

  // Navigation Links
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
      document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      const target = item.dataset.target;
      if (target === 'home') {
        const h = document.getElementById('heroBanner');
        if (h) h.scrollIntoView({ behavior: 'smooth' });
      } else if (target === 'studio') {
        const s = document.getElementById('studioSection');
        if (s) s.scrollIntoView({ behavior: 'smooth' });
      } else if (target === 'dna') {
        if (spotifyShell) spotifyShell.classList.remove('right-panel-closed');
      } else if (target === 'presets') {
        const p = document.getElementById('presetsShelf');
        if (p) p.scrollIntoView({ behavior: 'smooth' });
      }
    });
  });

  // Mobile Nav items
  document.querySelectorAll('.mobile-nav-item').forEach(item => {
    item.addEventListener('click', () => {
      document.querySelectorAll('.mobile-nav-item').forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      const target = item.dataset.target;
      if (target === 'home') {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      } else if (target === 'studio') {
        const s = document.getElementById('studioSection');
        if (s) s.scrollIntoView({ behavior: 'smooth' });
      } else if (target === 'results') {
        const r = document.getElementById('resultsSection');
        if (r) r.scrollIntoView({ behavior: 'smooth' });
      } else if (target === 'dna') {
        showToast('Acoustic DNA is displayed in the Right Panel on desktop!', 'info', 'ACOUSTIC DNA');
      }
    });
  });

  // 5. Generate Music Request
  if (btnGenerate) {
    btnGenerate.addEventListener('click', () => {
      submitMusicGeneration();
    });
  }

  if (btnRegenerateNew) {
    btnRegenerateNew.addEventListener('click', () => {
      if (seedInput) {
        seedInput.value = Math.floor(Math.random() * 9000000000) + 1000000000;
      }
      submitMusicGeneration();
    });
  }

  async function submitMusicGeneration() {
    if (!currentFile && !currentJobId) return;

    showProcessingState("Initializing DSP pipeline & decoding acoustic timbre...");

    const formData = new FormData();
    if (currentFile) {
      formData.append('file', currentFile);
    } else if (currentJobId) {
      formData.append('existing_job_id', currentJobId);
    }

    formData.append('beat_preference', beatSelect ? beatSelect.value : 'hiphop');
    formData.append('energy_preference', energySelect ? energySelect.value : 'balanced');
    if (seedInput && seedInput.value.trim()) {
      formData.append('seed', seedInput.value.trim());
    }

    try {
      const response = await fetch(buildUrl('/api/generate'), {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to submit generation job');
      }

      const data = await response.json();
      currentJobId = data.job_id;
      pollJobStatus(currentJobId);

    } catch (err) {
      showToast(err.message, 'error', 'COMPOSITION FAILED');
      hideProcessingState();
    }
  }

  function showProcessingState(stageText) {
    if (emptyState) emptyState.classList.add('hidden');
    if (resultsDisplay) resultsDisplay.classList.add('hidden');
    if (processingState) processingState.classList.remove('hidden');
    if (progressPercent) progressPercent.textContent = '10%';
    if (progressBarFill) progressBarFill.style.width = '10%';
    if (progressStage) progressStage.textContent = stageText || 'Processing...';
    if (btnGenerate) btnGenerate.disabled = true;

    // Scroll to results section
    const resSec = document.getElementById('resultsSection');
    if (resSec) resSec.scrollIntoView({ behavior: 'smooth' });
  }

  function hideProcessingState() {
    if (processingState) processingState.classList.add('hidden');
    if (btnGenerate) btnGenerate.disabled = false;
  }

  // 6. Job Status Polling
  function pollJobStatus(jobId) {
    let failCount = 0;
    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(buildUrl(`/api/status/${jobId}`));
        if (!res.ok) {
          failCount++;
          if (failCount >= 8) {
            clearInterval(pollInterval);
            showToast('Service temporarily unavailable or restarted. Please try generating again.', 'error', 'CONNECTION ERROR');
            hideProcessingState();
          }
          return;
        }
        failCount = 0;

        const statusData = await res.json();
        const pct = Math.max(10, statusData.progress || 10);
        if (progressPercent) progressPercent.textContent = `${pct}%`;
        if (progressBarFill) progressBarFill.style.width = `${pct}%`;
        if (progressStage) progressStage.textContent = statusData.stage || 'Synthesizing audio...';

        if (statusData.status === 'completed') {
          clearInterval(pollInterval);
          loadFinalResults(jobId);
        } else if (statusData.status === 'failed') {
          clearInterval(pollInterval);
          showToast(statusData.error || 'DSP synthesis error', 'error', 'SYNTHESIS ERROR');
          hideProcessingState();
        }
      } catch (e) {
        failCount++;
        console.error("Polling error:", e);
        if (failCount >= 8) {
          clearInterval(pollInterval);
          showToast('Connection interrupted. Please try again.', 'error', 'NETWORK ERROR');
          hideProcessingState();
        }
      }
    }, 800);
  }

  // 7. Load & Render Results
  async function loadFinalResults(jobId) {
    try {
      const res = await fetch(buildUrl(`/api/result/${jobId}`));
      if (!res.ok) throw new Error('Failed to retrieve result');

      currentResultData = await res.json();
      renderCompositionUI(currentResultData);
      const candCount = (currentResultData.candidates && currentResultData.candidates.length) || 2;
      showToast(`1-minute music composition synthesized across ${candCount} candidates!`, 'success', 'TA DAA!');

    } catch (err) {
      showToast(err.message, 'error', 'LOAD ERROR');
      hideProcessingState();
    }
  }

  function renderCompositionUI(result) {
    if (processingState) processingState.classList.add('hidden');
    if (emptyState) emptyState.classList.add('hidden');
    if (resultsDisplay) resultsDisplay.classList.remove('hidden');
    if (btnGenerate) btnGenerate.disabled = false;

    // Populate DNA Panel
    const analysis = result.source.analysis || {};
    if (catPill) catPill.textContent = analysis.classification || 'TEXTURAL';

    const rDens = analysis.rhythmic_density || 0;
    if (meterRhythm) meterRhythm.style.width = `${Math.min(100, rDens * 35)}%`;
    if (valRhythm) valRhythm.textContent = rDens.toFixed(2);

    const harm = analysis.harmonicity || 0.5;
    if (meterHarmonic) meterHarmonic.style.width = `${Math.min(100, harm * 100)}%`;
    if (valHarmonic) valHarmonic.textContent = harm.toFixed(2);

    const text = analysis.noisiness || 0.5;
    if (meterTexture) meterTexture.style.width = `${Math.min(100, text * 100)}%`;
    if (valTexture) valTexture.textContent = text.toFixed(2);

    const bright = analysis.brightness || 0.5;
    if (meterEnergy) meterEnergy.style.width = `${Math.min(100, bright * 100)}%`;
    if (valEnergy) valEnergy.textContent = bright.toFixed(2);

    if (dnaDetectedPitch) {
      dnaDetectedPitch.textContent = analysis.has_reliable_pitch ? `${analysis.nearest_note}` : 'None (Resonators Applied)';
    }
    if (dnaClassReason) {
      dnaClassReason.textContent = (analysis.reasons && analysis.reasons.length > 0) ? analysis.reasons[0] : 'Balanced harmonic & transient spectrum';
    }

    // Source Audio Setup
    if (srcNameDisplay) srcNameDisplay.textContent = result.source.filename || 'source.wav';
    sourceAudioPlayer.src = buildUrl(result.source.audio_url);
    drawWaveformFromUrl(result.source.audio_url, srcWaveformCanvas, '#22d3ee');

    // Build Candidate Tabs
    if (candidateTabs) {
      candidateTabs.innerHTML = '';
      result.candidates.forEach((cand, idx) => {
        const tab = document.createElement('button');
        tab.type = 'button';
        tab.className = `cand-tab ${idx === 0 ? 'active' : ''}`;
        tab.dataset.index = idx;
        tab.innerHTML = `<span>Candidate ${String.fromCharCode(65 + idx)}</span> ${cand.is_winner ? '<span class="tag-winner">WINNER</span>' : ''}`;
        tab.addEventListener('click', () => selectCandidate(idx));
        candidateTabs.appendChild(tab);
      });
    }

    // Select Winner by default
    selectCandidate(0);
  }

  function selectCandidate(index) {
    if (!currentResultData || !currentResultData.candidates[index]) return;
    activeCandidateIndex = index;
    activePlayingTarget = 'result';

    // Update active tab styles
    const tabs = candidateTabs.querySelectorAll('.cand-tab');
    tabs.forEach((t, idx) => {
      t.classList.toggle('active', idx === index);
    });

    const cand = currentResultData.candidates[index];
    const letter = String.fromCharCode(65 + index);

    // Update Banner
    if (metaGenre) {
      const pref = (currentResultData.settings && currentResultData.settings.beat_preference) || (beatSelect ? beatSelect.value : 'hiphop');
      const genreLabels = {
        'hiphop': 'Hip Hop',
        'rap': 'Rap / Trap',
        'pop': 'Pop',
        'minimal': 'Minimal',
        'rhythmic': 'Rhythmic Pulse',
        'light_percussion': 'Light Percussion',
        'none': 'Ambient'
      };
      metaGenre.textContent = genreLabels[pref.toLowerCase()] || pref.toUpperCase();
    }
    if (metaTempo) metaTempo.textContent = `${cand.tempo_bpm} BPM`;
    if (metaScale) metaScale.textContent = cand.scale_name;
    if (metaDuration) metaDuration.textContent = `${cand.duration.toFixed(1)}s`;
    if (metaSourceRatio) {
      const srcPct = cand.score && cand.score.source_usage_ratio !== undefined ? Math.round(cand.score.source_usage_ratio * 100) : 100;
      metaSourceRatio.textContent = `${srcPct}%`;
    }
    if (metaSynthRatio) {
      const synPct = cand.score && cand.score.synthetic_audio_ratio !== undefined ? Math.round(cand.score.synthetic_audio_ratio * 100) : 0;
      metaSynthRatio.textContent = `${synPct}%`;
    }
    if (metaScore) metaScore.textContent = `${cand.score.total} / 100`;
    if (metaSeed) metaSeed.textContent = cand.seed;

    if (resFormDisplay) resFormDisplay.textContent = cand.form;
    if (dnaFormName) dnaFormName.textContent = cand.form;
    if (dnaActiveStems) dnaActiveStems.textContent = Object.values(cand.stems).join(', ');
    if (dnaPaletteSlices) {
      const pal = cand.palette || {};
      const tot = pal.total_slices || 48;
      dnaPaletteSlices.textContent = `${tot} slices (${pal.impacts || 0} impacts, ${pal.pulses || 0} pulses, ${pal.movements || 0} movements, ${pal.ambience || 0} beds)`;
    }

    // Set Audio Player
    const candUrl = buildUrl(cand.audio_url);
    resultAudioPlayer.src = candUrl;
    if (btnDownload) {
      btnDownload.href = candUrl;
      btnDownload.download = `unnecessaryfm_${currentResultData.job_id}_cand${letter}.wav`;
    }
    if (playerBtnDownload) {
      playerBtnDownload.href = candUrl;
      playerBtnDownload.download = `unnecessaryfm_${currentResultData.job_id}_cand${letter}.wav`;
    }

    // Update Bottom Player & Right Panel
    const trackName = `Candidate ${letter} ${cand.is_winner ? '[Winner]' : ''}`;
    if (playerTrackTitle) playerTrackTitle.textContent = trackName;
    if (playerTrackSubtitle) playerTrackSubtitle.textContent = `${cand.scale_name} • ${cand.tempo_bpm} BPM • ${cand.form}`;
    if (rightPanelTrackTitle) rightPanelTrackTitle.textContent = trackName;
    if (rightPanelTrackSub) rightPanelTrackSub.textContent = `Procedural Composition • ${cand.tempo_bpm} BPM`;
    if (libTrackTitle) libTrackTitle.textContent = trackName;
    if (libTrackMeta) libTrackMeta.textContent = `${cand.scale_name} • ${cand.tempo_bpm} BPM`;

    // Reset Play Buttons
    resetResultPlayState();

    // Draw Result Waveform in Spotify Green
    drawWaveformFromUrl(candUrl, resWaveformCanvas, '#1ed760');
  }

  // 8. Audio Playback & Interactive Waveform Scrubbing
  if (btnPlaySource) {
    btnPlaySource.addEventListener('click', () => {
      toggleSourcePlayback();
    });
  }

  if (btnPlayResult) {
    btnPlayResult.addEventListener('click', () => {
      toggleResultPlayback();
    });
  }

  // Persistent Player Play/Pause Button
  if (playerBtnPlay) {
    playerBtnPlay.addEventListener('click', () => {
      if (activePlayingTarget === 'source') {
        toggleSourcePlayback();
      } else {
        toggleResultPlayback();
      }
    });
  }

  function toggleSourcePlayback() {
    activePlayingTarget = 'source';
    if (sourceAudioPlayer.paused) {
      resultAudioPlayer.pause();
      resetResultPlayState();
      sourceAudioPlayer.play();
      updatePlayIcons(true, 'source');
    } else {
      sourceAudioPlayer.pause();
      updatePlayIcons(false, 'source');
    }
  }

  function toggleResultPlayback() {
    activePlayingTarget = 'result';
    if (resultAudioPlayer.paused) {
      sourceAudioPlayer.pause();
      if (srcPlayIcon) srcPlayIcon.innerHTML = '&#9658;';
      if (srcPlayText) srcPlayText.textContent = 'Play Source';

      resultAudioPlayer.play();
      updatePlayIcons(true, 'result');
    } else {
      resultAudioPlayer.pause();
      updatePlayIcons(false, 'result');
    }
  }

  function updatePlayIcons(isPlaying, target) {
    if (isPlaying) {
      if (target === 'result') {
        if (resPlayIcon) resPlayIcon.innerHTML = '&#9646;&#9646;';
        if (resPlayText) resPlayText.textContent = 'Pause Composition';
      } else {
        if (srcPlayIcon) srcPlayIcon.innerHTML = '&#9646;&#9646;';
        if (srcPlayText) srcPlayText.textContent = 'Pause Source';
      }
      if (playerPlaySvg) {
        playerPlaySvg.innerHTML = '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
      }
    } else {
      if (target === 'result') {
        if (resPlayIcon) resPlayIcon.innerHTML = '&#9658;';
        if (resPlayText) resPlayText.textContent = 'Play Composition';
      } else {
        if (srcPlayIcon) srcPlayIcon.innerHTML = '&#9658;';
        if (srcPlayText) srcPlayText.textContent = 'Play Source';
      }
      if (playerPlaySvg) {
        playerPlaySvg.innerHTML = '<polygon points="8 5 19 12 8 19 8 5"/>';
      }
    }
  }

  // Source Audio Progress Sync
  sourceAudioPlayer.addEventListener('timeupdate', () => {
    if (activePlayingTarget !== 'source') return;
    const cur = sourceAudioPlayer.currentTime;
    const dur = sourceAudioPlayer.duration || 1;
    if (srcTimeDisplay) srcTimeDisplay.textContent = `${formatTime(cur)} / ${formatTime(dur)}`;
    if (playerCurrentTime) playerCurrentTime.textContent = formatTime(cur);
    if (playerTotalDuration) playerTotalDuration.textContent = formatTime(dur);

    const pct = (cur / dur) * 100;
    if (srcPlayhead) srcPlayhead.style.left = `${pct}%`;
    if (playerSliderFill) playerSliderFill.style.width = `${pct}%`;
    if (playerSliderThumb) playerSliderThumb.style.left = `${pct}%`;
  });

  sourceAudioPlayer.addEventListener('ended', () => {
    if (srcPlayIcon) srcPlayIcon.innerHTML = '&#9658;';
    if (srcPlayText) srcPlayText.textContent = 'Play Source';
    if (srcPlayhead) srcPlayhead.style.left = '0%';
    updatePlayIcons(false, 'source');
  });

  // Result Audio Progress Sync
  resultAudioPlayer.addEventListener('timeupdate', () => {
    if (activePlayingTarget !== 'result') return;
    const cur = resultAudioPlayer.currentTime;
    const dur = resultAudioPlayer.duration || 60;
    if (resTimeDisplay) resTimeDisplay.textContent = `${formatTime(cur)} / ${formatTime(dur)}`;
    if (playerCurrentTime) playerCurrentTime.textContent = formatTime(cur);
    if (playerTotalDuration) playerTotalDuration.textContent = formatTime(dur);

    const pct = (cur / dur) * 100;
    if (resPlayhead) resPlayhead.style.left = `${pct}%`;
    if (playerSliderFill) playerSliderFill.style.width = `${pct}%`;
    if (playerSliderThumb) playerSliderThumb.style.left = `${pct}%`;
  });

  resultAudioPlayer.addEventListener('ended', () => {
    resetResultPlayState();
    if (isLooping) {
      resultAudioPlayer.currentTime = 0;
      resultAudioPlayer.play();
      updatePlayIcons(true, 'result');
    }
  });

  function resetResultPlayState() {
    if (resPlayIcon) resPlayIcon.innerHTML = '&#9658;';
    if (resPlayText) resPlayText.textContent = 'Play Composition';
    if (resPlayhead) resPlayhead.style.left = '0%';
    if (playerPlaySvg) {
      playerPlaySvg.innerHTML = '<polygon points="8 5 19 12 8 19 8 5"/>';
    }
    if (playerSliderFill) playerSliderFill.style.width = '0%';
    if (playerSliderThumb) playerSliderThumb.style.left = '0%';
  }

  // Scrubber Seeking on Waveform Canvases
  if (srcCanvasWrapper) {
    srcCanvasWrapper.addEventListener('click', (e) => {
      const rect = srcCanvasWrapper.getBoundingClientRect();
      const pos = (e.clientX - rect.left) / rect.width;
      if (sourceAudioPlayer.duration) {
        sourceAudioPlayer.currentTime = pos * sourceAudioPlayer.duration;
      }
    });
  }

  if (resCanvasWrapper) {
    resCanvasWrapper.addEventListener('click', (e) => {
      const rect = resCanvasWrapper.getBoundingClientRect();
      const pos = (e.clientX - rect.left) / rect.width;
      if (resultAudioPlayer.duration) {
        resultAudioPlayer.currentTime = pos * resultAudioPlayer.duration;
      }
    });
  }

  // Bottom Player Scrubber Bar Drag & Click
  if (playerSliderTrack) {
    playerSliderTrack.addEventListener('click', (e) => {
      const rect = playerSliderTrack.getBoundingClientRect();
      const pos = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const activePlayer = (activePlayingTarget === 'source') ? sourceAudioPlayer : resultAudioPlayer;
      if (activePlayer.duration) {
        activePlayer.currentTime = pos * activePlayer.duration;
      }
    });
  }

  // Bottom Player Next / Previous Controls
  if (playerBtnNext) {
    playerBtnNext.addEventListener('click', () => {
      if (currentResultData && currentResultData.candidates.length > 0) {
        const nextIdx = (activeCandidateIndex + 1) % currentResultData.candidates.length;
        selectCandidate(nextIdx);
        toggleResultPlayback();
      }
    });
  }

  if (playerBtnPrev) {
    playerBtnPrev.addEventListener('click', () => {
      if (activePlayingTarget === 'result' && sourceAudioPlayer.src) {
        toggleSourcePlayback();
      } else if (currentResultData && currentResultData.candidates.length > 0) {
        const prevIdx = (activeCandidateIndex - 1 + currentResultData.candidates.length) % currentResultData.candidates.length;
        selectCandidate(prevIdx);
        toggleResultPlayback();
      }
    });
  }

  if (playerBtnShuffle) {
    playerBtnShuffle.addEventListener('click', () => {
      if (currentResultData && currentResultData.candidates.length > 0) {
        const randIdx = Math.floor(Math.random() * currentResultData.candidates.length);
        selectCandidate(randIdx);
        showToast(`Switched to Candidate ${String.fromCharCode(65 + randIdx)}`, 'info');
      }
    });
  }

  if (playerBtnLoop) {
    playerBtnLoop.addEventListener('click', () => {
      isLooping = !isLooping;
      playerBtnLoop.classList.toggle('active', isLooping);
      showToast(isLooping ? 'Looping enabled' : 'Looping disabled', 'info');
    });
  }

  if (btnPlayerFav) {
    btnPlayerFav.addEventListener('click', () => {
      btnPlayerFav.classList.toggle('active');
      const isFav = btnPlayerFav.classList.contains('active');
      showToast(isFav ? 'Added composition to favorites!' : 'Removed from favorites', 'info');
    });
  }

  // Volume Control
  if (volumeSliderTrack) {
    volumeSliderTrack.addEventListener('click', (e) => {
      const rect = volumeSliderTrack.getBoundingClientRect();
      const pos = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      setVolume(pos);
    });
  }

  if (playerBtnVolume) {
    playerBtnVolume.addEventListener('click', () => {
      if (isMuted) {
        setVolume(previousVolume || 0.8);
        isMuted = false;
      } else {
        previousVolume = currentVolume;
        setVolume(0);
        isMuted = true;
      }
    });
  }

  function setVolume(val) {
    currentVolume = val;
    sourceAudioPlayer.volume = val;
    resultAudioPlayer.volume = val;
    if (volumeSliderFill) volumeSliderFill.style.width = `${val * 100}%`;

    if (val === 0) {
      if (volumeSvg) volumeSvg.innerHTML = '<path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z"/>';
    } else {
      if (volumeSvg) volumeSvg.innerHTML = '<path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>';
    }
  }

  // Right Panel Toggle
  if (playerBtnTogglePanel) {
    playerBtnTogglePanel.addEventListener('click', () => {
      if (spotifyShell) spotifyShell.classList.toggle('right-panel-closed');
    });
  }

  if (btnCloseRightPanel) {
    btnCloseRightPanel.addEventListener('click', () => {
      if (spotifyShell) spotifyShell.classList.add('right-panel-closed');
    });
  }

  if (btnScrollToDNA) {
    btnScrollToDNA.addEventListener('click', () => {
      if (spotifyShell) spotifyShell.classList.remove('right-panel-closed');
      showToast('Acoustic DNA Panel opened on the right!', 'info', 'ACOUSTIC DNA');
    });
  }

  // Library Items Click
  if (libCurrentTrack) {
    libCurrentTrack.addEventListener('click', () => {
      const resSec = document.getElementById('resultsSection');
      if (resSec) resSec.scrollIntoView({ behavior: 'smooth' });
      if (resultAudioPlayer.src) toggleResultPlayback();
    });
  }

  if (libSourceTrack) {
    libSourceTrack.addEventListener('click', () => {
      if (sourceAudioPlayer.src) toggleSourcePlayback();
    });
  }

  // 9. Waveform Canvas Drawing
  async function drawWaveformFromUrl(url, canvas, color) {
    if (!canvas) return;
    try {
      const response = await fetch(buildUrl(url));
      const arrayBuffer = await response.arrayBuffer();
      const ctx = getDecodeAudioContext();
      const audioBuffer = await ctx.decodeAudioData(arrayBuffer);

      const channelData = audioBuffer.getChannelData(0);
      drawWaveform(channelData, canvas, color);
    } catch (err) {
      console.warn("Waveform decode note:", err);
      drawFallbackWaveform(canvas, color);
    }
  }

  function drawWaveform(samples, canvas, color) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Dark sleek canvas backdrop
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, height);

    // Center subtle guideline
    ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.fillRect(0, Math.floor(height / 2), width, 1);

    const numBars = 160;
    const step = Math.floor(samples.length / numBars);
    const barWidth = Math.max(2, Math.floor((width / numBars) * 0.75));

    ctx.fillStyle = color || '#1ed760';

    for (let i = 0; i < numBars; i++) {
      let maxVal = 0;
      const start = i * step;
      for (let j = 0; j < step; j += 4) {
        const val = Math.abs(samples[start + j] || 0);
        if (val > maxVal) maxVal = val;
      }

      const barHeight = Math.max(4, Math.floor(Math.pow(maxVal, 0.72) * (height * 0.88)));
      const x = Math.floor((i / numBars) * width);
      const y = Math.floor((height - barHeight) / 2);

      // Rounded Spotify style sound bars
      ctx.beginPath();
      const radius = 2;
      ctx.roundRect(x, y, barWidth, barHeight, radius);
      ctx.fill();
    }
  }

  function drawFallbackWaveform(canvas, color) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = color || '#1ed760';

    for (let i = 0; i < 120; i++) {
      const barH = 10 + Math.sin(i * 0.18) * (height * 0.35);
      const x = Math.floor(i * (width / 120));
      const y = Math.floor((height - barH) / 2);
      ctx.beginPath();
      ctx.roundRect(x, y, 3, barH, 2);
      ctx.fill();
    }
  }

  function formatTime(seconds) {
    if (!seconds || isNaN(seconds)) return '00:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }

})();
