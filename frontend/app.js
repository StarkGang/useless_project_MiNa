/**
 * TheUnnecessaryFM — Frontend Application Logic
 * Pure DSP / Procedural Music Generation Interactive Studio
 */

(() => {
  // DOM Elements
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('audioFileInput');
  const dropZoneContent = document.getElementById('dropZoneContent');
  const selectedFilePill = document.getElementById('selectedFilePill');
  const selectedFileName = document.getElementById('selectedFileName');
  const selectedFileSize = document.getElementById('selectedFileSize');
  const btnRemoveFile = document.getElementById('btnRemoveFile');

  const btnRecord = document.getElementById('btnRecord');
  const recordBtnText = document.getElementById('recordBtnText');
  const recTimer = document.getElementById('recTimer');
  const recTimeDisplay = document.getElementById('recTimeDisplay');
  const liveMicCanvas = document.getElementById('liveMicCanvas');

  const beatSelect = document.getElementById('beatSelect');
  const energySelect = document.getElementById('energySelect');
  const seedInput = document.getElementById('seedInput');
  const btnRandomizeSeed = document.getElementById('btnRandomizeSeed');
  const btnGenerate = document.getElementById('btnGenerate');

  const emptyState = document.getElementById('emptyState');
  const processingState = document.getElementById('processingState');
  const resultsDisplay = document.getElementById('resultsDisplay');
  const progressPercent = document.getElementById('progressPercent');
  const progressStage = document.getElementById('progressStage');
  const progressBarFill = document.getElementById('progressBarFill');

  // Candidate tabs & banners
  const candidateTabs = document.getElementById('candidateTabs');
  const metaTempo = document.getElementById('metaTempo');
  const metaScale = document.getElementById('metaScale');
  const metaDuration = document.getElementById('metaDuration');
  const metaSourceRatio = document.getElementById('metaSourceRatio');
  const metaSynthRatio = document.getElementById('metaSynthRatio');
  const metaScore = document.getElementById('metaScore');
  const metaSeed = document.getElementById('metaSeed');

  // Waveform displays
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

  // DNA Panel
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

  const sourceAudioPlayer = document.getElementById('sourceAudioPlayer');
  const resultAudioPlayer = document.getElementById('resultAudioPlayer');

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

  // Web Audio Context for Waveform Decoding
  let decodeAudioCtx = null;
  function getDecodeAudioContext() {
    if (!decodeAudioCtx) {
      decodeAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    return decodeAudioCtx;
  }

  // 1. File Upload & Drag-and-Drop
  dropZone.addEventListener('click', (e) => {
    if (e.target !== btnRemoveFile) {
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

  btnRemoveFile.addEventListener('click', (e) => {
    e.stopPropagation();
    resetFileInput();
  });

  function handleFileSelected(file) {
    currentFile = file;
    selectedFileName.textContent = file.name;
    selectedFileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
    dropZoneContent.classList.add('hidden');
    selectedFilePill.classList.remove('hidden');
    btnGenerate.disabled = false;
  }

  function resetFileInput() {
    currentFile = null;
    fileInput.value = '';
    selectedFilePill.classList.add('hidden');
    dropZoneContent.classList.remove('hidden');
    if (!currentJobId) {
      btnGenerate.disabled = true;
    }
  }

  // 2. Microphone Recording (5 - 60s)
  btnRecord.addEventListener('click', async () => {
    if (!isRecording) {
      startMicRecording();
    } else {
      stopMicRecording();
    }
  });

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
        const recordedFile = new File([audioBlob], `mic_recording_${Date.now()}.webm`, { type: 'audio/webm' });
        handleFileSelected(recordedFile);
        cleanupMicStream();
      };

      mediaRecorder.start(100);
      isRecording = true;
      btnRecord.classList.add('recording');
      recordBtnText.textContent = 'Stop Recording';
      recTimer.classList.remove('hidden');
      liveMicCanvas.classList.remove('hidden');

      recordStartTime = Date.now();
      updateRecordTimer();
      recordTimerInterval = setInterval(updateRecordTimer, 500);

      // Start live mic visualizer
      startLiveMicVisualizer(micStream);

      // Auto stop at 60s
      setTimeout(() => {
        if (isRecording) {
          stopMicRecording();
        }
      }, 60000);

    } catch (err) {
      alert('Microphone access was denied or not available: ' + err.message);
    }
  }

  function stopMicRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
    }
    isRecording = false;
    btnRecord.classList.remove('recording');
    recordBtnText.textContent = 'Record Mic (5–60s)';
    recTimer.classList.add('hidden');
    liveMicCanvas.classList.add('hidden');
    if (recordTimerInterval) {
      clearInterval(recordTimerInterval);
      recordTimerInterval = null;
    }
  }

  function updateRecordTimer() {
    const elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
    const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const s = String(elapsed % 60).padStart(2, '0');
    recTimeDisplay.textContent = `${m}:${s}`;
  }

  function startLiveMicVisualizer(stream) {
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

      canvasCtx.fillStyle = 'rgba(0, 0, 0, 0.4)';
      canvasCtx.fillRect(0, 0, liveMicCanvas.width, liveMicCanvas.height);

      const barWidth = (liveMicCanvas.width / dataArray.length) * 2;
      let x = 0;
      for (let i = 0; i < dataArray.length; i++) {
        const barHeight = (dataArray[i] / 255) * liveMicCanvas.height;
        canvasCtx.fillStyle = '#ef4444';
        canvasCtx.fillRect(x, liveMicCanvas.height - barHeight, barWidth - 1, barHeight);
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
  btnRandomizeSeed.addEventListener('click', () => {
    seedInput.value = Math.floor(Math.random() * 9000000000) + 1000000000;
  });

  // 4. Generate Music Request
  btnGenerate.addEventListener('click', () => {
    submitMusicGeneration();
  });

  btnRegenerateNew.addEventListener('click', () => {
    seedInput.value = Math.floor(Math.random() * 9000000000) + 1000000000;
    submitMusicGeneration();
  });

  async function submitMusicGeneration() {
    if (!currentFile && !currentJobId) return;

    showProcessingState("Initializing DSP pipeline & decoding audio...");

    const formData = new FormData();
    if (currentFile) {
      formData.append('file', currentFile);
    } else if (currentJobId) {
      formData.append('existing_job_id', currentJobId);
    }

    formData.append('beat_preference', beatSelect.value);
    formData.append('energy_preference', energySelect.value);
    if (seedInput.value.trim()) {
      formData.append('seed', seedInput.value.trim());
    }

    try {
      const response = await fetch('/api/generate', {
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
      alert("Error: " + err.message);
      hideProcessingState();
    }
  }

  function showProcessingState(stageText) {
    emptyState.classList.add('hidden');
    resultsDisplay.classList.add('hidden');
    processingState.classList.remove('hidden');
    progressPercent.textContent = '10%';
    progressBarFill.style.width = '10%';
    progressStage.textContent = stageText || 'Processing...';
    btnGenerate.disabled = true;
  }

  function hideProcessingState() {
    processingState.classList.add('hidden');
    btnGenerate.disabled = false;
  }

  // 5. Job Status Polling
  function pollJobStatus(jobId) {
    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/status/${jobId}`);
        if (!res.ok) throw new Error('Status check failed');

        const statusData = await res.json();
        const pct = Math.max(10, statusData.progress || 10);
        progressPercent.textContent = `${pct}%`;
        progressBarFill.style.width = `${pct}%`;
        progressStage.textContent = statusData.stage || 'Synthesizing audio...';

        if (statusData.status === 'completed') {
          clearInterval(pollInterval);
          loadFinalResults(jobId);
        } else if (statusData.status === 'failed') {
          clearInterval(pollInterval);
          alert('Composition failed: ' + (statusData.error || 'Unknown DSP error'));
          hideProcessingState();
        }
      } catch (e) {
        console.error("Polling error:", e);
      }
    }, 800);
  }

  // 6. Load & Render Results
  async function loadFinalResults(jobId) {
    try {
      const res = await fetch(`/api/result/${jobId}`);
      if (!res.ok) throw new Error('Failed to retrieve result');

      currentResultData = await res.json();
      renderCompositionUI(currentResultData);

    } catch (err) {
      alert("Error loading result: " + err.message);
      hideProcessingState();
    }
  }

  function renderCompositionUI(result) {
    processingState.classList.add('hidden');
    emptyState.classList.add('hidden');
    resultsDisplay.classList.remove('hidden');
    btnGenerate.disabled = false;

    // Populate DNA Panel
    const analysis = result.source.analysis || {};
    catPill.textContent = analysis.classification || 'MIXED';

    const rDens = analysis.rhythmic_density || 0;
    meterRhythm.style.width = `${Math.min(100, rDens * 35)}%`;
    valRhythm.textContent = rDens.toFixed(2);

    const harm = analysis.harmonicity || 0.5;
    meterHarmonic.style.width = `${Math.min(100, harm * 100)}%`;
    valHarmonic.textContent = harm.toFixed(2);

    const text = analysis.noisiness || 0.5;
    meterTexture.style.width = `${Math.min(100, text * 100)}%`;
    valTexture.textContent = text.toFixed(2);

    const bright = analysis.brightness || 0.5;
    meterEnergy.style.width = `${Math.min(100, bright * 100)}%`;
    valEnergy.textContent = bright.toFixed(2);

    dnaDetectedPitch.textContent = analysis.has_reliable_pitch ? `${analysis.nearest_note}` : 'None (Tuned Resonators Applied)';
    dnaClassReason.textContent = (analysis.reasons && analysis.reasons.length > 0) ? analysis.reasons[0] : 'Balanced harmonic & transient mix';

    // Source Audio Setup
    srcNameDisplay.textContent = result.source.filename || 'source.wav';
    sourceAudioPlayer.src = result.source.audio_url;
    drawWaveformFromUrl(result.source.audio_url, srcWaveformCanvas, '#94a3b8', '#64748b');

    // Build Candidate Tabs
    candidateTabs.innerHTML = '';
    result.candidates.forEach((cand, idx) => {
      const tab = document.createElement('button');
      tab.type = 'button';
      tab.className = `cand-tab ${idx === 0 ? 'active' : ''}`;
      tab.dataset.index = idx;
      tab.innerHTML = `Candidate ${String.fromCharCode(65 + idx)} ${cand.is_winner ? '<span class="tag-winner">WINNER</span>' : ''}`;
      tab.addEventListener('click', () => selectCandidate(idx));
      candidateTabs.appendChild(tab);
    });

    // Select Winner by default
    selectCandidate(0);
  }

  function selectCandidate(index) {
    if (!currentResultData || !currentResultData.candidates[index]) return;
    activeCandidateIndex = index;

    // Update active tab styles
    const tabs = candidateTabs.querySelectorAll('.cand-tab');
    tabs.forEach((t, idx) => {
      t.classList.toggle('active', idx === index);
    });

    const cand = currentResultData.candidates[index];

    // Update Banner
    metaTempo.textContent = `${cand.tempo_bpm} BPM`;
    metaScale.textContent = cand.scale_name;
    metaDuration.textContent = `${cand.duration.toFixed(1)}s`;
    if (metaSourceRatio) {
      const srcPct = cand.score && cand.score.source_usage_ratio !== undefined ? Math.round(cand.score.source_usage_ratio * 100) : 100;
      metaSourceRatio.textContent = `${srcPct}%`;
    }
    if (metaSynthRatio) {
      const synPct = cand.score && cand.score.synthetic_audio_ratio !== undefined ? Math.round(cand.score.synthetic_audio_ratio * 100) : 0;
      metaSynthRatio.textContent = `${synPct}%`;
    }
    metaScore.textContent = `${cand.score.total} / 100`;
    metaSeed.textContent = cand.seed;

    resFormDisplay.textContent = cand.form;
    dnaFormName.textContent = cand.form;
    dnaActiveStems.textContent = Object.values(cand.stems).join(', ');
    if (dnaPaletteSlices) {
      const pal = cand.palette || {};
      const tot = pal.total_slices || 48;
      dnaPaletteSlices.textContent = `${tot} slices (${pal.impacts || 0} impacts, ${pal.pulses || 0} pulses, ${pal.movements || 0} movements, ${pal.ambience || 0} beds)`;
    }

    // Set Audio Player
    resultAudioPlayer.src = cand.audio_url;
    btnDownload.href = cand.audio_url;
    btnDownload.download = `unnecessaryfmn_${currentResultData.job_id}_cand${index}.wav`;

    // Reset Play Buttons
    resetResultPlayState();

    // Draw Result Waveform
    drawWaveformFromUrl(cand.audio_url, resWaveformCanvas, '#06b6d4', '#10b981');
  }

  // 7. Audio Playback & Interactive Waveform Scrubbing
  btnPlaySource.addEventListener('click', () => {
    if (sourceAudioPlayer.paused) {
      resultAudioPlayer.pause();
      resetResultPlayState();
      sourceAudioPlayer.play();
      srcPlayIcon.innerHTML = '&#9646;&#9646;';
      srcPlayText.textContent = 'Pause Source';
    } else {
      sourceAudioPlayer.pause();
      srcPlayIcon.innerHTML = '&#9658;';
      srcPlayText.textContent = 'Play Source';
    }
  });

  sourceAudioPlayer.addEventListener('timeupdate', () => {
    const cur = sourceAudioPlayer.currentTime;
    const dur = sourceAudioPlayer.duration || 1;
    srcTimeDisplay.textContent = `${formatTime(cur)} / ${formatTime(dur)}`;
    const pct = (cur / dur) * 100;
    srcPlayhead.style.transform = `translateX(${pct}%)`;
    srcPlayhead.style.left = `${pct}%`;
  });

  sourceAudioPlayer.addEventListener('ended', () => {
    srcPlayIcon.innerHTML = '&#9658;';
    srcPlayText.textContent = 'Play Source';
    srcPlayhead.style.left = '0%';
  });

  btnPlayResult.addEventListener('click', () => {
    if (resultAudioPlayer.paused) {
      sourceAudioPlayer.pause();
      srcPlayIcon.innerHTML = '&#9658;';
      srcPlayText.textContent = 'Play Source';

      resultAudioPlayer.play();
      resPlayIcon.innerHTML = '&#9646;&#9646;';
      resPlayText.textContent = 'Pause Composition';
    } else {
      resultAudioPlayer.pause();
      resPlayIcon.innerHTML = '&#9658;';
      resPlayText.textContent = 'Play Composition';
    }
  });

  resultAudioPlayer.addEventListener('timeupdate', () => {
    const cur = resultAudioPlayer.currentTime;
    const dur = resultAudioPlayer.duration || 60;
    resTimeDisplay.textContent = `${formatTime(cur)} / ${formatTime(dur)}`;
    const pct = (cur / dur) * 100;
    resPlayhead.style.left = `${pct}%`;
  });

  resultAudioPlayer.addEventListener('ended', () => {
    resetResultPlayState();
  });

  function resetResultPlayState() {
    resPlayIcon.innerHTML = '&#9658;';
    resPlayText.textContent = 'Play Composition';
    resPlayhead.style.left = '0%';
  }

  // Click-to-seek on canvas
  srcCanvasWrapper.addEventListener('click', (e) => {
    const rect = srcCanvasWrapper.getBoundingClientRect();
    const pos = (e.clientX - rect.left) / rect.width;
    if (sourceAudioPlayer.duration) {
      sourceAudioPlayer.currentTime = pos * sourceAudioPlayer.duration;
    }
  });

  resCanvasWrapper.addEventListener('click', (e) => {
    const rect = resCanvasWrapper.getBoundingClientRect();
    const pos = (e.clientX - rect.left) / rect.width;
    if (resultAudioPlayer.duration) {
      resultAudioPlayer.currentTime = pos * resultAudioPlayer.duration;
    }
  });

  // 8. Waveform Canvas Drawing
  async function drawWaveformFromUrl(url, canvas, color1, color2) {
    try {
      const response = await fetch(url);
      const arrayBuffer = await response.arrayBuffer();
      const ctx = getDecodeAudioContext();
      const audioBuffer = await ctx.decodeAudioData(arrayBuffer);

      const channelData = audioBuffer.getChannelData(0);
      drawWaveform(channelData, canvas, color1, color2);
    } catch (err) {
      console.warn("Could not render decoded waveform:", err);
      // Fallback synthetic wave
      drawFallbackWaveform(canvas, color1);
    }
  }

  function drawWaveform(samples, canvas, color1, color2) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    const numBars = 140;
    const step = Math.floor(samples.length / numBars);
    const barWidth = (width / numBars) * 0.7;

    const grad = ctx.createLinearGradient(0, 0, 0, height);
    grad.addColorStop(0, color1);
    grad.addColorStop(1, color2);
    ctx.fillStyle = grad;

    for (let i = 0; i < numBars; i++) {
      let maxVal = 0;
      const start = i * step;
      for (let j = 0; j < step; j += 4) {
        const val = Math.abs(samples[start + j] || 0);
        if (val > maxVal) maxVal = val;
      }

      // Nonlinear boost for visual clarity
      const barHeight = Math.max(4, Math.pow(maxVal, 0.7) * (height * 0.85));
      const x = (i / numBars) * width;
      const y = (height - barHeight) / 2;

      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, barHeight, 2);
      ctx.fill();
    }
  }

  function drawFallbackWaveform(canvas, color) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = color;

    for (let i = 0; i < 100; i++) {
      const barH = 10 + Math.sin(i * 0.2) * 20;
      ctx.fillRect(i * (width / 100), (height - barH) / 2, 3, barH);
    }
  }

  function formatTime(seconds) {
    if (!seconds || isNaN(seconds)) return '00:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }

})();
