document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const micBtn = document.getElementById("mic-btn");
    const recordingStatus = document.getElementById("recording-status");
    const recordingTimer = document.getElementById("recording-timer");
    const queryForm = document.getElementById("query-form");
    const queryInput = document.getElementById("query-input");
    const relevanceBadge = document.getElementById("relevance-badge");
    const transcriptContainer = document.getElementById("transcript-container");
    const transcriptText = document.getElementById("transcript-text");
    const answerDisplay = document.getElementById("answer-text-display");
    const loader = document.getElementById("loader");
    const loaderStatus = document.getElementById("loader-status");
    const sourcesCount = document.getElementById("sources-count");
    const sourcesList = document.getElementById("sources-list");
    
    // Latency Elements
    const latencyStt = document.getElementById("latency-stt");
    const latencyRetrieval = document.getElementById("latency-retrieval");
    const latencyGen = document.getElementById("latency-gen");
    const latencyTotal = document.getElementById("latency-total");
    
    // Recorder State variables
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let timerInterval = null;
    let secondsElapsed = 0;
    
    // ----------------------------------------------------
    // Timer helper functions
    // ----------------------------------------------------
    function startTimer() {
        secondsElapsed = 0;
        recordingTimer.textContent = "00:00";
        recordingTimer.classList.remove("hidden");
        
        timerInterval = setInterval(() => {
            secondsElapsed++;
            const mins = String(Math.floor(secondsElapsed / 60)).padStart(2, '0');
            const secs = String(secondsElapsed % 60).padStart(2, '0');
            recordingTimer.textContent = `${mins}:${secs}`;
        }, 1000);
    }
    
    function stopTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
    }
    
    // ----------------------------------------------------
    // Recording voice audio
    // ----------------------------------------------------
    async function startRecording() {
        audioChunks = [];
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Standard constraints for ElevenLabs Scribe v2 compatibility (default browser MIME type)
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };
            
            mediaRecorder.onstop = async () => {
                // Combine audio chunks into a blob
                const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
                stream.getTracks().forEach(track => track.stop()); // Stop mic hardware
                
                // Upload and query RAG voice API
                await submitVoiceQuery(audioBlob);
            };
            
            mediaRecorder.start();
            isRecording = true;
            micBtn.classList.add("recording");
            recordingStatus.textContent = "Recording... Click mic again to stop & query";
            startTimer();
        } catch (err) {
            console.error("Error accessing microphone:", err);
            recordingStatus.textContent = "Error accessing microphone. Check permissions.";
            alert("Could not access microphone. Please check site permissions.");
        }
    }
    
    function stopRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            isRecording = false;
            micBtn.classList.remove("recording");
            recordingStatus.textContent = "Uploading recording...";
            stopTimer();
        }
    }
    
    micBtn.addEventListener("click", () => {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });
    
    // ----------------------------------------------------
    // API Submit query functions
    // ----------------------------------------------------
    
    function showLoading(statusMsg) {
        loaderStatus.textContent = statusMsg;
        loader.classList.remove("hidden");
        answerDisplay.classList.add("hidden");
        
        // Reset latency and badges during load
        relevanceBadge.textContent = "Processing";
        relevanceBadge.className = "badge";
        latencyStt.textContent = "-";
        latencyRetrieval.textContent = "-";
        latencyGen.textContent = "-";
        latencyTotal.textContent = "-";
    }
    
    function hideLoading() {
        loader.classList.add("hidden");
        answerDisplay.classList.remove("hidden");
    }
    
    async function submitTextQuery(query) {
        showLoading("Searching vector index...");
        transcriptContainer.classList.add("hidden");
        
        const selectedLang = document.getElementById("lang-select").value;
        const useLLM = document.getElementById("llm-toggle").checked;
        
        try {
            const response = await fetch("/api/query", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ query: query, k: 5, language: selectedLang, use_llm: useLLM })
            });
            
            if (!response.ok) {
                throw new Error(`Server returned error: ${response.status}`);
            }
            
            const data = await response.json();
            updateUI(data, false);
        } catch (err) {
            console.error("Error submitting text query:", err);
            renderError(err.message);
        } finally {
            hideLoading();
        }
    }
    
    async function submitVoiceQuery(audioBlob) {
        showLoading("Transcribing voice recording...");
        
        const selectedLang = document.getElementById("lang-select").value;
        const useLLM = document.getElementById("llm-toggle").checked;
        
        try {
            const formData = new FormData();
            // Provide filename so FastAPI can parse extension correctly
            formData.append("file", audioBlob, "query_recording.webm");
            formData.append("k", 5);
            formData.append("language", selectedLang);
            formData.append("use_llm", useLLM);
            
            const response = await fetch("/api/query-voice", {
                method: "POST",
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Server returned error: ${response.status}`);
            }
            
            const data = await response.json();
            updateUI(data, true);
        } catch (err) {
            console.error("Error submitting voice query:", err);
            renderError(err.message);
        } finally {
            hideLoading();
            recordingStatus.textContent = "Click mic to start recording";
        }
    }
    
    // ----------------------------------------------------
    // UI Rendering functions
    // ----------------------------------------------------
    
    function updateUI(data, isVoice) {
        // 1. Render transcript if voice
        if (isVoice && data.transcript) {
            transcriptText.textContent = data.transcript;
            transcriptContainer.classList.remove("hidden");
        } else {
            transcriptContainer.classList.add("hidden");
        }
        
        // 2. Render answer
        answerDisplay.innerHTML = `<div class="answer-text">${data.answer}</div>`;
        
        // 3. Render relevance badge
        if (data.relevance_passed) {
            relevanceBadge.textContent = "Grounded";
            relevanceBadge.className = "badge success";
        } else {
            relevanceBadge.textContent = "Fallback Active";
            relevanceBadge.className = "badge error";
        }
        
        // 4. Render latency times
        if (isVoice && data.latency_ms.stt) {
            latencyStt.textContent = `${data.latency_ms.stt.toFixed(0)} ms`;
        } else {
            latencyStt.textContent = "N/A";
        }
        
        latencyRetrieval.textContent = `${data.latency_ms.retrieval.toFixed(0)} ms`;
        latencyGen.textContent = `${data.latency_ms.generation.toFixed(0)} ms`;
        latencyTotal.textContent = `${data.latency_ms.total_rag.toFixed(0)} ms`;
        
        // 5. Render sources
        sourcesCount.textContent = `${data.sources.length} Sources`;
        
        if (data.sources.length === 0) {
            sourcesList.innerHTML = `<div class="sources-placeholder">No sources retrieved.</div>`;
            return;
        }
        
        sourcesList.innerHTML = "";
        data.sources.forEach(src => {
            const scorePct = (src.score * 100).toFixed(0);
            const isSelected = src.metadata.is_selected === 1 || src.metadata.is_selected === "1";
            
            const sourceCard = document.createElement("div");
            sourceCard.className = "source-item";
            sourceCard.innerHTML = `
                <div class="source-meta">
                    <span class="source-id"><i class="fa-solid fa-file-lines"></i> ${src.document_id}</span>
                    <div class="source-score-container">
                        <span>Score: ${(src.score).toFixed(4)}</span>
                        <div class="score-meter" title="Match Score: ${scorePct}%">
                            <div class="score-fill" style="width: ${scorePct}%"></div>
                        </div>
                    </div>
                </div>
                <div class="source-text">${src.text}</div>
            `;
            sourcesList.appendChild(sourceCard);
        });
    }
    
    function renderError(errMsg) {
        answerDisplay.innerHTML = `
            <div class="answer-text" style="color: var(--status-error);">
                <i class="fa-solid fa-triangle-exclamation"></i> 
                <strong>Error processing request:</strong> ${errMsg}
            </div>
        `;
        relevanceBadge.textContent = "Error";
        relevanceBadge.className = "badge error";
    }
    
    // ----------------------------------------------------
    // Event listeners & Startup
    // ----------------------------------------------------
    queryForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const text = queryInput.value.trim();
        if (text) {
            submitTextQuery(text);
            queryInput.value = "";
        }
    });

    async function loadEvaluationStats() {
        try {
            const response = await fetch("/api/evaluation-results");
            if (!response.ok) {
                console.warn("Evaluation results not found on server.");
                return;
            }
            const data = await response.json();
            
            // Format timestamp nicely
            document.getElementById("eval-timestamp").textContent = `Tested: ${data.timestamp}`;
            
            // Retrieval Stats
            document.getElementById("eval-retrieval-p50").textContent = `${data.retrieval.p50_ms.toFixed(2)} ms`;
            document.getElementById("eval-retrieval-p70").textContent = `${data.retrieval.p70_ms.toFixed(2)} ms`;
            document.getElementById("eval-retrieval-p100").textContent = `${data.retrieval.p100_ms.toFixed(2)} ms`;
            document.getElementById("eval-recall").textContent = `${data.retrieval.recall_accuracy_percent.toFixed(2)}%`;
            
            // RAG Stats
            document.getElementById("eval-rag-p50").textContent = `${(data.rag.total.p50_ms / 1000).toFixed(2)} s`;
            document.getElementById("eval-rag-p70").textContent = `${(data.rag.total.p70_ms / 1000).toFixed(2)} s`;
            document.getElementById("eval-rag-p100").textContent = `${(data.rag.total.p100_ms / 1000).toFixed(2)} s`;
            document.getElementById("eval-count").textContent = `${data.rag.total_run} Queries`;
        } catch (err) {
            console.error("Error loading evaluation stats:", err);
        }
    }

    // Load overall benchmarks immediately on load
    loadEvaluationStats();
});
