// Frontend-Anwendungslogik für GuardRAG

// Globale Variablen
let currentUploadedFiles = [];
let chatHistory = [];

// Tab-Funktionalität
function showTab(tabName) {
    // Alle Tabs ausblenden
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Gewählten Tab anzeigen
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
    
    // Tab-spezifische Initialisierung
    if (tabName === 'status') {
        checkSystemStatus();
    } else if (tabName === 'security') {
        loadSecurityStats();
    }
}

// Zeit aktualisieren
function updateTime() {
    document.getElementById('currentTime').textContent = new Date().toLocaleString('de-DE');
}

// Browser-Info setzen
function setBrowserInfo() {
    const browser = navigator.userAgent.includes('Chrome') ? 'Chrome' : 
                   navigator.userAgent.includes('Firefox') ? 'Firefox' : 
                   navigator.userAgent.includes('Safari') ? 'Safari' : 'Unbekannt';
    document.getElementById('browserInfo').textContent = browser;
}

// System-Status prüfen
async function checkSystemStatus() {
    const statusDiv = document.getElementById('systemStatus');
    const backendStatusSpan = document.getElementById('backendStatus');
    
    try {
        statusDiv.innerHTML = '<div class="loading"></div>Prüfe System-Status...';
        
        // Health Check
        const health = await api.getHealth();
        
        // System Status
        const systemStatus = await api.getSystemStatus();
        
        statusDiv.innerHTML = `
            <div class="status">
                ✅ Frontend: Aktiv und verbunden<br>
                ✅ Backend: ${health.status || 'running'}<br>
                📄 Dokumente geladen: ${systemStatus.documents_loaded || 0}<br>
                🛡️ Input Guardrails: ${systemStatus.guardrails_enabled?.input ? 'Aktiv' : 'Inaktiv'}<br>
                🛡️ Output Guardrails: ${systemStatus.guardrails_enabled?.output ? 'Aktiv' : 'Inaktiv'}<br>
                🔄 Letzte Prüfung: ${new Date().toLocaleTimeString()}
            </div>
        `;
        
        backendStatusSpan.innerHTML = '✅ Verbunden (Port 8000)';
        
    } catch (error) {
        console.error('System Status Error:', error);
        statusDiv.innerHTML = `
            <div class="error">
                ❌ Backend-Verbindung fehlgeschlagen<br>
                Fehler: ${error.message}<br>
                💡 Stellen Sie sicher, dass der FastAPI-Server läuft (python main.py)
            </div>
        `;
        backendStatusSpan.innerHTML = '❌ Nicht erreichbar';
    }
}

// COLPALI-Statistiken laden
async function loadColpaliStats() {
    const statusDiv = document.getElementById('colpaliStatus');
    
    try {
        statusDiv.innerHTML = '<div class="loading"></div>Lade COLPALI-Statistiken...';
        
        const stats = await api.getColpaliStats();
        
        statusDiv.innerHTML = `
            <div class="status">
                📊 COLPALI Status: ${stats.status}<br>
                🧠 Instanzen: ${stats.colpali_instances?.total_instances || 0}<br>
                💾 Speicher-Optimierung: ${stats.memory_optimization?.memory_saved || 'Unbekannt'}<br>
                🔄 Letzte Aktualisierung: ${new Date().toLocaleTimeString()}
            </div>
        `;
    } catch (error) {
        console.error('COLPALI Stats Error:', error);
        statusDiv.innerHTML = `
            <div class="error">
                ❌ COLPALI-Statistiken nicht verfügbar<br>
                Fehler: ${error.message}
            </div>
        `;
    }
}

// COLPALI-Cache leeren
async function clearColpaliCache() {
    try {
        const result = await api.clearColpaliCache();
        
        document.getElementById('colpaliStatus').innerHTML = `
            <div class="status">
                ✅ ${result.message}<br>
                🔄 Cache geleert: ${new Date().toLocaleTimeString()}
            </div>
        `;
        
        // Stats neu laden
        setTimeout(loadColpaliStats, 1000);
    } catch (error) {
        console.error('Clear Cache Error:', error);
        document.getElementById('colpaliStatus').innerHTML = `
            <div class="error">
                ❌ Cache konnte nicht geleert werden<br>
                Fehler: ${error.message}
            </div>
        `;
    }
}

// Datei-Upload
async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadStatus = document.getElementById('uploadStatus');
    const uploadProgress = document.getElementById('uploadProgress');
    const uploadProgressFill = document.getElementById('uploadProgressFill');
    
    if (!fileInput.files.length) {
        uploadStatus.innerHTML = '<div class="error">❌ Bitte wählen Sie eine Datei aus</div>';
        return;
    }
    
    const file = fileInput.files[0];
    
    try {
        // UI-Updates
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Lädt hoch...';
        uploadProgress.style.display = 'block';
        uploadProgressFill.style.width = '0%';
        
        // Progress-Animation
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 30;
            if (progress > 90) progress = 90;
            uploadProgressFill.style.width = progress + '%';
        }, 200);
        
        uploadStatus.innerHTML = '<div class="loading"></div>Lade Datei hoch und verarbeite...';
        
        // Upload durchführen
        const result = await api.uploadDocument(file);
        
        // Progress abschließen
        clearInterval(progressInterval);
        uploadProgressFill.style.width = '100%';
        
        uploadStatus.innerHTML = `
            <div class="status">
                ✅ ${result.message}<br>
                📄 Datei: ${file.name}<br>
                📑 Seiten verarbeitet: ${result.pages_processed || 'Unbekannt'}<br>
                🆔 Datei-ID: ${result.file_id}
            </div>
        `;
        
        // Dateiliste aktualisieren
        currentUploadedFiles.push({
            name: file.name,
            id: result.file_id,
            pages: result.pages_processed,
            uploadTime: new Date()
        });
        
        updateFileList();
        
    } catch (error) {
        console.error('Upload Error:', error);
        uploadStatus.innerHTML = `
            <div class="error">
                ❌ Upload fehlgeschlagen<br>
                Fehler: ${error.message}
            </div>
        `;
        uploadProgressFill.style.width = '0%';
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = '📤 Datei hochladen';
        setTimeout(() => {
            uploadProgress.style.display = 'none';
        }, 2000);
    }
}

// Dateiliste aktualisieren
function updateFileList() {
    const fileListDiv = document.getElementById('fileList');
    
    if (currentUploadedFiles.length === 0) {
        fileListDiv.innerHTML = '<em>Keine Dokumente hochgeladen</em>';
        return;
    }
    
    let html = '';
    currentUploadedFiles.forEach(file => {
        html += `
            <div style="margin: 10px 0; padding: 10px; background: #f0f8ff; border-radius: 5px;">
                📄 <strong>${file.name}</strong><br>
                📑 Seiten: ${file.pages || 'Unbekannt'}<br>
                🕒 Hochgeladen: ${file.uploadTime.toLocaleString('de-DE')}
            </div>
        `;
    });
    
    fileListDiv.innerHTML = html;
}

// Alle Dokumente löschen
async function clearAllDocuments() {
    if (!confirm('Möchten Sie wirklich alle Dokumente löschen?')) {
        return;
    }
    
    try {
        const result = await api.clearDocuments();
        
        document.getElementById('fileList').innerHTML = `
            <div class="status">
                ✅ ${result.message}<br>
                🔄 Geleert: ${new Date().toLocaleTimeString()}
            </div>
        `;
        
        currentUploadedFiles = [];
        setTimeout(updateFileList, 2000);
        
    } catch (error) {
        console.error('Clear Documents Error:', error);
        document.getElementById('fileList').innerHTML = `
            <div class="error">
                ❌ Dokumente konnten nicht gelöscht werden<br>
                Fehler: ${error.message}
            </div>
        `;
    }
}

// Chat-Nachricht senden
async function sendMessage() {
    const input = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const chatArea = document.getElementById('chatArea');
    const ragMode = document.getElementById('ragMode').value;
    const collection = document.getElementById('collection').value;
    
    const message = input.value.trim();
    if (!message) return;
    
    // User-Nachricht hinzufügen
    addChatMessage('user', message);
    
    try {
        // UI-Updates
        sendBtn.disabled = true;
        sendBtn.innerHTML = '<div class="loading"></div>';
        input.disabled = true;
        
        // Bot-"Thinking"-Nachricht
        const thinkingId = addChatMessage('bot', '<div class="loading"></div>Verarbeite Ihre Anfrage...');
        
        // RAG-Abfrage durchführen
        let result;
        if (ragMode === 'enhanced') {
            result = await api.enhancedRagQuery(message, collection);
        } else {
            result = await api.ragQuery(message);
        }
        
        // Thinking-Nachricht entfernen
        document.getElementById(thinkingId).remove();
        
        // Bot-Antwort zusammenstellen
        let response = `🤖 ${result.answer || result.sanitized_answer || 'Keine Antwort erhalten'}`;
        
        // Zusätzliche Informationen anhängen
        if (result.confidence) {
            response += `<br><br>📊 Konfidenz: ${(result.confidence * 100).toFixed(1)}%`;
        }
        
        if (result.processing_time) {
            response += `<br>⏱️ Verarbeitungszeit: ${result.processing_time.toFixed(2)}s`;
        }
        
        // Quellen anzeigen
        if (result.sources && result.sources.length > 0) {
            response += '<br><br>📚 Quellen:';
            result.sources.forEach((source, index) => {
                response += `
                    <div class="source">
                        📄 Seite ${source.page_number || 'Unbekannt'} 
                        (Score: ${(source.score * 100).toFixed(1)}%)
                        ${source.text_preview ? '<br>' + source.text_preview : ''}
                    </div>
                `;
            });
        }
        
        // Warnungen anzeigen
        if (result.warnings && result.warnings.length > 0) {
            response += '<br><br>⚠️ Warnungen:';
            result.warnings.forEach(warning => {
                response += `<br>• ${warning}`;
            });
        }
        
        addChatMessage('bot', response);
        
        // Antwort-Info aktualisieren
        updateLastResponseInfo(result);
        
    } catch (error) {
        console.error('Chat Error:', error);
        // Thinking-Nachricht entfernen falls vorhanden
        const thinkingMsg = document.querySelector('.message.bot-message .loading');
        if (thinkingMsg) {
            thinkingMsg.closest('.message').remove();
        }
        
        addChatMessage('bot', `❌ Fehler bei der Verarbeitung:<br>${error.message}`);
    } finally {
        sendBtn.disabled = false;
        sendBtn.innerHTML = '📤';
        input.disabled = false;
        input.value = '';
        input.focus();
    }
}

// Chat-Nachricht hinzufügen
function addChatMessage(sender, message) {
    const chatArea = document.getElementById('chatArea');
    const messageId = 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.id = messageId;
    messageDiv.innerHTML = sender === 'user' ? `👤 ${message}` : message;
    
    chatArea.appendChild(messageDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
    
    // Chat-History speichern
    chatHistory.push({ sender, message, timestamp: new Date() });
    
    return messageId;
}

// Letzte Antwort-Info aktualisieren
function updateLastResponseInfo(result) {
    const infoDiv = document.getElementById('lastResponseInfo');
    
    let info = `📊 Letzte Abfrage-Details:<br>`;
    info += `⏱️ Verarbeitungszeit: ${result.processing_time?.toFixed(2) || 'Unbekannt'}s<br>`;
    info += `🎯 Konfidenz: ${result.confidence ? (result.confidence * 100).toFixed(1) + '%' : 'Unbekannt'}<br>`;
    info += `📄 Quellen gefunden: ${result.sources?.length || 0}<br>`;
    
    if (result.input_validation) {
        info += `🛡️ Input-Validierung: ${result.input_validation.is_valid ? '✅ Gültig' : '❌ Blockiert'}<br>`;
    }
    
    if (result.output_validation) {
        info += `🛡️ Output-Validierung: ${result.output_validation.is_valid ? '✅ Gültig' : '❌ Blockiert'}<br>`;
    }
    
    infoDiv.innerHTML = info;
}

// Chat leeren
function clearChat() {
    document.getElementById('chatArea').innerHTML = `
        <div class="message bot-message">
            🤖 Chat wurde geleert. Stellen Sie Fragen zu Ihren hochgeladenen Dokumenten.
        </div>
    `;
    chatHistory = [];
    document.getElementById('lastResponseInfo').innerHTML = '<em>Noch keine Abfrage durchgeführt</em>';
}

// Enter-Taste im Chat
function handleEnter(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Sicherheitsstatistiken laden
async function loadSecurityStats() {
    const statsDiv = document.getElementById('securityStats');
    
    try {
        statsDiv.innerHTML = '<div class="loading"></div>Lade Sicherheitsstatistiken...';
        
        const stats = await api.getSecurityStats();
        
        let html = `
            <div class="status">
                🚀 System-Status: ${stats.system_status || 'Unbekannt'}<br>
                🔢 Sicherheitsereignisse gesamt: ${stats.total_security_events || 0}<br><br>
        `;
        
        // Input Guardrails Status
        if (stats.input_guardrails) {
            const inputStatus = stats.input_guardrails.status || 'unknown';
            const statusIcon = inputStatus === 'disabled' ? '⚪' : 
                             inputStatus === 'error' ? '❌' : '✅';
            html += `
                ${statusIcon} <strong>Input Guardrails:</strong> ${inputStatus}<br>
                • Blockierte Anfragen: ${stats.input_guardrails.blocked_requests || 0}<br>
                • Letzter Status: ${stats.input_guardrails.last_activity || stats.input_guardrails.reason || 'Keine Daten'}<br><br>
            `;
        }
        
        // Output Guardrails Status
        if (stats.output_guardrails) {
            const outputStatus = stats.output_guardrails.status || 'unknown';
            const statusIcon = outputStatus === 'disabled' ? '⚪' : 
                             outputStatus === 'error' ? '❌' : 
                             outputStatus === 'enabled_but_not_initialized' ? '🟡' : '✅';
            html += `
                ${statusIcon} <strong>Output Guardrails:</strong> ${outputStatus}<br>
                • Blockierte Antworten: ${stats.output_guardrails.blocked_responses || 0}<br>
            `;
            
            if (stats.output_guardrails.error) {
                html += `• Fehler: ${stats.output_guardrails.error}<br>`;
            }
            if (stats.output_guardrails.last_activity) {
                html += `• Letzte Aktivität: ${stats.output_guardrails.last_activity}<br>`;
            }
            html += '<br>';
        }
        
        // Enhanced Guardrails (falls verfügbar)
        if (stats.enhanced_available && stats.enhanced_guardrails) {
            html += `
                ✨ <strong>Enhanced Guardrails:</strong> Verfügbar<br>
                • Erweiterte Statistiken verfügbar<br><br>
            `;
        } else if (stats.enhanced_available === false) {
            html += `⚪ <strong>Enhanced Guardrails:</strong> Nicht verfügbar<br><br>`;
        }
        
        html += `🔄 Letzte Aktualisierung: ${new Date().toLocaleTimeString()}</div>`;
        statsDiv.innerHTML = html;
        
    } catch (error) {
        console.error('Security Stats Error:', error);
        statsDiv.innerHTML = `
            <div class="error">
                ❌ Sicherheitsstatistiken nicht verfügbar<br>
                Fehler: ${error.message}<br><br>
                💡 Mögliche Ursachen:<br>
                • Backend nicht erreichbar<br>
                • GuardRAG Agent nicht initialisiert<br>
                • Guardrails nicht konfiguriert
            </div>
        `;
    }
}

// Sicherheitslevel aktualisieren
async function updateSecurityLevel() {
    const level = document.getElementById('securityLevel').value;
    const statusDiv = document.getElementById('securityUpdateStatus');
    
    try {
        statusDiv.innerHTML = '<div class="loading"></div>Aktualisiere Sicherheitslevel...';
        
        const result = await api.updateSecurityLevel(level);
        
        let statusClass = result.success ? 'status' : 'warning';
        let icon = result.success ? '✅' : '⚠️';
        
        let html = `
            <div class="${statusClass}">
                ${icon} ${result.message}<br>
        `;
        
        if (result.updated_components && result.updated_components.length > 0) {
            html += `� Aktualisierte Komponenten: ${result.updated_components.join(', ')}<br>`;
        }
        
        if (result.note) {
            html += `💡 Hinweis: ${result.note}<br>`;
        }
        
        html += `�🔄 Aktualisiert: ${new Date().toLocaleTimeString()}</div>`;
        
        statusDiv.innerHTML = html;
        
        // Statistiken neu laden
        setTimeout(loadSecurityStats, 1000);
        
    } catch (error) {
        console.error('Update Security Level Error:', error);
        statusDiv.innerHTML = `
            <div class="error">
                ❌ Sicherheitslevel konnte nicht aktualisiert werden<br>
                Fehler: ${error.message}<br><br>
                💡 Hinweis: Standard GuardRAG hat begrenzte Sicherheitseinstellungen
            </div>
        `;
    }
}

// Konkurrenten-Liste aktualisieren
async function updateCompetitors() {
    const competitorText = document.getElementById('competitorList').value;
    const statusDiv = document.getElementById('competitorUpdateStatus');
    
    const competitors = competitorText
        .split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0);
    
    if (competitors.length === 0) {
        statusDiv.innerHTML = '<div class="warning">⚠️ Keine Konkurrenten eingegeben</div>';
        return;
    }
    
    try {
        statusDiv.innerHTML = '<div class="loading"></div>Aktualisiere Konkurrenten-Liste...';
        
        const result = await api.updateCompetitors(competitors);
        
        statusDiv.innerHTML = `
            <div class="status">
                ✅ ${result.message}<br>
                📋 Konkurrenten: ${result.competitors.join(', ')}<br>
                🔄 Aktualisiert: ${new Date().toLocaleTimeString()}
            </div>
        `;
        
    } catch (error) {
        console.error('Update Competitors Error:', error);
        statusDiv.innerHTML = `
            <div class="error">
                ❌ Konkurrenten-Liste konnte nicht aktualisiert werden<br>
                Fehler: ${error.message}
            </div>
        `;
    }
}

// Initialisierung
document.addEventListener('DOMContentLoaded', function() {
    setBrowserInfo();
    updateTime();
    setInterval(updateTime, 1000);
    
    // Initial System-Status laden
    checkSystemStatus();
    
    // Datei-Input Event-Listener
    document.getElementById('fileInput').addEventListener('change', function() {
        if (this.files.length > 0) {
            document.getElementById('uploadStatus').innerHTML = `
                <div class="status">📄 Datei ausgewählt: ${this.files[0].name}</div>
            `;
        }
    });
});
