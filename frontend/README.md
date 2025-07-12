# GuardRAG Frontend

Ein modernes HTML/JavaScript-Frontend für das GuardRAG-System mit vollständiger FastAPI-Integration.

## 📁 Struktur

```
frontend/
├── index.html      # Haupt-HTML-Datei mit UI
├── api.js          # FastAPI-Integration und API-Aufrufe
└── app.js          # Frontend-Anwendungslogik
```

## 🚀 Schnellstart

1. **Backend starten** (Terminal 1):
   ```bash
   python main.py
   ```

2. **Frontend starten** (Terminal 2):
   ```bash
   python start_frontend.py
   ```

3. **Browser öffnen**:
   - Frontend: http://localhost:8080
   - Backend-API: http://localhost:8000

## 🎯 Features

### 📊 System-Status
- Live-Überwachung des FastAPI-Backends
- COLPALI-Modell Statistiken
- Systemgesundheit und Komponenten-Status

### 📁 Datei-Upload
- Drag & Drop Upload (PDF, DOCX, TXT, MD, HTML)
- Progress-Anzeige mit Echtzeit-Updates
- Automatische Dokumentenverarbeitung mit COLPALI
- Datei-Management und Löschfunktionen

### 💬 RAG-Chat
- **Standard RAG**: Basis-Funktionalität mit Guardrails
- **Enhanced RAG**: Deutsche PII-Erkennung und erweiterte Sicherheit
- Quellen-Anzeige mit Relevanz-Scores
- Verarbeitungszeit und Konfidenz-Anzeigen
- Chat-Historie und Antwort-Details

### 🔒 Sicherheit
- Live-Sicherheitsstatistiken
- Einstellbare Sicherheitslevel (Niedrig/Mittel/Hoch)
- Konkurrenten-Filter Management
- Input/Output Guardrails Überwachung

## 🔧 API-Integration

Das Frontend kommuniziert vollständig mit der FastAPI über folgende Endpoints:

### Basis-Funktionen
- `GET /health` - System-Gesundheit
- `GET /system-status` - Detaillierter System-Status
- `GET /enhanced-system-status` - Erweiterte Statistiken

### Dokumenten-Management
- `POST /upload-document` - Datei-Upload
- `DELETE /clear-documents` - Alle Dokumente löschen

### RAG-Abfragen
- `POST /rag-query` - Standard RAG-Abfrage
- `POST /enhanced-rag-query` - Enhanced RAG mit PII-Schutz

### Sicherheit
- `GET /security/stats` - Sicherheitsstatistiken
- `POST /security/update-level` - Sicherheitslevel ändern
- `POST /update-competitors` - Konkurrenten-Liste aktualisieren

### COLPALI-Management
- `GET /colpali-stats` - Modell-Statistiken
- `POST /colpali-clear-cache` - Cache leeren

## 🎨 UI-Features

- **Responsive Design**: Funktioniert auf Desktop und Mobil
- **Tab-Navigation**: Übersichtliche Kategorisierung
- **Live-Updates**: Automatische Status-Aktualisierungen
- **Progress-Anzeigen**: Visuelles Feedback bei längeren Operationen
- **Error-Handling**: Benutzerfreundliche Fehlermeldungen
- **Dark/Light Theme**: Moderne Farbgebung

## 🔧 Konfiguration

### Frontend-Server (start_frontend.py)
- **Port**: 8080 (änderbar in start_frontend.py)
- **Host**: 127.0.0.1
- **CORS**: Automatisch konfiguriert

### API-Verbindung (api.js)
- **Backend-URL**: http://localhost:8000 (änderbar in api.js)
- **Timeout**: 5 Sekunden für Health-Checks
- **Auto-Retry**: Bei Verbindungsfehlern

## 🚨 Fehlerbehebung

### Backend nicht erreichbar
```
⚠️ FastAPI Backend nicht erreichbar (Port 8000)
```
**Lösung**: Starten Sie das Backend mit `python main.py`

### Port bereits belegt
```
❌ Port 8080 ist bereits belegt
```
**Lösung**: Ändern Sie den Port in `start_frontend.py` oder beenden Sie andere Anwendungen

### Upload-Fehler
- Prüfen Sie unterstützte Dateiformate: PDF, DOCX, TXT, MD, HTML
- Stellen Sie sicher, dass das Backend läuft
- Überprüfen Sie die Datei-Größe (Standard-Limits beachten)

## 🔄 Entwicklung

### Lokale Änderungen
1. Änderungen in `frontend/` vornehmen
2. Browser-Cache leeren (Strg+F5)
3. Seite neu laden

### API-Debugging
- Browser-Entwicklertools öffnen (F12)
- Network-Tab für API-Aufrufe prüfen
- Console für JavaScript-Fehler überwachen

## 📱 Browser-Kompatibilität

- ✅ Chrome 80+
- ✅ Firefox 75+
- ✅ Safari 13+
- ✅ Edge 80+

## 🔐 Sicherheitshinweise

- Frontend läuft nur lokal (127.0.0.1)
- CORS ist für lokale Entwicklung konfiguriert
- Für Produktionsumgebung zusätzliche Sicherheitsmaßnahmen erforderlich
- Keine sensiblen Daten im Browser-Cache
