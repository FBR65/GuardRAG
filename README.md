# GuardRAG - Secure Document RAG System

Ein fortschrittliches RAG-System (Retrieval-Augmented Generation) auf Basis von COLPALI mit integrierten Guardrails für sichere und vertrauenswürdige Dokumentenverarbeitung.

## 🎯 Überblick

GuardRAG kombiniert modernste Vision-Language-Modelle (COLPALI) mit umfassenden Sicherheitsmechanismen (Guardrails), um eine robuste und vertrauenswürdige Lösung für die Analyse wissenschaftlicher Dokumente zu bieten.

### Kernfunktionen

- **COLPALI-Integration**: Effiziente visuelle Dokumentenretrieval mit dem `vidore/colqwen2.5-v0.2` Modell
- **Input Guardrails**: Validierung und Filterung eingehender Anfragen
- **Output Guardrails**: Überprüfung generierter Antworten auf Faktentreue und Sicherheit
- **File Upload**: Unterstützung für PDF, DOCX, TXT, MD, HTML-Dateien
- **FastAPI-Interface**: RESTful API mit automatischer Dokumentation
- **MCP-Server**: Integration als Model Context Protocol Server

## 🏗️ Architektur

```
Benutzeranfrage
    ↓
[ Input Guardrail ] ─── Prüfung auf Angemessenheit, Relevanz, Sicherheit
    ↓ (ZUGELASSEN)
[ COLPALI Retrieval ] ── Visuelle Dokumentensuche
    ↓
[ LLM Generation ] ────── Antwortgenerierung basierend auf Quellen
    ↓
[ Output Guardrail ] ─── Faktentreue, Vollständigkeit, Sicherheitsprüfung
    ↓ (GÜLTIG)
Antwort an Nutzer
```

## 🚀 Installation

### Voraussetzungen

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) Paketmanager
- [Ollama](https://ollama.com/) für lokale LLM-Inferenz
- GPU mit CUDA-Unterstützung (empfohlen)

### 1. Repository klonen

```bash
git clone <repository-url>
cd GuardRAG
```

### 2. Virtuelle Umgebung erstellen

```bash
# Mit uv (empfohlen)
uv venv
.venv\Scripts\activate  # Windows

# Dependencies installieren
uv pip install -e .
```

### 3. Ollama-Modelle installieren

```bash
# Basis-LLM für Guardrails und Generation
ollama pull qwen2.5:latest

# Alternative Modelle (optional)
ollama pull granite3.3:8b
ollama pull qwen2.5-coder:latest
ollama pull llava:latest
```

### 4. Umgebungsvariablen konfigurieren

```bash
# .env Datei erstellen
copy .env.example .env

# .env bearbeiten und Konfiguration anpassen
```

### 5. Server starten

```bash
python main.py
```

Die API ist verfügbar unter: `http://localhost:8000`
Dokumentation: `http://localhost:8000/docs`

## 📖 API-Nutzung

### Dokument hochladen

```bash
curl -X POST "http://localhost:8000/upload-document" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "success": true,
  "message": "Document processed successfully",
  "file_id": "1641234567_document.pdf",
  "pages_processed": 15
}
```

### RAG-Abfrage stellen

```bash
curl -X POST "http://localhost:8000/rag-query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Welche Methodik wurde in der Studie verwendet?",
    "max_results": 5
  }'
```

**Response:**
```json
{
  "answer": "Basierend auf den bereitgestellten Dokumenten verwendete die Studie eine quantitative Methodik mit...",
  "confidence": 0.85,
  "processing_time": 2.34,
  "sources": [
    {
      "page_number": 3,
      "score": 0.92,
      "explanation": "Seite 3 enthält übereinstimmende Begriffe: methodik, studie",
      "text_preview": "Die Methodik dieser Studie basiert auf..."
    }
  ],
  "warnings": [],
  "guardrail_checks": {
    "input_validation": {
      "result": "accepted",
      "reason": "Wissenschaftliche Begriffe gefunden: methodik, studie",
      "confidence": 0.8
    },
    "output_validation": {
      "result": "approved", 
      "reason": "Alle Validierungen bestanden",
      "confidence": 0.85
    }
  }
}
```

### System-Status abrufen

```bash
curl -X GET "http://localhost:8000/system-status"
```

## 🛡️ Guardrails

### Input Guardrails

**Zweck**: Filterung problematischer oder irrelevanter Anfragen

**Prüfungen**:
- Grundlegende Validierung (Länge, Format)
- Keyword-Filterung (verbotene Begriffe)
- LLM-basierte Bewertung (Angemessenheit, Relevanz, Sicherheit)

**Beispiele abgelehnter Anfragen**:
- Hassrede oder diskriminierende Inhalte
- Anweisungen für illegale Aktivitäten
- Themenirrelevante Fragen

### Output Guardrails

**Zweck**: Qualitätssicherung und Sicherheitsprüfung der Antworten

**Prüfungen**:
- Faktentreue (Abgleich mit Quellen)
- Vollständigkeit (angemessene Beantwortung)
- Toxizitätserkennung
- Halluzinationsdetektion

## 🔧 Konfiguration

### Umgebungsvariablen

| Variable | Beschreibung | Default |
|----------|--------------|---------|
| `LLM_ENDPOINT` | Ollama API Endpoint | `http://localhost:11434/v1` |
| `LLM_MODEL` | Modell für Generation | `qwen2.5:latest` |
| `COLPALI_MODEL` | COLPALI Modell | `vidore/colqwen2.5-v0.2` |
| `ENABLE_INPUT_GUARDRAILS` | Input-Validierung aktivieren | `true` |
| `ENABLE_OUTPUT_GUARDRAILS` | Output-Validierung aktivieren | `true` |
| `TOXICITY_THRESHOLD` | Schwellwert für Toxizität | `0.3` |
| `CONFIDENCE_THRESHOLD` | Mindestvertrauen | `0.7` |

### Verfügbare Ollama-Modelle

- `granite3.2-vision:2b` - Kompaktes Vision-Modell
- `granite-code:8b` - Code-spezialisiert  
- `qwen2.5-coder:latest` - Coding und Analyse
- `qwen3:latest` - Allzweck-Modell
- `granite3.3:8b` - Ausgewogenes Modell
- `llava:latest` - Vision-Language-Modell
- `qwen2.5:latest` - Empfohlenes Standardmodell
- `google/gemma3:latest` - Google Gemma
- `bge-m3:latest` - Embedding-Modell

## 🔍 COLPALI-Integration

GuardRAG nutzt COLPALI (Collaborative Learning for Vision-Language Models) für effizienten visuellen Dokumentenabruf:

### Funktionsweise

1. **PDF-Verarbeitung**: Konvertierung zu hochauflösenden Bildern (200 DPI)
2. **Embedding-Generierung**: COLPALI erstellt Multi-Vector-Embeddings
3. **Visuelle Suche**: Similaritätsberechnung zwischen Query und Dokumentenseiten
4. **Ranking**: Relevanz-Score-basierte Sortierung

### Vorteile

- **Layout-Bewusstsein**: Erkennt Tabellen, Diagramme, Strukturen
- **OCR-frei**: Keine fehleranfällige Texterkennung nötig
- **Multimodal**: Text und visuelle Elemente gemeinsam
- **Effizienz**: Schnelle Suche in großen Dokumentensammlungen

## 🧪 Testing

```bash
# Tests ausführen
uv run pytest

# Mit Coverage
uv run pytest --cov=src

# Spezifische Tests
uv run pytest tests/test_guardrails.py
```

## 📊 Monitoring

### Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

### Logs

Logs werden in strukturiertem Format ausgegeben:
```
2025-01-10 15:30:45 - guardrag.main - INFO - GuardRAG application started successfully
2025-01-10 15:31:02 - guardrag.rag_agent - INFO - Processing query: "Welche Methodik wurde verwendet?"
2025-01-10 15:31:03 - guardrag.input_guardrail - INFO - Input validation passed
2025-01-10 15:31:05 - guardrag.colpali - INFO - Retrieved 5 relevant pages
2025-01-10 15:31:07 - guardrag.output_guardrail - INFO - Output validation approved
```

## 🛠️ Entwicklung

### Projektstruktur

```
GuardRAG/
├── src/                          # Kernkomponenten
│   ├── colpali_integration.py    # COLPALI-Wrapper
│   ├── input_guardrail.py        # Eingabevalidierung
│   ├── output_guardrail.py       # Ausgabevalidierung
│   └── rag_agent.py              # Haupt-RAG-Agent
├── mcp_services/                 # Bestehende MCP-Services
├── main.py                       # FastAPI-Anwendung
├── pyproject.toml               # Projektkonfiguration
├── .env.example                 # Umgebungsvorlage
└── uploads/                     # Upload-Verzeichnis
```

### Code-Stil

```bash
# Formatierung
uv run black src/
uv run isort src/

# Linting
uv run flake8 src/
uv run mypy src/
```

## 🚨 Troubleshooting

### Häufige Probleme

1. **CUDA Out of Memory**
   ```bash
   # Kleineres COLPALI-Modell verwenden
   export COLPALI_MODEL=vidore/colSmol-256M
   ```

2. **Ollama Connection Error**
   ```bash
   # Ollama-Service prüfen
   ollama list
   ollama serve
   ```

3. **PDF Conversion Fehler**
   ```bash
   # LibreOffice installieren (für Office-Dateien)
   # WeasyPrint installieren (für HTML)
   uv pip install weasyprint
   ```

### Debug-Modus

```bash
# Verbose Logging aktivieren
export UVICORN_LOG_LEVEL=debug
python main.py
```

## 🤝 Beitragen

1. Fork des Repositories
2. Feature Branch erstellen (`git checkout -b feature/neue-funktion`)
3. Änderungen committen (`git commit -am 'Neue Funktion hinzufügen'`)
4. Branch pushen (`git push origin feature/neue-funktion`)
5. Pull Request erstellen

## 📄 Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe [LICENSE](LICENSE) für Details.

## 🙏 Danksagungen

- [COLPALI](https://github.com/illuin-tech/colpali) für das exzellente Vision-Language-Retrieval-Framework
- [PydanticAI](https://ai.pydantic.dev/) für die strukturierte LLM-Integration
- [FastAPI](https://fastapi.tiangolo.com/) für das moderne Web-Framework
- [Ollama](https://ollama.com/) für lokale LLM-Inferenz

## 📞 Support

Bei Fragen oder Problemen:
- Issues im Repository erstellen
- Dokumentation unter `/docs` konsultieren
- Logs für Fehlerdiagnose nutzen