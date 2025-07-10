# GuardRAG Setup & Usage Guide

## 🚀 Quick Start

### 1. Abhängigkeiten installieren

```powershell
# Mit uv (empfohlen)
uv sync

# Oder mit pip
.venv\Scripts\activate
pip install -e .
```

### 2. Umgebung konfigurieren

Kopiere `.env.example` zu `.env` und passe die Werte an:

```bash
# LLM Configuration (Ollama)
LLM_ENDPOINT=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:latest

# COLPALI Configuration
COLPALI_MODEL=vidore/colqwen2.5-v0.2

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 3. Services starten

#### Option A: Docker Compose (empfohlen)

```powershell
# Alle Services starten (Ollama + Qdrant + GuardRAG)
docker-compose up -d

# Ollama Modelle pullen
docker exec guardrag-ollama-1 ollama pull qwen2.5:latest
docker exec guardrag-ollama-1 ollama pull granite3.2-vision:2b
```

#### Option B: Lokal

```powershell
# 1. Qdrant starten
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage:z qdrant/qdrant:latest

# 2. Ollama starten (falls nicht installiert)
# Download von https://ollama.ai und installieren
ollama serve
ollama pull qwen2.5:latest

# 3. GuardRAG starten
.venv\Scripts\activate
python main.py
```

### 4. Verfügbare LLM-Modelle

Das System unterstützt folgende Ollama-Modelle:

- `granite3.2-vision:2b` - Kompaktes Vision-Language-Modell
- `granite-code:8b` - Code-spezialisiertes Modell
- `qwen2.5-coder:latest` - Erweiterte Coding-Capabilities
- `qwen3:latest` - Neueste Qwen-Version
- `granite3.3:8b` - Allzweck-Modell
- `llava:latest` - Vision-Language-Modell
- `qwen2.5:latest` - Standardmodell (empfohlen)
- `google/gemma3:latest` - Google's Gemma-Modell
- `bge-m3:latest` - Embedding-Modell

### 5. API-Endpunkte testen

```powershell
# Health Check
curl http://localhost:8000/health

# Dokument hochladen
curl -X POST "http://localhost:8000/upload-document" -F "file=@example.pdf"

# RAG Query
curl -X POST "http://localhost:8000/rag-query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Was ist der Hauptinhalt des Dokuments?", "max_results": 5}'

# System Status
curl http://localhost:8000/system-status
```

## 🛠️ Entwicklung

### Test-Suite

Das GuardRAG-System verfügt über eine umfangreiche Test-Suite mit verschiedenen Test-Kategorien:

#### Verfügbare Test-Dateien

- **`tests/test_comprehensive.py`** - Umfassende Unit- und Integrationstests
  - Qdrant-Integration (Vektorspeicher)
  - COLPALI-Integration (Dokumentenverarbeitung)
  - Input/Output-Guardrails
  - RAG-Agent-Funktionalität
  - Performance-Tests

- **`tests/test_api.py`** - FastAPI-Endpunkt-Tests
  - Alle REST-API-Endpunkte
  - Fehlerbehandlung
  - Validierung
  - File-Upload-Tests

- **`tests/test_guardrails.py`** - Legacy-Guardrail-Tests
  - Basis-Validierung
  - Keyword-Filterung
  - Async-Validierung

- **`test_setup.py`** - Basis-Setup-Verifizierung

#### Test-Suite ausführen

```powershell
# Vollständige Test-Suite ausführen
python run_tests.py

# Einzelne Test-Kategorien
python tasks.py test-unit          # Unit Tests
python tasks.py test-api           # API Tests  
python tasks.py test-coverage      # Tests mit Coverage

# Mit pytest direkt
.venv\Scripts\activate
python -m pytest tests/ -v                    # Alle Tests
python -m pytest tests/test_comprehensive.py -v  # Nur comprehensive Tests
python -m pytest tests/test_api.py -v         # Nur API Tests

# Coverage Report generieren
python -m pytest --cov=src --cov-report=html --cov-report=term
```

#### Test-Utilities

- **`run_tests.py`** - Haupttest-Runner mit Setup-Verifizierung
- **`tasks.py`** - Entwicklungsaufgaben-Manager
- **`pytest.ini`** - Pytest-Konfiguration

#### Coverage-Reports

Nach dem Ausführen der Tests mit Coverage:
- **HTML-Report**: `htmlcov/index.html`
- **Terminal-Report**: Direkte Ausgabe
- **XML-Report**: `coverage.xml` (für CI/CD)

### Development Tasks

```powershell
# Entwicklungsumgebung einrichten
python tasks.py dev-setup

# Code formatieren
python tasks.py format

# Code prüfen  
python tasks.py lint

# Dependencies installieren
python tasks.py install-dev

# Server starten
python tasks.py start

# Alle verfügbaren Tasks anzeigen
python tasks.py help
```

### Code-Qualität

```powershell
# Formatting
.venv\Scripts\python.exe -m black src/ tests/
.venv\Scripts\python.exe -m isort src/ tests/

# Linting
.venv\Scripts\python.exe -m flake8 src/
.venv\Scripts\python.exe -m mypy src/

# Tests mit Coverage
.venv\Scripts\python.exe -m pytest --cov=src --cov-report=term-missing --cov-report=html
```

## 📁 Projektstruktur

```
GuardRAG/
├── src/                          # Kernkomponenten
│   ├── colpali_integration.py    # COLPALI + Qdrant Integration
│   ├── qdrant_integration.py     # Vektor-Datenbank
│   ├── input_guardrail.py        # Eingabevalidierung
│   ├── output_guardrail.py       # Ausgabevalidierung
│   └── rag_agent.py              # Haupt-RAG-Agent
├── mcp_fileconverter/            # PDF-Konvertierung
├── main.py                       # FastAPI-Anwendung
├── docker-compose.yml           # Multi-Service Setup
├── pyproject.toml               # Projekt-Konfiguration
└── .env.example                 # Umgebungsvorlage
```

## 🔧 Konfiguration

### Qdrant

```python
# Lokale Qdrant-Instanz
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Qdrant Cloud
QDRANT_URL=https://your-cluster.qdrant.tech
QDRANT_API_KEY=your-api-key
```

### COLPALI

Das System nutzt standardmäßig das `vidore/colqwen2.5-v0.2` Modell für visuelle Dokumentenanalyse.

### Guardrails

```python
# Input Guardrails aktivieren/deaktivieren
ENABLE_INPUT_GUARDRAILS=true

# Output Guardrails aktivieren/deaktivieren
ENABLE_OUTPUT_GUARDRAILS=true
```

## 🐛 Troubleshooting

### Häufige Probleme

1. **Qdrant Connection Failed**
   ```bash
   # Prüfe ob Qdrant läuft
   curl http://localhost:6333/collections
   ```

2. **COLPALI Model Download**
   ```bash
   # Manuell Model herunterladen
   python -c "from colpali_engine.models import ColQwen2; ColQwen2.from_pretrained('vidore/colqwen2.5-v0.2')"
   ```

3. **Memory Issues**
   ```bash
   # Docker Memory Limits erhöhen
   docker-compose up --scale guardrag=1 --memory=8g
   ```

## 📚 API-Dokumentation

Nach dem Start ist die interaktive API-Dokumentation verfügbar unter:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🤝 Beiträge

1. Fork das Repository
2. Erstelle einen Feature-Branch
3. Committe deine Änderungen
4. Erstelle einen Pull Request

## 📄 Lizenz

Siehe `LICENSE.md` für Details.
