# 🛡️ GuardRAG - Secure Document RAG System

Ein fortschrittliches RAG-System (Retrieval-Augmented Generation) auf Basis von COLPALI mit integrierten Guardrails für sichere und vertrauenswürdige Dokumentenverarbeitung.

## 🎯 Überblick

GuardRAG kombiniert modernste Vision-Language-Modelle (COLPALI) mit umfassenden Sicherheitsmechanismen (Guardrails), um eine robuste und vertrauenswürdige Lösung für die Analyse wissenschaftlicher Dokumente zu bieten.

### ✨ Kernfunktionen

- **🔍 COLPALI-Integration**: Effiziente visuelle Dokumentenretrieval mit dem `vidore/colqwen2.5-v0.2` Modell
- **🛡️ Input Guardrails**: Validierung und Filterung eingehender Anfragen
- **✅ Output Guardrails**: Überprüfung generierter Antworten auf Faktentreue und Sicherheit
- **📁 File Upload**: Unterstützung für PDF, DOCX, TXT, MD, HTML-Dateien
- **🚀 FastAPI-Interface**: RESTful API mit automatischer Dokumentation
- **🧠 Hybrid LLM+Regex**: Intelligente Content-Validierung mit automatischem Lernen
- **🔒 PII-Schutz**: Erkennung und Sanitisierung persönlicher Daten

## 🏗️ Systemarchitektur

```
Benutzeranfrage
    ↓
[ Input Guardrail ] ─── 🔍 Hybrid LLM+Regex Prüfung:
    │                   • Profanität & Toxizität
    │                   • PII-Erkennung (E-Mail, Tel, IBAN)
    │                   • Gefährliche Inhalte (Drogen, Waffen, Sprengstoff)
    │                   • Konkurrenten-Erwähnungen
    │                   • Automatisches Lernen neuer Bedrohungen
    ↓ (ZUGELASSEN)
[ COLPALI Retrieval ] ── 📊 Visuelle Dokumentensuche:
    │                   • Layout-bewusste Analyse
    │                   • OCR-freie Texterkennung
    │                   • Multimodale Embeddings
    ↓
[ LLM Generation ] ────── 🤖 Antwortgenerierung basierend auf Quellen
    ↓
[ Output Guardrail ] ─── ✅ Qualitätsprüfung:
    │                   • Faktentreue-Abgleich
    │                   • Halluzinations-Detektion
    │                   • Sicherheitsprüfung
    ↓ (GÜLTIG)
Sichere Antwort an Nutzer
```

## 🚀 Installation & Quick Start

### Voraussetzungen

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) Paketmanager (empfohlen)
- [Ollama](https://ollama.com/) für lokale LLM-Inferenz
- GPU mit CUDA-Unterstützung (empfohlen)
- Docker (optional, für Services)

### 1. Repository klonen

```bash
git clone <repository-url>
cd GuardRAG
```

### 2. Abhängigkeiten installieren

```powershell
# Mit uv (empfohlen)
uv sync

# Oder mit pip
.venv\Scripts\activate
pip install -e .
```

### 3. Umgebung konfigurieren

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

# Guardrails Configuration
ENABLE_INPUT_GUARDRAILS=true
ENABLE_OUTPUT_GUARDRAILS=true
TOXICITY_THRESHOLD=0.3
CONFIDENCE_THRESHOLD=0.7
```

### 4. Services starten

#### Option A: Docker Compose (empfohlen)

```powershell
# Alle Services starten (Ollama + Qdrant + GuardRAG)
docker-compose up -d

# Ollama Modelle pullen
docker exec guardrag-ollama-1 ollama pull qwen2.5:latest
docker exec guardrag-ollama-1 ollama pull granite3.2-vision:2b
```

#### Option B: Lokale Services

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

### 5. Verfügbarkeitsprüfung

```bash
# Health Check
curl http://localhost:8000/health

# API-Dokumentation
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

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

## 🛡️ Guardrails - Intelligente Sicherheitssysteme

### 🔍 Input Guardrails

**Zweck**: Hybrid LLM+Regex Filterung problematischer oder irrelevanter Anfragen

**Prüfkategorien**:

#### 1. 💬 Profanität & Toxizität
- **Deutsche & englische Schimpfwörter**: fuck, scheiße, arschloch, etc.
- **Diskriminierende Begriffe**: rassistische, sexistische, homophobe Sprache
- **Hasssprache**: Nazi-Begriffe, Fremdenfeindlichkeit
- **Toxische Kommunikationsmuster**: Bedrohungen, Beleidigungen

#### 2. 🔒 PII-Erkennung (Persönliche Daten)
- **E-Mail-Adressen**: max.mustermann@beispiel.de
- **Telefonnummern**: +49 123 456789, deutsche und internationale Formate
- **Deutsche Adressen**: Musterstraße 123, 12345 Berlin
- **Bankdaten**: IBAN, Kreditkartennummern
- **Personennamen**: spaCy-basierte Named Entity Recognition
- **Ausweisnummern**: Deutsche Steuer-ID, Personalausweis

#### 3. ⚠️ Gefährliche Inhalte (mit automatischem Lernen)
- **Sprengstoff-Chemikalien**: Toluol, Schwefel, Salpetersäure, TNT, C4
- **Drogen**: CBD, THC, Cannabis, Methamphetamine, Kokain, LSD
- **Waffen**: Pistole, Gewehr, AK47, Munition
- **Illegale Aktivitäten**: Hacking, Phishing, Geldwäsche

**🧠 Hybrid-Ansatz**:
1. **Regex-Trigger** erkennen verdächtige Begriffe (schnell)
2. **LLM analysiert Kontext** (intelligent): 
   - "Puderzucker für Guggelhupf" → ✅ harmlos (Backen)
   - "Puderzucker + Salz explodieren" → ❌ gefährlich (Sprengstoff)
3. **Automatisches Lernen**: LLM-erkannte Gefahren ohne Regex → Lern-Log

#### 4. 🏢 Konkurrenten-Erkennung
- **Blockierte Anbieter**: OpenAI, ChatGPT, Claude, Gemini, Anthropic
- **Benutzerdefinierte Listen**: Erweiterbar
- **Automatische Blockierung**: Bei Erwähnung von Konkurrenzprodukten

#### 5. 🧹 Text-Sanitisierung
- **PII-Ersetzung**: E-Mails → [EMAIL], Telefon → [PHONE]
- **Overlap-Resolution**: Überlappende Erkennungen bereinigen
- **Struktur-Erhaltung**: Textformat bleibt erhalten

### ✅ Output Guardrails

**Zweck**: Qualitätssicherung und Sicherheitsprüfung der generierten Antworten

**Prüfungen**:
- **Faktentreue**: Abgleich mit Quellendokumenten
- **Vollständigkeit**: Angemessene Beantwortung der Frage
- **Halluzinations-Detektion**: Erkennung erfundener Informationen
- **Toxizitätsprüfung**: Sicherstellung sauberer Ausgaben
- **Relevanz-Check**: Passende Antwort zur Frage

### 📊 Beispiele

#### ✅ Zugelassene Anfragen
```
"Wie funktioniert maschinelles Lernen?"
"CBD Kekse backen - welche Dosierung?"
"Welche Methodik wurde in der Studie verwendet?"
```

#### ❌ Blockierte Anfragen
```
"Du bist so dumm!" (Toxizität)
"Wie stelle ich Sprengstoff her?" (Gefährlicher Inhalt)
"Meine E-Mail ist max@test.de" (PII)
"ChatGPT ist besser" (Konkurrent)
```

#### 🧹 Sanitisierte Ausgaben
```
Input:  "Kontaktiere mich unter max@test.de"
Output: "Kontaktiere mich unter [EMAIL]"
```

## 🔧 Konfiguration & Verfügbare Modelle

### Umgebungsvariablen

| Variable | Beschreibung | Default |
|----------|--------------|---------|
| `LLM_ENDPOINT` | Ollama API Endpoint | `http://localhost:11434/v1` |
| `LLM_API_KEY` | API Key für LLM | `ollama` |
| `LLM_MODEL` | Modell für Generation | `qwen2.5:latest` |
| `COLPALI_MODEL` | COLPALI Modell | `vidore/colqwen2.5-v0.2` |
| `QDRANT_HOST` | Qdrant Server Host | `localhost` |
| `QDRANT_PORT` | Qdrant Server Port | `6333` |
| `ENABLE_INPUT_GUARDRAILS` | Input-Validierung aktivieren | `true` |
| `ENABLE_OUTPUT_GUARDRAILS` | Output-Validierung aktivieren | `true` |
| `TOXICITY_THRESHOLD` | Schwellwert für Toxizität | `0.3` |
| `CONFIDENCE_THRESHOLD` | Mindestvertrauen | `0.7` |

### 🤖 Unterstützte LLM-Modelle

Das System unterstützt **alle OpenAI-kompatiblen API-Endpunkte**:

#### 🏠 Lokale Modelle (Ollama) z. B.
- **`qwen2.5:latest`** ⭐ - Empfohlenes Standardmodell (Allzweck)
- **`qwen3:latest`** - Neueste Qwen-Version
- **`granite3.3:8b`** - Ausgewogenes Allzweck-Modell
- **`google/gemma3:latest`** - Google's Gemma-Modell
- **Alle weiteren Ollama-Modelle** - Vollständige Kompatibilität

#### ☁️ Cloud-APIs (OpenAI-kompatibel)
- **OpenAI**: GPT-4, GPT-3.5-turbo, GPT-4-turbo
- **Anthropic**: Claude-3.5-Sonnet, Claude-3-Haiku (via kompatible APIs)
- **Groq**: Llama-3, Mixtral, Gemma (schnelle Inferenz)
- **Together AI**: Llama-2/3, Code Llama, Mistral
- **Perplexity**: Llama-3, Mixtral-8x7B
- **OpenRouter**: Zugang zu 100+ Modellen
- **Azure OpenAI**: Enterprise-GPT-Modelle
- **AWS Bedrock**: Claude, Llama über OpenAI-Proxy

#### 🔧 Konfiguration für verschiedene Anbieter

```bash
# Lokale Ollama-Installation
LLM_ENDPOINT=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:latest

# OpenAI
LLM_ENDPOINT=https://api.openai.com/v1
LLM_API_KEY=sk-your-openai-key
LLM_MODEL=gpt-4

# Groq (schnelle Inferenz)
LLM_ENDPOINT=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_your-groq-key
LLM_MODEL=llama-3.1-70b-versatile

# Together AI
LLM_ENDPOINT=https://api.together.xyz/v1
LLM_API_KEY=your-together-key
LLM_MODEL=meta-llama/Llama-3-70b-chat-hf

# OpenRouter (Multi-Provider)
LLM_ENDPOINT=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-your-openrouter-key
LLM_MODEL=anthropic/claude-3.5-sonnet

# Lokale vLLM-Installation
LLM_ENDPOINT=http://localhost:8000/v1
LLM_API_KEY=token-abc123
LLM_MODEL=microsoft/DialoGPT-medium
```

Das System funktioniert mit **jedem OpenAI-kompatiblen Endpunkt** - einfach Endpoint, API-Key und Modellname konfigurieren!

### 📂 Projektstruktur

```
GuardRAG/
├── src/                           # 🎯 Kernkomponenten
│   ├── modern_guardrails.py       # 🛡️ Hybrid LLM+Regex Guardrails
│   ├── colpali_integration.py     # 🔍 COLPALI + Qdrant Integration
│   ├── qdrant_integration.py      # 📊 Vektor-Datenbank
│   ├── input_guardrail.py         # 🚫 Eingabevalidierung
│   ├── output_guardrail.py        # ✅ Ausgabevalidierung
│   └── rag_agent.py               # 🤖 Haupt-RAG-Agent
├── mcp_fileconverter/             # 📁 PDF-Konvertierung
│   └── file2pdf.py               # 🔄 Datei-zu-PDF-Konverter
├── tests/                         # 🧪 Test-Suite (optional)
│   └── (weitere Test-Dateien)     # � Ergänzende Tests
├── main.py                        # 🚀 FastAPI-Anwendung
├── test_hybrid_guardrails.py     # �️ Guardrails-Tests
├── test_complete_system.py       # 🧪 System-Tests
├── tasks.py                      # ⚙️ Development Tasks
├── docker-compose.yml            # 🐳 Multi-Service Setup
├── pyproject.toml                # ⚙️ Projekt-Konfiguration
├── .env.example                  # 📝 Umgebungsvorlage
└── uploads/                      # 📁 Upload-Verzeichnis
```

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

## 🧪 Testing & CLI-Tester

### 🖥️ Test-Skripte

GuardRAG enthält verschiedene Test-Skripte für die Funktionsvalidierung:

```bash
# Hybrid Guardrails System testen
python test_hybrid_guardrails.py

# Komplettes System testen
python test_complete_system.py

# Development Tasks ausführen
python tasks.py help
```

**🎯 Test-Funktionen**:
- **�️ Guardrails-Tests**: Hybrid LLM+Regex Validation
- **🔍 System-Tests**: Vollständige Integration
- **🧪 Unit-Tests**: Einzelkomponenten-Validierung
- **📊 Development Tasks**: Code-Qualität und Automatisierung

**🎛️ Available Test Commands**:
```bash
# Hybrid Guardrails testen
python test_hybrid_guardrails.py

# Komplettes System validieren  
python test_complete_system.py

# Development Tasks
python tasks.py dev-setup      # Entwicklungsumgebung einrichten
python tasks.py test-unit      # Unit Tests ausführen
python tasks.py format         # Code formatieren
python tasks.py lint          # Code-Qualität prüfen
python tasks.py help          # Alle verfügbaren Tasks
```

**📋 Test-Ausgabe-Format**:
```
�️ GUARDRAILS TEST RESULTS
==================================================
✅ Input Validation: PASSED/FAILED
🧹 Sanitization: '[EMAIL] entfernt' 
� Detection Categories:
   • Profanität: 0 found
   • PII: 1 detected
   • Dangerous Content: 0 found
   • Toxicity: 0 found
🎯 Confidence: 0.95
⏱️ Processing Time: 0.12s
```

### 🧪 Automated Test Suite

```powershell
# Test-Skripte ausführen
python test_hybrid_guardrails.py    # Guardrails-System testen
python test_complete_system.py      # Komplette Integration

# Development Tasks
python tasks.py test-unit            # Unit Tests
python tasks.py dev-setup           # Entwicklungsumgebung

# Direkte Test-Skripte
python test_hybrid_guardrails.py    # Guardrails-System testen
python test_complete_system.py      # Komplette Integration
```

**📁 Verfügbare Test-Dateien**:
- **`test_hybrid_guardrails.py`** - Hybrid LLM+Regex Guardrails-Tests
- **`test_complete_system.py`** - Komplette System-Integration-Tests
- **`tasks.py`** - Entwicklungsaufgaben-Manager und Test-Runner

### 📊 Test-Output

Die Test-Skripte liefern strukturierte Ausgaben zur Validierung der Guardrails-Funktionalität.

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

## 🛠️ Entwicklung & Code-Qualität

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

### 🔧 Konfiguration für verschiedene Umgebungen

#### Qdrant-Konfiguration
```python
# Lokale Qdrant-Instanz
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Qdrant Cloud
QDRANT_URL=https://your-cluster.qdrant.tech
QDRANT_API_KEY=your-api-key
```

#### COLPALI-Konfiguration
Das System nutzt standardmäßig das `vidore/colqwen2.5-v0.2` Modell für visuelle Dokumentenanalyse. Bei Memory-Problemen kann ein kleineres Modell verwendet werden:

```bash
# Kleineres COLPALI-Modell bei Memory-Problemen
export COLPALI_MODEL=vidore/colSmol-256M
```

## 🚨 Troubleshooting & FAQ

### Häufige Probleme

#### 1. **CUDA Out of Memory**
```bash
# Kleineres COLPALI-Modell verwenden
export COLPALI_MODEL=vidore/colSmol-256M

# Docker Memory Limits erhöhen
docker-compose up --scale guardrag=1 --memory=8g
```

#### 2. **Ollama Connection Error**
```bash
# Ollama-Service prüfen
ollama list
ollama serve

# API-Verfügbarkeit testen
curl http://localhost:11434/api/tags
```

#### 3. **Qdrant Connection Failed**
```bash
# Prüfe ob Qdrant läuft
curl http://localhost:6333/collections

# Qdrant in Docker starten
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

#### 4. **PDF Conversion Fehler**
```bash
# LibreOffice installieren (für Office-Dateien)
# WeasyPrint installieren (für HTML)
uv pip install weasyprint
```

#### 5. **COLPALI Model Download**
```bash
# Manuell Model herunterladen
python -c "from colpali_engine.models import ColQwen2; ColQwen2.from_pretrained('vidore/colqwen2.5-v0.2')"
```

#### 6. **Test-Skripte starten nicht**
```bash
# Prüfe verfügbare Test-Dateien
python test_hybrid_guardrails.py
python test_complete_system.py

# Prüfe moderne_guardrails Module-Import
python -c "from src.modern_guardrails import create_guardrails_validator"

# spaCy deutsche Modelle installieren
python -m spacy download de_core_news_lg
python -m spacy download de_core_news_sm
```

### Debug-Modus

```bash
# Verbose Logging aktivieren
export UVICORN_LOG_LEVEL=debug
python main.py

# Guardrails Learning-Logs überwachen
tail -f guardrails_learning.jsonl
```

### Performance-Optimierung

```bash
# GPU-Nutzung prüfen
nvidia-smi

# Memory-Usage überwachen
htop

# Qdrant Performance tuning
curl -X PUT "http://localhost:6333/collections/documents/index" \
  -H "Content-Type: application/json" \
  -d '{"field_name": "vector", "field_schema": "Float"}'
```

## 📄 Lizenz

Dieses Projekt steht unter der **GNU Affero General Public License v3.0 (AGPL-3.0)**.

### Was bedeutet das?

- ✅ **Freie Nutzung**: Kostenlose Nutzung für alle Zwecke
- ✅ **Quellcode-Zugang**: Vollständiger Quellcode verfügbar
- ✅ **Modifikationen erlaubt**: Anpassungen und Erweiterungen möglich
- ✅ **Weitergabe erlaubt**: Redistribution unter gleicher Lizenz

### Copyleft-Bedingungen

- 📤 **Quellcode-Pflicht**: Bei Weitergabe muss Quellcode mitgeliefert werden
- 🌐 **Network-Copyleft**: Auch bei Online-Services muss Quellcode verfügbar sein
- � **Gleiche Lizenz**: Abgeleitete Werke müssen unter AGPL-3.0 stehen

Siehe [LICENSE.md](LICENSE.md) für vollständige Details.

## 🙏 Danksagungen

- **[COLPALI](https://github.com/illuin-tech/colpali)** für das exzellente Vision-Language-Retrieval-Framework
- **[PydanticAI](https://ai.pydantic.dev/)** für die strukturierte LLM-Integration
- **[FastAPI](https://fastapi.tiangolo.com/)** für das moderne Web-Framework
- **[Ollama](https://ollama.com/)** für lokale LLM-Inferenz
- **[Qdrant](https://qdrant.tech/)** für die hochperformante Vektordatenbank
- **[spaCy](https://spacy.io/)** für Named Entity Recognition
- **[OpenAI](https://openai.com/)** für die API-Kompatibilität


### 🔍 Weitere Ressourcen

- **API-Dokumentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)
- **Health Check**: http://localhost:8000/health
- **System Status**: http://localhost:8000/system-status

---

🛡️ **GuardRAG** - Sichere, intelligente und lernfähige Dokumentenanalyse mit modernsten AI-Guardrails!