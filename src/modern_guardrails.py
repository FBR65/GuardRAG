#!/usr/bin/env python3
"""
Modern Guardrails System für GuardRAG
Integrierte Toxizitäts-, PII- und Content-Validierung
"""

import logging
import time
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv is optional

# spaCy imports
try:
    import spacy

    SPACY_AVAILABLE = True
    # Versuche das deutsche Modell zu laden
    try:
        nlp = spacy.load("de_core_news_lg")
        logger = logging.getLogger(__name__)
        logger.info("Loaded German spaCy model: de_core_news_lg")
    except OSError:
        try:
            nlp = spacy.load("de_core_news_sm")
            logger = logging.getLogger(__name__)
            logger.info("Loaded German spaCy model: de_core_news_sm")
        except OSError:
            nlp = spacy.load("en_core_web_sm")
            logger = logging.getLogger(__name__)
            logger.warning("German spaCy model not found, using English model")
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None
    logger = logging.getLogger(__name__)
    logger.warning("spaCy not available, using regex-only PII detection")


class ValidationResult(Enum):
    """Validation result types."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    WARNING = "warning"


class GuardrailType(Enum):
    """Types of guardrail validations."""

    PROFANITY = "profanity"
    TOXICITY = "toxicity"
    PII = "pii"
    COMPETITOR = "competitor"
    CONTENT_SAFETY = "content_safety"
    DANGEROUS_CONTENT = "dangerous_content"


@dataclass
class GuardrailResult:
    """Result of guardrail validation."""

    result: ValidationResult
    validator_type: GuardrailType
    reason: str
    confidence: float
    details: Optional[Dict[str, Any]] = None
    sanitized_text: Optional[str] = None
    suggestions: Optional[List[str]] = None


@dataclass
class ValidationDetails:
    """Detailed validation result."""

    passed: bool
    validator_name: str
    input_text: str
    sanitized_text: Optional[str] = None
    reason: Optional[str] = None
    confidence: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class ProfanityValidator:
    """Erweiterte Profanitäts-Validator für deutsche und englische Sprache"""

    def __init__(self):
        self.profanity_words = {
            # Englische Schimpfwörter
            "fuck",
            "fucking",
            "fucked",
            "fucker",
            "shit",
            "shitty",
            "bullshit",
            "damn",
            "damned",
            "goddamn",
            "bitch",
            "bitching",
            "bastard",
            "ass",
            "asshole",
            "arse",
            "hell",
            "crap",
            "piss",
            "cock",
            "dick",
            "penis",
            "pussy",
            "cunt",
            "whore",
            "slut",
            "hooker",
            "prostitute",
            # Deutsche Schimpfwörter (Basis)
            "scheiße",
            "scheiss",
            "kacke",
            "verdammt",
            "verflucht",
            "gottverdammt",
            "arschloch",
            "arsch",
            "hintern",
            "fotze",
            "muschi",
            "möse",
            "hure",
            "nutte",
            "schlampe",
            "ficken",
            "fick",
            "gefickt",
            "wichser",
            "wichsen",
            "onanieren",
            "pimmel",
            "schwanz",
            "glied",
            # Deutsche Beleidigungen
            "hurensohn",
            "mistkerl",
            "drecksack",
            "vollidiot",
            "vollpfosten",
            "trottel",
            "depp",
            "idiot",
            "schwachkopf",
            "spasti",
            "mongo",
            "behinderter",
            # Diskriminierende deutsche Begriffe
            "schwuchtel",
            "tunte",
            "homo",
            "lesbe",
            "kampflesbe",
            "dyke",
            "kanake",
            "türke",
            "ausländer",
            "neger",
            "mohr",
            "schwarzer",
            "schlitzauge",
            "chinese",
            "japs",
            # Vulgäre Ausdrücke
            "porno",
            "geil",
            "pervers",
            "versaut",
            "dreckig",
            "schmutzig",
            "abspritzen",
            "kommen",
            "orgasmus",
            # Hasserfüllte Begriffe
            "nazi",
            "faschistisch",
            "rassistisch",
            "antisemitisch",
            "judenhass",
            "fremdenhass",
            "islamophob",
            "homophob",
            "sexistisch",
            # Weitere vulgäre deutsche Begriffe
            "pisser",
            "kacken",
            "scheißen",
            "furzen",
            "pupsen",
            "rotzen",
            "kotzen",
            "spucken",
            "sabbern",
        }

    def validate(self, text: str) -> ValidationDetails:
        """Validiert Text auf Profanität"""
        text_lower = text.lower()
        found_words = []

        for word in self.profanity_words:
            if word in text_lower:
                found_words.append(word)

        passed = len(found_words) == 0
        reason = f"Profanity detected: {found_words}" if not passed else None

        return ValidationDetails(
            passed=passed,
            validator_name="ProfanityValidator",
            input_text=text,
            reason=reason,
            confidence=1.0 if found_words else 0.9,
            details={"found_words": found_words},
        )


class ToxicLanguageValidator:
    """Erweiterte Toxizitäts-Validator mit deutschen und englischen Mustern"""

    def __init__(self):
        self.toxic_patterns = [
            # Englische toxische Muster (ohne "die" wegen false positives)
            r"\b(kill|death|murder|violence)\b",
            r"\b(hate|stupid|idiot|moron)\b",
            r"\b(shut up|fuck off)\b",
            r"\b(bomb|explosive|terrorism)\b",
            # Spezifische englische "die" Kontexte (nur toxische)
            r"\b(die\s+now|go\s+die|should\s+die|gonna\s+die|must\s+die|want\s+to\s+die|wanna\s+die)\b",
            # Deutsche toxische Grundmuster
            r"\b(töten|sterben|hass|dumm|idiot|umbringen|erschießen|vernichten)\b",
            r"\b(halt die fresse|verpiss dich|fick dich|halt's maul)\b",
            # Deutsche Beleidigungen und Hassrede
            r"\b(nazi|arschloch|hurensohn|wichser|vollidiot|schwuchtel|tunte|transe)\b",
            r"\b(kanake|schlitzauge|neger|judensau|ziegenficker|terrorist)\b",
            r"\b(nutte|schlampe|hure|fotze|schwanz|pimmel)\b",
            # Rassistische/diskriminierende deutsche Begriffe
            r"\b(ausländer\s+raus|deutschland\s+den\s+deutschen|sieg\s+heil)\b",
            r"\b(schwarzfahren|anschwärzen|schwarzes?\s+schaf|schwarzer?\s+tag)\b",
            r"\b(zigeunerschnitzel|mohren\w*|negerkuss)\b",
            # Sexistische/homophobe Begriffe
            r"\b(lesbe|schwuchteln?|homo\w*|faggot|dyke)\b",
            r"\b(weibsvolk|tussi|zicke|emanze|kampflesbe)\b",
            # Antisemitische Begriffe
            r"\b(jude\w*\s+(?:raus|weg|tot)|holocaustleugnung|auschwitzlüge)\b",
            r"\b(rothschild\w*|illuminati|weltjudentum|zionistisch\w*)\b",
            # Negative Kommunikationsmuster (kontextabhängig)
            r"\b(das\s+geht\s+nicht|unmöglich|niemals|auf\s+keinen\s+fall)\b",
            r"\b(du\s+musst|ihr\s+sollt|das\s+ist\s+falsch|völlig\s+daneben|völlig\s+falsch)\b",
            r"\b(schwachsinn|blödsinn|quatsch|unsinn|totaler\s+mist)\b",
            # Gewaltandrohungen
            r"\b(bring\w*\s+dich\s+um|schlag\w*\s+tot|mach\w*\s+platt)\b",
            r"\b(abstechen|abknallen|fertig\s+machen|kaputt\s+machen)\b",
            # Extremistische Begriffe
            r"\b(reichsbürger|nwo|chemtrails|deep\s+state|qanon)\b",
            r"\b(lügenpresse|system\w*|volksverräter|gutmensch\w*)\b",
        ]

    def validate(self, text: str) -> ValidationDetails:
        """Validiert Text auf toxische Sprache"""
        text_lower = text.lower()
        found_patterns = []

        for pattern in self.toxic_patterns:
            if re.search(pattern, text_lower):
                found_patterns.append(pattern)

        passed = len(found_patterns) == 0
        reason = (
            f"Toxic language patterns detected: {len(found_patterns)}"
            if not passed
            else None
        )

        return ValidationDetails(
            passed=passed,
            validator_name="ToxicLanguageValidator",
            input_text=text,
            reason=reason,
            confidence=0.8,
            details={"found_patterns": found_patterns},
        )


class AdvancedPIIValidator:
    """Erweiterte PII Validator mit spaCy Integration und Text-Sanitisierung"""

    def __init__(self, replace_pii: bool = True):
        self.replace_pii = replace_pii

        # Erweiterte Regex-Patterns für deutsche und internationale PII
        self.pii_patterns = {
            # Email-Adressen
            "email": {
                "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "replacement": "[EMAIL]",
            },
            # Deutsche Telefonnummern (erweitert)
            "phone_de": {
                "pattern": r"\b(?:\+49\s?|0)(?:\(\d+\)\s?|\d+[\s-]?)[\d\s-]{6,}\b",
                "replacement": "[PHONE]",
            },
            # Internationale Telefonnummern
            "phone_intl": {
                "pattern": r"\+\d{1,4}[\s-]?\d{1,4}[\s-]?\d{4,}",
                "replacement": "[PHONE]",
            },
            # Deutsche IBAN
            "iban_de": {
                "pattern": r"\bDE\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}\b",
                "replacement": "[IBAN]",
            },
            # Internationale IBAN (vereinfacht)
            "iban_intl": {
                "pattern": r"\b[A-Z]{2}\d{2}[\s]?[\d\s]{15,34}\b",
                "replacement": "[IBAN]",
            },
            # US Social Security Numbers
            "ssn": {"pattern": r"\b\d{3}-\d{2}-\d{4}\b", "replacement": "[SSN]"},
            # Kreditkartennummern
            "credit_card": {
                "pattern": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                "replacement": "[CREDIT_CARD]",
            },
            # Deutsche Postleitzahlen + Ort
            "postal_code_de": {
                "pattern": r"\b\d{5}\s+[A-ZÄÖÜ][a-zäöüß-]+(?:\s+[A-ZÄÖÜ][a-zäöüß-]*)*\b",
                "replacement": "[PLZ_ORT]",
            },
            # Straße + Hausnummer (deutsch)
            "street_address_de": {
                "pattern": r"\b[A-ZÄÖÜ][a-zäöüß-]+(?:straße|str\.|gasse|weg|platz|allee)\s+\d+[a-z]?\b",
                "replacement": "[ADRESSE]",
            },
            # Geburtsdaten (verschiedene Formate)
            "birthdate": {
                "pattern": r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})\b",
                "replacement": "[GEBURTSDATUM]",
            },
            # Deutsche Steuer-ID (11 Ziffern)
            "tax_id_de": {"pattern": r"\b\d{11}\b", "replacement": "[STEUER_ID]"},
            # Ausweis-/Personalausweisnummern (deutsch, vereinfacht)
            "id_number_de": {
                "pattern": r"\b[A-Z]\d{8}\b",
                "replacement": "[AUSWEIS_NR]",
            },
        }

        # Deutsche Orte/Städte (eine Auswahl für bessere Erkennung)
        self.german_cities = {
            "berlin",
            "hamburg",
            "münchen",
            "köln",
            "frankfurt",
            "stuttgart",
            "düsseldorf",
            "dortmund",
            "essen",
            "leipzig",
            "bremen",
            "dresden",
            "hannover",
            "nürnberg",
            "duisburg",
            "bochum",
            "wuppertal",
            "bielefeld",
            "bonn",
            "münster",
            "karlsruhe",
            "mannheim",
            "augsburg",
            "wiesbaden",
            "gelsenkirchen",
            "mönchengladbach",
            "braunschweig",
            "chemnitz",
            "kiel",
            "aachen",
            "halle",
            "magdeburg",
            "freiburg",
            "krefeld",
            "lübeck",
            "oberhausen",
            "erfurt",
            "mainz",
            "rostock",
            "kassel",
        }

    def _detect_names_with_spacy(self, text: str) -> List[Tuple[str, int, int]]:
        """Erkennt Personennamen mit spaCy"""
        if not SPACY_AVAILABLE or nlp is None:
            return []

        doc = nlp(text)
        names = []

        for ent in doc.ents:
            if ent.label_ in ["PER", "PERSON"]:  # Personennamen
                names.append((ent.text, ent.start_char, ent.end_char))

        return names

    def _detect_locations_with_spacy(self, text: str) -> List[Tuple[str, int, int]]:
        """Erkennt Orte mit spaCy"""
        if not SPACY_AVAILABLE or nlp is None:
            return []

        doc = nlp(text)
        locations = []

        for ent in doc.ents:
            if ent.label_ in ["LOC", "GPE", "ORG"]:  # Orte, geopolitische Entitäten
                # Filtere bekannte deutsche Städte
                if ent.text.lower() in self.german_cities:
                    locations.append((ent.text, ent.start_char, ent.end_char))

        return locations

    def validate(self, text: str) -> ValidationDetails:
        """Validiert Text auf PII und ersetzt gefundene Daten optional"""
        found_pii = {}
        sanitized_text = text
        replacements_made = []
        all_replacements = []

        # 1. Regex-basierte PII-Erkennung
        for pii_type, pattern_info in self.pii_patterns.items():
            pattern = pattern_info["pattern"]
            replacement = pattern_info["replacement"]

            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                found_pii[pii_type] = [match.group() for match in matches]

                for match in matches:
                    original = match.group()
                    start, end = match.span()
                    all_replacements.append(
                        {
                            "type": pii_type,
                            "original": original,
                            "replacement": replacement,
                            "start": start,
                            "end": end,
                        }
                    )

        # 2. spaCy-basierte Named Entity Recognition
        if SPACY_AVAILABLE:
            # Personennamen
            names = self._detect_names_with_spacy(text)
            if names:
                found_pii["person_names"] = [name[0] for name in names]
                for name, start, end in names:
                    all_replacements.append(
                        {
                            "type": "person_names",
                            "original": name,
                            "replacement": "[NAME]",
                            "start": start,
                            "end": end,
                        }
                    )

            # Orte/Städte
            locations = self._detect_locations_with_spacy(text)
            if locations:
                found_pii["locations"] = [loc[0] for loc in locations]
                for location, start, end in locations:
                    all_replacements.append(
                        {
                            "type": "locations",
                            "original": location,
                            "replacement": "[ORT]",
                            "start": start,
                            "end": end,
                        }
                    )

        # 3. Text-Sanitisierung wenn gewünscht
        if self.replace_pii and all_replacements:
            # Entferne überlappende Matches
            all_replacements.sort(key=lambda x: (x["start"], -x["end"]))
            filtered_replacements = []

            for repl in all_replacements:
                overlaps = False
                for existing in filtered_replacements:
                    if (
                        repl["start"] < existing["end"]
                        and repl["end"] > existing["start"]
                    ):
                        overlaps = True
                        break
                if not overlaps:
                    filtered_replacements.append(repl)

            # Sortiere rückwärts für korrekte String-Ersetzung
            filtered_replacements.sort(key=lambda x: x["start"], reverse=True)

            # Führe Replacements durch
            for repl in filtered_replacements:
                start, end = repl["start"], repl["end"]
                replacement = repl["replacement"]
                sanitized_text = (
                    sanitized_text[:start] + replacement + sanitized_text[end:]
                )
                replacements_made.append(repl)

        # Ergebnis zusammenstellen
        passed = len(found_pii) == 0
        reason = f"PII detected: {list(found_pii.keys())}" if not passed else None

        details = {
            "found_pii": found_pii,
            "spacy_available": SPACY_AVAILABLE,
            "replacements_made": replacements_made if self.replace_pii else None,
        }

        return ValidationDetails(
            passed=passed,
            validator_name="AdvancedPIIValidator",
            input_text=text,
            sanitized_text=sanitized_text if self.replace_pii else None,
            reason=reason,
            confidence=0.95 if SPACY_AVAILABLE else 0.8,
            details=details,
        )


class CompetitorValidator:
    """Competitor Mention Validator"""

    def __init__(self, competitors: Optional[List[str]] = None):
        self.competitors = [
            comp.lower()
            for comp in (
                competitors
                or [
                    "openai",
                    "chatgpt",
                    "claude",
                    "gemini",
                    "anthropic",
                    "meta",
                    "llama",
                    "microsoft",
                    "copilot",
                    "azure",
                    "aws",
                    "bedrock",
                    "vertex",
                    "palm",
                ]
            )
        ]

    def validate(self, text: str) -> ValidationDetails:
        """Validiert Text auf Wettbewerber-Erwähnungen"""
        text_lower = text.lower()
        found_competitors = []

        for competitor in self.competitors:
            if competitor in text_lower:
                found_competitors.append(competitor)

        passed = len(found_competitors) == 0
        reason = f"Competitors mentioned: {found_competitors}" if not passed else None

        return ValidationDetails(
            passed=passed,
            validator_name="CompetitorValidator",
            input_text=text,
            reason=reason,
            confidence=1.0,
            details={"found_competitors": found_competitors},
        )


class DangerousContentValidator:
    """
    HYBRID-Validator für gefährliche Inhalte:
    1. REGEX findet Trigger-Wörter (schnell)
    2. LLM analysiert Kontext und Absicht (intelligent)

    Beispiel:
    ✅ "Puderzucker über Guggelhupf" → harmlos (backen)
    ❌ "Puderzucker und Salz explodieren" → gefährlich (Sprengstoff-Frage)
    """

    def __init__(
        self,
        llm_endpoint: str = None,
        llm_api_key: str = None,
        llm_model: str = None,
    ):
        """
        Initialisiert den Hybrid-Validator

        Args:
            llm_endpoint: OpenAI-kompatibler API Endpoint
            llm_api_key: API Key für LLM-Zugang
            llm_model: Zu verwendendes LLM-Modell
        """
        # Load from environment variables if not provided
        self.llm_endpoint = llm_endpoint or os.getenv(
            "LLM_ENDPOINT", "http://localhost:11434/v1"
        )
        self.llm_api_key = llm_api_key or os.getenv("LLM_API_KEY", "ollama")
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "qwen2.5:latest")
        self.logger = logging.getLogger(__name__)  # Logger hinzufügen

        # TRIGGER-WÖRTER für schnelle Regex-Erkennung
        self.danger_triggers = {
            # Sprengstoff-relevante Wörter
            "explosives": [
                "sprengstoff",
                "bombe",
                "explosion",
                "explodieren",
                "detonation",
                "tnt",
                "c4",
                "plastiksprengstoff",
                "puderzucker",
                "zucker",
                "salz",
                "kochsalz",
                "fertilizer",
                "dünger",
                "ammonium",
                "nitrat",
                "molotov",
                "thermit",
                "schwarzpulver",
                "napalm",
                "mischen",
                "chemikalien",
                # Erweiterte Sprengstoff-Chemikalien
                "toluol",
                "schwefel",
                "salpetersäure",
                "schwefelsäure",
                "glycerin",
                "nitroglycerin",
                "peroxide",
                "wasserstoffperoxid",
                "aceton",
                "hexamin",
                "rdx",
                "petn",
                "semtex",
                "tannerit",
                "kaliumnitrat",
                "kaliumchlorat",
                "kaliumperchlorat",
                "aluminiumpulver",
                "eisenpulver",
                "magnesium",
                "phosphor",
                "natriumchlorat",
                "triacetontriperoxid",
                "tatp",
                "hmtd",
                "picrat",
                "pikrinsäure",
                "tetryl",
                "composition",
                "plastique",
                "det cord",
                "zünder",
                "sprengkapsel",
                "detonator",
                "timer",
                "fernzündung",
            ],
            # Waffen-relevante Wörter
            "weapons": [
                "waffe",
                "pistole",
                "gewehr",
                "schrotflinte",
                "maschinengewehr",
                "ak47",
                "ar15",
                "munition",
                "patronen",
                "kugeln",
                "scharfschütze",
            ],
            # Drogen-relevante Wörter
            "drugs": [
                "drogen",
                "droge",
                "meth",
                "methamphetamine",
                "crystal",
                "lsd",
                "heroin",
                "kokain",
                "ecstasy",
                "mdma",
                "amphetamine",
                "speed",
                # Cannabis-verwandte Substanzen
                "cbd",
                "cannabidiol",
                "thc",
                "cannabis",
                "marihuana",
                "marijuana",
                "hanf",
                "gras",
                "weed",
                "joint",
                "haschisch",
                "hash",
                "edibles",
                # Weitere Drogen
                "ketamine",
                "pcp",
                "opium",
                "morphin",
                "fentanyl",
                "crack",
                "koks",
                "pilze",
                "magic mushrooms",
                "psilocybin",
                "dmt",
                "ayahuasca",
            ],
            # Illegale Aktivitäten
            "illegal": [
                "hack",
                "hacking",
                "phishing",
                "betrug",
                "darknet",
                "malware",
                "virus",
                "trojaner",
                "geldwäsche",
                "identität",
                "fälschen",
            ],
            # Gewalt und Selbstverletzung
            "violence": [
                "töten",
                "ermorden",
                "umbringen",
                "selbstmord",
                "suizid",
                "folter",
                "vergiften",
                "kidnapping",
                "entführung",
                "attentat",
                "anschlag",
            ],
        }

        # OpenAI Client Setup wenn Parameter verfügbar
        self.use_llm = False
        if self.llm_endpoint and self.llm_api_key:
            try:
                import openai

                self.openai_client = openai.OpenAI(
                    base_url=self.llm_endpoint, api_key=self.llm_api_key
                )
                self.use_llm = True
                self.logger.info(
                    f"✅ LLM initialized: {self.llm_endpoint} with model {self.llm_model}"
                )
            except ImportError:
                self.logger.warning(
                    "OpenAI library not available - falling back to regex-only validation"
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI client: {e}")
        else:
            self.logger.warning(
                f"LLM not initialized - missing endpoint({self.llm_endpoint}) or key({bool(self.llm_api_key)})"
            )

    def _check_trigger_words(self, text: str) -> tuple[bool, list, str]:
        """
        STUFE 1: Schnelle Regex-Suche nach Trigger-Wörtern

        Returns:
            (trigger_found, found_triggers, category)
        """
        text_lower = text.lower()
        found_triggers = []
        trigger_category = None

        for category, triggers in self.danger_triggers.items():
            for trigger in triggers:
                if trigger in text_lower:
                    found_triggers.append(trigger)
                    trigger_category = category

        return len(found_triggers) > 0, found_triggers, trigger_category

    def _analyze_with_llm(
        self, text: str, found_triggers: list, trigger_category: str
    ) -> dict:
        """
        STUFE 2: LLM analysiert Kontext und Absicht

        Args:
            text: Original-Text
            found_triggers: Gefundene Trigger-Wörter
            trigger_category: Kategorie der Trigger

        Returns:
            {danger_level: int, category: str, reasoning: str, is_dangerous: bool}
        """
        if not self.use_llm:
            # Fallback: Regex-basierte Entscheidung
            return {
                "danger_level": 8,
                "category": trigger_category,
                "reasoning": f"Trigger words detected: {found_triggers} (LLM not available)",
                "is_dangerous": True,
            }

        try:
            prompt = f"""Du bist ein Sicherheitsexperte. Analysiere den folgenden Text auf gefährliche Absichten.

TEXT: "{text}"

KONTEXT: 
- Regex hat diese Trigger-Wörter gefunden: {found_triggers}
- Kategorie: {trigger_category}

Beurteile die ABSICHT und den KONTEXT:
- Ist das eine harmlose Erwähnung (z.B. Backen, Kochen, Spiele, Bildung)?
- Oder eine gefährliche Anfrage (Anleitung für Waffen/Sprengstoff/Drogen)?

Antworte NUR in diesem JSON-Format:
{{
  "danger_level": <1-10, wobei 1=harmlos, 10=extrem gefährlich>,
  "category": "<explosives|weapons|drugs|illegal|violence|harmless>",
  "reasoning": "<kurze Begründung der Einschätzung>",
  "is_dangerous": <true|false>
}}"""

            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON Response
            import json

            result = json.loads(content)

            # Validiere Antwort
            required_keys = ["danger_level", "category", "reasoning", "is_dangerous"]
            if all(key in result for key in required_keys):
                return result
            else:
                raise ValueError("Invalid LLM response format")

        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            # Fallback zu konservativer Entscheidung
            return {
                "danger_level": 7,
                "category": trigger_category,
                "reasoning": f"LLM analysis failed, conservative blocking due to triggers: {found_triggers}",
                "is_dangerous": True,
            }

    def validate(self, text: str) -> ValidationDetails:
        """
        ERWEITERTE HYBRID-VALIDIERUNG:
        1. Regex sucht Trigger-Wörter (schnell)
        2. LLM analysiert IMMER den Kontext (intelligent)
        3. Falls LLM ohne Regex-Trigger etwas findet → LERNEN!
        """
        # STUFE 1: Trigger-Wort Suche
        has_triggers, found_triggers, trigger_category = self._check_trigger_words(text)

        # STUFE 2: LLM analysiert IMMER (nicht nur bei Triggern!)
        if self.use_llm:
            llm_result = self._analyze_with_llm(
                text, found_triggers, trigger_category or "unknown"
            )
            is_dangerous = llm_result["is_dangerous"]

            # LERN-MECHANISMUS: LLM fand Gefahr, aber Regex nicht!
            if is_dangerous and not has_triggers:
                self._log_learning_opportunity(text, llm_result)

            confidence = 0.9
            analysis_method = "hybrid_llm_primary"
        else:
            # Fallback ohne LLM - nur Regex
            if has_triggers:
                is_dangerous = True
                llm_result = {
                    "danger_level": 8,
                    "category": trigger_category,
                    "reasoning": f"Trigger words detected: {found_triggers} (no LLM available)",
                    "is_dangerous": True,
                }
                analysis_method = "regex_only"
                confidence = 0.7
            else:
                is_dangerous = False
                llm_result = {
                    "danger_level": 1,
                    "category": "harmless",
                    "reasoning": "No triggers found and no LLM available",
                    "is_dangerous": False,
                }
                analysis_method = "regex_only"
                confidence = 0.6  # Niedrige Konfidenz ohne LLM

        reason = None
        if is_dangerous:
            trigger_info = (
                f" (Triggers: {found_triggers})"
                if found_triggers
                else " (No regex triggers)"
            )
            reason = f"Dangerous content detected - {llm_result['reasoning']} (Level: {llm_result['danger_level']}/10){trigger_info}"

        return ValidationDetails(
            passed=not is_dangerous,
            validator_name="DangerousContentValidator",
            input_text=text,
            reason=reason,
            confidence=confidence,
            details={
                "analysis_method": analysis_method,
                "triggers_found": found_triggers,
                "trigger_category": trigger_category,
                "llm_used": self.use_llm,
                "llm_analysis": llm_result,
                "danger_level": llm_result["danger_level"],
                "learning_triggered": is_dangerous
                and not has_triggers
                and self.use_llm,
            },
        )

    def _log_learning_opportunity(self, text: str, llm_result: dict):
        """
        LERN-MECHANISMUS: Protokolliert gefährliche Inhalte, die Regex übersehen hat

        Args:
            text: Der gefährliche Text
            llm_result: LLM-Analyse-Ergebnis
        """
        import json
        from datetime import datetime

        learning_data = {
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "danger_level": llm_result.get("danger_level", 0),
            "category": llm_result.get("category", "unknown"),
            "reasoning": llm_result.get("reasoning", ""),
            "suggested_triggers": self._extract_potential_triggers(text, llm_result),
        }

        # Log für Entwickler
        self.logger.warning(
            "🧠 LEARNING OPPORTUNITY: LLM detected danger without regex triggers!"
        )
        self.logger.warning(f"📝 Text: '{text}'")
        self.logger.warning(f"⚠️  Level: {llm_result.get('danger_level')}/10")
        self.logger.warning(f"📂 Category: {llm_result.get('category')}")
        self.logger.warning(f"💭 Reasoning: {llm_result.get('reasoning')}")

        # Speichere in Datei für spätere Analyse
        try:
            learning_file = "guardrails_learning.jsonl"
            with open(learning_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(learning_data, ensure_ascii=False) + "\n")
            self.logger.info(f"💾 Learning data saved to {learning_file}")
        except Exception as e:
            self.logger.error(f"Failed to save learning data: {e}")

    def _extract_potential_triggers(self, text: str, llm_result: dict) -> list:
        """
        Extrahiert potentielle neue Trigger-Wörter aus dem Text

        Returns:
            List von Wörtern, die als Trigger hinzugefügt werden könnten
        """
        import re

        # Einfache Wort-Extraktion (kann später verbessert werden)
        words = re.findall(r"\b\w{3,}\b", text.lower())

        # Filtere nur "verdächtige" Wörter
        potential_triggers = []

        for word in words:
            # Filtere häufige/harmlose Wörter
            if word not in [
                "ich",
                "und",
                "das",
                "ist",
                "mit",
                "für",
                "ein",
                "der",
                "die",
                "dann",
                "wenn",
                "aber",
                "über",
                "kann",
                "haben",
                "werden",
                "sind",
                "wurde",
                "einen",
                "einer",
                "bekomme",
                "mache",
            ]:
                potential_triggers.append(word)

        return potential_triggers[:5]  # Max 5 Vorschläge


class ModernGuardrailsValidator:
    """
    Moderne Guardrails-Validator für GuardRAG
    Integriert alle Validatoren und bietet ein einheitliches Interface
    """

    def __init__(
        self,
        enable_profanity: bool = True,
        enable_toxicity: bool = True,
        enable_pii: bool = True,
        enable_competitor: bool = True,
        enable_dangerous_content: bool = True,
        sanitize_pii: bool = True,
        custom_competitors: Optional[List[str]] = None,
        # NEU: LLM-Parameter für intelligente Dangerous Content Detection
        llm_endpoint: str = None,
        llm_api_key: str = None,
        llm_model: str = None,
    ):
        self.validators = []

        if enable_profanity:
            self.validators.append(ProfanityValidator())

        if enable_toxicity:
            self.validators.append(ToxicLanguageValidator())

        if enable_pii:
            self.validators.append(AdvancedPIIValidator(replace_pii=sanitize_pii))

        if enable_competitor:
            self.validators.append(CompetitorValidator(custom_competitors))

        if enable_dangerous_content:
            # HYBRID-Validator mit LLM-Integration
            self.validators.append(
                DangerousContentValidator(
                    llm_endpoint=llm_endpoint,
                    llm_api_key=llm_api_key,
                    llm_model=llm_model,
                )
            )

        self.logger = logging.getLogger(__name__)

    def validate_input(self, text: str) -> Dict[str, Any]:
        """
        Validiert Input-Text mit allen aktivierten Validatoren

        Returns:
            Dict mit Validierungsergebnissen und optional sanitisiertem Text
        """
        start_time = time.time()
        results = []
        sanitized_text = text
        overall_passed = True

        for validator in self.validators:
            try:
                result = validator.validate(text)

                # Aktualisiere sanitized_text wenn verfügbar
                if result.sanitized_text:
                    sanitized_text = result.sanitized_text

                # Sammle Ergebnisse
                results.append(
                    {
                        "validator": result.validator_name,
                        "passed": result.passed,
                        "reason": result.reason,
                        "confidence": result.confidence,
                        "details": result.details,
                    }
                )

                if not result.passed:
                    overall_passed = False

            except Exception as e:
                self.logger.error(
                    f"Validator {validator.__class__.__name__} failed: {e}"
                )
                results.append(
                    {
                        "validator": validator.__class__.__name__,
                        "passed": False,
                        "reason": f"Validation error: {e}",
                        "confidence": 0.0,
                        "details": None,
                    }
                )
                overall_passed = False

        # Gesamtergebnis
        failed_validators = [r["validator"] for r in results if not r["passed"]]
        processing_time = time.time() - start_time

        return {
            "input_text": text,
            "sanitized_text": sanitized_text if sanitized_text != text else None,
            "overall_passed": overall_passed,
            "failed_validators": failed_validators,
            "validator_results": results,
            "processing_time_ms": round(processing_time * 1000, 2),
            "timestamp": time.time(),
        }

    def get_guardrail_result(self, text: str) -> GuardrailResult:
        """
        Gibt ein strukturiertes GuardrailResult zurück
        """
        validation_result = self.validate_input(text)

        if validation_result["overall_passed"]:
            return GuardrailResult(
                result=ValidationResult.ACCEPTED,
                validator_type=GuardrailType.CONTENT_SAFETY,
                reason="All validations passed",
                confidence=1.0,
                sanitized_text=validation_result.get("sanitized_text"),
                details=validation_result,
            )
        else:
            failed_types = []
            reasons = []

            for validator_result in validation_result["validator_results"]:
                if not validator_result["passed"]:
                    validator_name = validator_result["validator"]
                    if "Profanity" in validator_name:
                        failed_types.append(GuardrailType.PROFANITY)
                    elif "Toxic" in validator_name:
                        failed_types.append(GuardrailType.TOXICITY)
                    elif "PII" in validator_name:
                        failed_types.append(GuardrailType.PII)
                    elif "Competitor" in validator_name:
                        failed_types.append(GuardrailType.COMPETITOR)
                    elif "DangerousContent" in validator_name:
                        failed_types.append(GuardrailType.DANGEROUS_CONTENT)

                    reasons.append(validator_result["reason"])

            primary_type = (
                failed_types[0] if failed_types else GuardrailType.CONTENT_SAFETY
            )
            combined_reason = "; ".join(reasons)

            return GuardrailResult(
                result=ValidationResult.REJECTED,
                validator_type=primary_type,
                reason=combined_reason,
                confidence=0.9,
                sanitized_text=validation_result.get("sanitized_text"),
                details=validation_result,
                suggestions=[
                    "Please rephrase your request without inappropriate content."
                ],
            )


# Convenience function for easy integration
def create_guardrails_validator(
    enable_profanity: bool = True,
    enable_toxicity: bool = True,
    enable_pii: bool = True,
    enable_competitor: bool = True,
    enable_dangerous_content: bool = True,
    sanitize_pii: bool = True,
    custom_competitors: Optional[List[str]] = None,
    # NEU: LLM-Parameter für intelligente Dangerous Content Detection
    llm_endpoint: str = None,
    llm_api_key: str = None,
    llm_model: str = None,
) -> ModernGuardrailsValidator:
    """
    Factory function to create a guardrails validator with sensible defaults
    Unterstützt jetzt auch LLM-Parameter für intelligente Dangerous Content Detection
    """
    return ModernGuardrailsValidator(
        enable_profanity=enable_profanity,
        enable_toxicity=enable_toxicity,
        enable_pii=enable_pii,
        enable_competitor=enable_competitor,
        enable_dangerous_content=enable_dangerous_content,
        sanitize_pii=sanitize_pii,
        custom_competitors=custom_competitors,
        llm_endpoint=llm_endpoint,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )


# Export key classes for external use
__all__ = [
    "ModernGuardrailsValidator",
    "GuardrailResult",
    "ValidationResult",
    "GuardrailType",
    "create_guardrails_validator",
]
