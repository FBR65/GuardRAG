#!/usr/bin/env python3
"""
Modern Input Guardrails für GuardRAG System
Integriert erweiterte Toxizitäts-, PII- und Content-Validierung
"""

import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv is optional

try:
    from .modern_guardrails import (
        GuardrailResult,
        ValidationResult,
        create_guardrails_validator,
    )
except ImportError:
    # Fallback für direktes Ausführen
    from modern_guardrails import (
        GuardrailResult,
        ValidationResult,
        create_guardrails_validator,
    )

logger = logging.getLogger(__name__)


@dataclass
class InputValidationResult:
    """Ergebnis der Input-Validierung mit erweiterten Informationen"""

    is_valid: bool
    sanitized_text: Optional[str]
    blocked_reason: Optional[str]
    confidence: float
    processing_time_ms: float
    failed_validators: List[str]
    suggestions: Optional[List[str]]
    details: Dict[str, Any]


class ModernInputGuardrail:
    """
    Moderne Input Guardrails für GuardRAG
    Verwendet das erweiterte Guardrails-System mit deutscher und englischer Unterstützung
    """

    def __init__(
        self,
        enable_profanity: bool = True,
        enable_toxicity: bool = True,
        enable_pii: bool = True,
        enable_competitor: bool = True,
        enable_dangerous_content: bool = True,  # NEU: Gefährliche Inhalte aktiviert
        sanitize_pii: bool = True,
        custom_competitors: Optional[List[str]] = None,
        danger_threshold: int = 5,
        # NEU: LLM-Parameter für intelligente Dangerous Content Detection
        llm_endpoint: str = None,
        llm_api_key: str = None,
        llm_model: str = None,
    ):
        """
        Initialisiert das moderne Input Guardrail System

        Args:
            enable_profanity: Aktiviert Profanitätserkennung
            enable_toxicity: Aktiviert Toxizitätserkennung
            enable_pii: Aktiviert PII-Erkennung
            enable_competitor: Aktiviert Konkurrenten-Erkennung
            enable_dangerous_content: Aktiviert gefährliche Inhalte Erkennung
            sanitize_pii: Aktiviert PII-Sanitisierung
            custom_competitors: Benutzerdefinierte Konkurrenten-Liste
            danger_threshold: Schwellenwert für Gefährlichkeit (1-10)
            llm_endpoint: OpenAI-kompatibler API Endpoint für intelligente Dangerous Content Detection
            llm_api_key: API Key für LLM-Zugang
            llm_model: Zu verwendendes LLM-Modell
        """
        self.danger_threshold = danger_threshold

        # Load LLM config from environment if not provided
        llm_endpoint = llm_endpoint or os.getenv("LLM_ENDPOINT")
        llm_api_key = llm_api_key or os.getenv("LLM_API_KEY")
        llm_model = llm_model or os.getenv("LLM_MODEL")

        # Initialisiere das Guardrails-System mit LLM-Unterstützung
        self.guardrails_validator = create_guardrails_validator(
            enable_profanity=enable_profanity,
            enable_toxicity=enable_toxicity,
            enable_pii=enable_pii,
            enable_competitor=enable_competitor,
            enable_dangerous_content=enable_dangerous_content,  # NEU: Parameter hinzugefügt
            sanitize_pii=sanitize_pii,
            custom_competitors=custom_competitors,
            # NEU: LLM-Parameter für Hybrid-Dangerous-Content-Detection
            llm_endpoint=llm_endpoint,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
        )

        # Statistiken
        self.validation_stats = {
            "total_queries": 0,
            "blocked_queries": 0,
            "blocked_by_profanity": 0,
            "blocked_by_toxicity": 0,
            "blocked_by_pii": 0,
            "blocked_by_competitor": 0,
            "blocked_by_dangerous_content": 0,  # NEU: Statistik für gefährliche Inhalte
            "sanitized_pii": 0,
            "avg_processing_time_ms": 0.0,
        }

        logger.info("Modern Input Guardrail initialized")

    def validate_query(self, query: str) -> InputValidationResult:
        """
        Validiert eine Benutzeranfrage mit dem modernen Guardrails-System

        Args:
            query: Die zu validierende Benutzeranfrage

        Returns:
            InputValidationResult mit detaillierten Validierungsinformationen
        """
        self.validation_stats["total_queries"] += 1

        try:
            # Führe Guardrails-Validierung durch
            guardrail_result = self.guardrails_validator.get_guardrail_result(query)
            validation_details = guardrail_result.details

            # Aktualisiere Statistiken
            self._update_stats(guardrail_result, validation_details)

            # Bestimme Ergebnis
            is_valid = guardrail_result.result == ValidationResult.ACCEPTED

            if not is_valid:
                self.validation_stats["blocked_queries"] += 1

            # Erstelle Ergebnis
            result = InputValidationResult(
                is_valid=is_valid,
                sanitized_text=guardrail_result.sanitized_text,
                blocked_reason=guardrail_result.reason if not is_valid else None,
                confidence=guardrail_result.confidence,
                processing_time_ms=validation_details.get("processing_time_ms", 0.0),
                failed_validators=validation_details.get("failed_validators", []),
                suggestions=guardrail_result.suggestions,
                details=validation_details,
            )

            # Aktualisiere durchschnittliche Verarbeitungszeit
            self._update_avg_processing_time(result.processing_time_ms)

            # Logge Ergebnis
            if not is_valid:
                logger.warning(f"Query blocked: {guardrail_result.reason}")
            else:
                logger.debug(
                    f"Query validated successfully in {result.processing_time_ms}ms"
                )

            return result

        except Exception as e:
            logger.error(f"Error during query validation: {e}")
            self.validation_stats["blocked_queries"] += 1

            return InputValidationResult(
                is_valid=False,
                sanitized_text=None,
                blocked_reason=f"Validation error: {str(e)}",
                confidence=0.0,
                processing_time_ms=0.0,
                failed_validators=["SystemError"],
                suggestions=["Please try rephrasing your query."],
                details={"error": str(e)},
            )

    def _update_stats(
        self, guardrail_result: GuardrailResult, validation_details: Dict[str, Any]
    ):
        """Aktualisiert die Validierungsstatistiken"""
        if guardrail_result.result != ValidationResult.ACCEPTED:
            failed_validators = validation_details.get("failed_validators", [])

            for validator in failed_validators:
                if "Profanity" in validator:
                    self.validation_stats["blocked_by_profanity"] += 1
                elif "Toxic" in validator:
                    self.validation_stats["blocked_by_toxicity"] += 1
                elif "PII" in validator:
                    self.validation_stats["blocked_by_pii"] += 1
                elif "Competitor" in validator:
                    self.validation_stats["blocked_by_competitor"] += 1

        # Zähle PII-Sanitisierungen
        if guardrail_result.sanitized_text:
            self.validation_stats["sanitized_pii"] += 1

    def _update_avg_processing_time(self, processing_time_ms: float):
        """Aktualisiert die durchschnittliche Verarbeitungszeit"""
        total = self.validation_stats["total_queries"]
        current_avg = self.validation_stats["avg_processing_time_ms"]

        # Berechne neuen Durchschnitt
        new_avg = ((current_avg * (total - 1)) + processing_time_ms) / total
        self.validation_stats["avg_processing_time_ms"] = round(new_avg, 2)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Gibt detaillierte Validierungsstatistiken zurück

        Returns:
            Dictionary mit Statistiken
        """
        total = self.validation_stats["total_queries"]
        blocked = self.validation_stats["blocked_queries"]

        return {
            **self.validation_stats,
            "block_rate_percent": round(
                (blocked / total * 100) if total > 0 else 0.0, 2
            ),
            "success_rate_percent": round(
                ((total - blocked) / total * 100) if total > 0 else 0.0, 2
            ),
        }

    def reset_statistics(self):
        """Setzt alle Statistiken zurück"""
        self.validation_stats = {
            "total_queries": 0,
            "blocked_queries": 0,
            "blocked_by_profanity": 0,
            "blocked_by_toxicity": 0,
            "blocked_by_pii": 0,
            "blocked_by_competitor": 0,
            "sanitized_pii": 0,
            "avg_processing_time_ms": 0.0,
        }
        logger.info("Validation statistics reset")

    def update_competitors(self, competitors: List[str]):
        """
        Aktualisiert die Liste der zu blockierenden Konkurrenten

        Args:
            competitors: Liste der Konkurrenten-Namen
        """
        # Erstelle neuen Validator mit aktualisierten Konkurrenten
        self.guardrails_validator = create_guardrails_validator(
            enable_profanity=True,
            enable_toxicity=True,
            enable_pii=True,
            enable_competitor=True,
            sanitize_pii=True,
            custom_competitors=competitors,
        )
        logger.info(f"Updated competitor list with {len(competitors)} entries")


# Convenience Functions für einfache Integration


def validate_input_text(
    text: str, sanitize_pii: bool = True, include_competitors: bool = True
) -> InputValidationResult:
    """
    Einfache Funktion zur Input-Validierung

    Args:
        text: Zu validierender Text
        sanitize_pii: Ob PII sanitisiert werden soll
        include_competitors: Ob Konkurrenten-Validierung aktiv sein soll

    Returns:
        InputValidationResult
    """
    guardrail = ModernInputGuardrail(
        sanitize_pii=sanitize_pii, enable_competitor=include_competitors
    )
    return guardrail.validate_query(text)


def is_text_safe(text: str) -> bool:
    """
    Einfache Sicherheitsprüfung für Text

    Args:
        text: Zu prüfender Text

    Returns:
        True wenn Text sicher ist, False wenn blockiert werden sollte
    """
    result = validate_input_text(text)
    return result.is_valid


# Export key classes and functions
__all__ = [
    "ModernInputGuardrail",
    "InputValidationResult",
    "validate_input_text",
    "is_text_safe",
]
