"""
Output Guardrails Module
Validates and filters RAG system responses before delivery to users.
"""

import logging
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
import re

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field

from .colpali_integration import RetrievalResult

logger = logging.getLogger(__name__)


class OutputValidationResult(Enum):
    """Output validation result types."""

    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REVISION = "requires_revision"
    WARNING = "warning"


@dataclass
class OutputGuardrailResult:
    """Result of output guardrail validation."""

    result: OutputValidationResult
    reason: str
    confidence: float
    revised_response: Optional[str] = None
    warnings: Optional[List[str]] = None


class ResponseValidationResponse(BaseModel):
    """Response model for output validation agent."""

    is_factually_accurate: bool = Field(
        description="Whether the response is factually accurate based on sources"
    )
    is_complete: bool = Field(
        description="Whether the response adequately addresses the question"
    )
    is_safe: bool = Field(description="Whether the response is safe and appropriate")
    has_hallucinations: bool = Field(
        description="Whether the response contains information not in sources"
    )
    toxicity_score: float = Field(
        description="Toxicity score between 0.0 (safe) and 1.0 (toxic)", ge=0.0, le=1.0
    )
    confidence: float = Field(
        description="Confidence in the assessment", ge=0.0, le=1.0
    )
    issues_found: List[str] = Field(
        default_factory=list, description="List of specific issues identified"
    )
    suggestions: List[str] = Field(
        default_factory=list, description="Suggestions for improvement"
    )


class OutputGuardrail:
    """
    Output validation guardrail that validates responses before delivery.
    """

    def __init__(
        self,
        llm_endpoint: str,
        llm_api_key: str,
        llm_model: str = "qwen2.5:latest",
        enable_factual_check: bool = True,
        enable_toxicity_check: bool = True,
        enable_completeness_check: bool = True,
        enable_score_check: bool = True,
        toxicity_threshold: float = 0.3,
        confidence_threshold: float = 0.7,
        relevance_score_threshold: float = 0.2,
    ):
        """
        Initialize output guardrail.

        Args:
            llm_endpoint: LLM API endpoint (Ollama)
            llm_api_key: API key
            llm_model: Model name
            enable_factual_check: Enable factual accuracy checking
            enable_toxicity_check: Enable toxicity detection
            enable_completeness_check: Enable completeness validation
            enable_score_check: Enable Qdrant relevance score checking
            toxicity_threshold: Threshold for toxicity rejection
            confidence_threshold: Minimum confidence for approval
            relevance_score_threshold: Maximum allowed Qdrant score (lower is better)
        """
        self.llm_endpoint = llm_endpoint
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.enable_factual_check = enable_factual_check
        self.enable_toxicity_check = enable_toxicity_check
        self.enable_completeness_check = enable_completeness_check
        self.enable_score_check = enable_score_check
        self.toxicity_threshold = toxicity_threshold
        self.confidence_threshold = confidence_threshold
        self.relevance_score_threshold = relevance_score_threshold

        # Standard response for rejected content (same as input guardrail)
        self.standard_rejection_response = (
            "ICH HABE KEINE INFORMATIONEN ZU IHRER ANFRAGE VORLIEGEN."
        )

        # Initialize LLM agent
        self._init_llm_agent()

        # Define toxic patterns
        self.toxic_patterns = [
            r"\b(?:hate|hatred|nazi|terrorist)\b",
            r"\b(?:kill|murder|suicide|self-harm)\b",
            r"\b(?:racist|sexist|homophobic)\b",
            r"\b(?:illegal|criminal|fraud)\b",
        ]

        # Define hallucination indicators
        self.hallucination_indicators = [
            "ich glaube",
            "wahrscheinlich",
            "möglicherweise",
            "könnte sein",
            "nicht sicher",
            "vermutlich",
            "eventuell",
            "laut meiner meinung",
        ]

    def _init_llm_agent(self):
        """Initialize the LLM agent for validation."""
        try:
            provider = OpenAIProvider(
                base_url=self.llm_endpoint, api_key=self.llm_api_key
            )
            model = OpenAIModel(provider=provider, model_name=self.llm_model)

            system_prompt = """Du bist ein Experte für die Validierung von KI-generierten Antworten aus einem wissenschaftlichen Dokumentenverständnissystem.

Deine Aufgabe ist es, generierte Antworten zu bewerten bezüglich:

1. FAKTISCHE GENAUIGKEIT: Sind alle Aussagen durch die bereitgestellten Quellen belegbar?
2. VOLLSTÄNDIGKEIT: Beantwortet die Antwort die gestellte Frage angemessen?
3. SICHERHEIT: Enthält die Antwort schädliche, toxische oder unangemessene Inhalte?
4. HALLUZINATIONEN: Werden Informationen hinzugefügt, die nicht in den Quellen stehen?

Bewertungskriterien:
- FAKTISCHE GENAUIGKEIT: Prüfe jeden Fakt gegen die Quellen
- VOLLSTÄNDIGKEIT: Wurde die Frage vollständig beantwortet?
- SICHERHEIT: Keine schädlichen, diskriminierenden oder gefährlichen Inhalte
- HALLUZINATIONEN: Keine Erfindung von Fakten oder Daten

Gib eine detaillierte Bewertung und konkrete Verbesserungsvorschläge."""

            self.validation_agent = Agent(
                model=model,
                result_type=ResponseValidationResponse,
                retries=3,
                system_prompt=system_prompt,
            )

            logger.info("Output validation agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize output validation agent: {e}")
            raise

    async def validate_response(
        self, response: str, query: str, retrieval_results: List[RetrievalResult]
    ) -> OutputGuardrailResult:
        """
        Validate a generated response against multiple criteria.

        Args:
            response: Generated response to validate
            query: Original user query
            retrieval_results: Source documents used for generation

        Returns:
            OutputGuardrailResult with validation outcome
        """
        # Step 1: Check Qdrant relevance scores first
        if self.enable_score_check:
            score_result = self._score_validation(retrieval_results)
            if score_result.result == OutputValidationResult.REJECTED:
                return score_result

        # Step 2: Basic validation
        basic_result = self._basic_validation(response)
        if basic_result.result == OutputValidationResult.REJECTED:
            return basic_result

        # Step 3: Pattern-based checks
        pattern_result = self._pattern_validation(response)
        if pattern_result.result == OutputValidationResult.REJECTED:
            return pattern_result

        # Step 4: Source-based validation
        if self.enable_factual_check:
            source_result = self._source_validation(response, retrieval_results)
            if source_result.result == OutputValidationResult.REJECTED:
                return source_result

        # Step 5: LLM-based comprehensive validation
        try:
            llm_result = await self._llm_validation(response, query, retrieval_results)
            return llm_result

        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return OutputGuardrailResult(
                result=OutputValidationResult.WARNING,
                reason=f"LLM validation nicht verfügbar: {str(e)}",
                confidence=0.5,
                warnings=["Vollständige Validierung nicht möglich"],
            )

    def _score_validation(
        self, retrieval_results: List[RetrievalResult]
    ) -> OutputGuardrailResult:
        """
        Validate Qdrant relevance scores. If all scores are above threshold,
        reject the response with standard message.

        Args:
            retrieval_results: Results from Qdrant retrieval

        Returns:
            OutputGuardrailResult indicating if scores are acceptable
        """
        if not retrieval_results:
            return OutputGuardrailResult(
                result=OutputValidationResult.REJECTED,
                reason=self.standard_rejection_response,
                confidence=1.0,
                revised_response=self.standard_rejection_response,
            )

        # Check if all scores are above threshold (worse than acceptable)
        poor_scores = [
            result.score
            for result in retrieval_results
            if result.score > self.relevance_score_threshold
        ]

        if len(poor_scores) == len(retrieval_results):
            # All results have poor relevance scores
            return OutputGuardrailResult(
                result=OutputValidationResult.REJECTED,
                reason=self.standard_rejection_response,
                confidence=1.0,
                revised_response=self.standard_rejection_response,
            )

        # At least some results have good scores
        good_scores = [
            result.score
            for result in retrieval_results
            if result.score <= self.relevance_score_threshold
        ]

        if good_scores:
            best_score = min(good_scores)
            return OutputGuardrailResult(
                result=OutputValidationResult.APPROVED,
                reason=f"Ausreichende Relevanz gefunden (beste Bewertung: {best_score:.3f})",
                confidence=0.8,
            )

        return OutputGuardrailResult(
            result=OutputValidationResult.APPROVED,
            reason="Score-Validierung bestanden",
            confidence=0.6,
        )

    def _basic_validation(self, response: str) -> OutputGuardrailResult:
        """Basic response validation."""
        response = response.strip()

        # Check if response is empty
        if not response:
            return OutputGuardrailResult(
                result=OutputValidationResult.REJECTED,
                reason="Leere Antwort generiert",
                confidence=1.0,
            )

        # Check response length
        if len(response) < 10:
            return OutputGuardrailResult(
                result=OutputValidationResult.REJECTED,
                reason="Antwort zu kurz und unvollständig",
                confidence=0.9,
            )

        if len(response) > 5000:
            return OutputGuardrailResult(
                result=OutputValidationResult.WARNING,
                reason="Antwort sehr lang, möglicherweise zu ausführlich",
                confidence=0.7,
                warnings=["Antwort könnte gekürzt werden"],
            )

        # Check for obvious errors
        error_indicators = ["error", "fehler", "exception", "traceback", "none", "null"]
        if any(indicator in response.lower() for indicator in error_indicators):
            return OutputGuardrailResult(
                result=OutputValidationResult.REJECTED,
                reason="Antwort enthält Fehlerindikatoren",
                confidence=0.8,
            )

        return OutputGuardrailResult(
            result=OutputValidationResult.APPROVED,
            reason="Grundlegende Validierung bestanden",
            confidence=0.6,
        )

    def _pattern_validation(self, response: str) -> OutputGuardrailResult:
        """Pattern-based validation for toxic content."""
        if not self.enable_toxicity_check:
            return OutputGuardrailResult(
                result=OutputValidationResult.APPROVED,
                reason="Toxizitätsprüfung deaktiviert",
                confidence=0.5,
            )

        response_lower = response.lower()

        # Check for toxic patterns
        for pattern in self.toxic_patterns:
            if re.search(pattern, response_lower):
                return OutputGuardrailResult(
                    result=OutputValidationResult.REJECTED,
                    reason=f"Toxisches Muster gefunden: {pattern}",
                    confidence=0.9,
                )

        # Check for hallucination indicators
        hallucination_count = sum(
            1
            for indicator in self.hallucination_indicators
            if indicator in response_lower
        )

        if hallucination_count > 2:
            return OutputGuardrailResult(
                result=OutputValidationResult.WARNING,
                reason="Mehrere Unsicherheitsindikatoren gefunden",
                confidence=0.7,
                warnings=["Antwort könnte spekulative Inhalte enthalten"],
            )

        return OutputGuardrailResult(
            result=OutputValidationResult.APPROVED,
            reason="Keine problematischen Muster gefunden",
            confidence=0.7,
        )

    def _source_validation(
        self, response: str, retrieval_results: List[RetrievalResult]
    ) -> OutputGuardrailResult:
        """Validate response against source documents."""
        if not retrieval_results:
            return OutputGuardrailResult(
                result=OutputValidationResult.WARNING,
                reason="Keine Quelldokumente für Validierung verfügbar",
                confidence=0.4,
                warnings=["Faktische Genauigkeit nicht prüfbar"],
            )

        # Extract text from sources
        source_texts = []
        for result in retrieval_results:
            if result.page.text_content:
                source_texts.append(result.page.text_content.lower())

        if not source_texts:
            return OutputGuardrailResult(
                result=OutputValidationResult.WARNING,
                reason="Quelltexte nicht verfügbar für Validierung",
                confidence=0.4,
                warnings=["Faktische Genauigkeit eingeschränkt prüfbar"],
            )

        # Simple word overlap check
        response_words = set(re.findall(r"\b\w+\b", response.lower()))
        source_words = set()
        for text in source_texts:
            source_words.update(re.findall(r"\b\w+\b", text))

        # Filter out common words
        common_words = {
            "der",
            "die",
            "das",
            "und",
            "oder",
            "ist",
            "sind",
            "war",
            "waren",
            "ein",
            "eine",
            "this",
            "that",
            "and",
            "or",
            "is",
            "are",
            "was",
            "were",
        }

        response_content_words = response_words - common_words
        overlap = response_content_words.intersection(source_words)

        if response_content_words:
            overlap_ratio = len(overlap) / len(response_content_words)

            if overlap_ratio < 0.3:
                return OutputGuardrailResult(
                    result=OutputValidationResult.WARNING,
                    reason=f"Geringe Übereinstimmung mit Quellen ({overlap_ratio:.2f})",
                    confidence=0.6,
                    warnings=["Mögliche Halluzinationen"],
                )

        return OutputGuardrailResult(
            result=OutputValidationResult.APPROVED,
            reason="Angemessene Übereinstimmung mit Quellen",
            confidence=0.7,
        )

    async def _llm_validation(
        self, response: str, query: str, retrieval_results: List[RetrievalResult]
    ) -> OutputGuardrailResult:
        """LLM-based comprehensive validation."""
        # Prepare source context
        source_context = "\n\n".join(
            [
                f"Quelle {i + 1} (Seite {result.page.page_number}): {result.page.text_content[:500]}..."
                for i, result in enumerate(retrieval_results[:3])
                if result.page.text_content
            ]
        )

        validation_prompt = f"""
Frage: {query}

Generierte Antwort: {response}

Verfügbare Quellen:
{source_context}

Bewerte die generierte Antwort anhand der bereitgestellten Quellen.
"""

        try:
            result = await self.validation_agent.run(validation_prompt)

            # Determine overall result
            if result.data.toxicity_score > self.toxicity_threshold:
                return OutputGuardrailResult(
                    result=OutputValidationResult.REJECTED,
                    reason=f"Toxizitätsschwelle überschritten ({result.data.toxicity_score:.2f})",
                    confidence=result.data.confidence,
                )

            if result.data.has_hallucinations:
                return OutputGuardrailResult(
                    result=OutputValidationResult.REQUIRES_REVISION,
                    reason="Halluzinationen erkannt",
                    confidence=result.data.confidence,
                    warnings=result.data.issues_found,
                )

            if not result.data.is_factually_accurate:
                return OutputGuardrailResult(
                    result=OutputValidationResult.REQUIRES_REVISION,
                    reason="Faktische Ungenauigkeiten erkannt",
                    confidence=result.data.confidence,
                    warnings=result.data.issues_found,
                )

            if not result.data.is_complete:
                return OutputGuardrailResult(
                    result=OutputValidationResult.REQUIRES_REVISION,
                    reason="Unvollständige Antwort",
                    confidence=result.data.confidence,
                    warnings=result.data.issues_found,
                )

            if result.data.confidence < self.confidence_threshold:
                return OutputGuardrailResult(
                    result=OutputValidationResult.WARNING,
                    reason=f"Niedrige Vertrauenswürdigkeit ({result.data.confidence:.2f})",
                    confidence=result.data.confidence,
                    warnings=result.data.issues_found,
                )

            return OutputGuardrailResult(
                result=OutputValidationResult.APPROVED,
                reason="Alle Validierungen bestanden",
                confidence=result.data.confidence,
            )

        except Exception as e:
            logger.error(f"LLM validation error: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get output guardrail statistics."""
        return {
            "factual_check_enabled": self.enable_factual_check,
            "toxicity_check_enabled": self.enable_toxicity_check,
            "completeness_check_enabled": self.enable_completeness_check,
            "score_check_enabled": self.enable_score_check,
            "toxicity_threshold": self.toxicity_threshold,
            "confidence_threshold": self.confidence_threshold,
            "relevance_score_threshold": self.relevance_score_threshold,
            "model": self.llm_model,
            "toxic_patterns_count": len(self.toxic_patterns),
            "standard_rejection_response": self.standard_rejection_response,
        }

    def get_security_stats(self) -> Dict[str, Any]:
        """Get security-focused statistics."""
        return self.get_stats()
