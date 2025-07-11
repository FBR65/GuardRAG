#!/usr/bin/env python3
"""
Enhanced RAG Agent Module
Koordiniert die gesamte RAG-Pipeline mit modernen Guardrails
"""

import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field

from .colpali_integration import COLPALIProcessor, RetrievalResult
from .modern_input_guardrail import ModernInputGuardrail, InputValidationResult
from .output_guardrail import OutputGuardrail, OutputValidationResult
from .qdrant_integration import QdrantConfig

logger = logging.getLogger(__name__)


@dataclass
class EnhancedRAGResponse:
    """Erweiterte RAG-Antwort mit detaillierten Metadaten"""

    answer: str
    sanitized_answer: Optional[str]  # PII-bereinigter Text falls erforderlich
    sources: List[RetrievalResult]
    confidence: float
    processing_time: float
    input_validation: InputValidationResult
    output_validation: Optional[OutputValidationResult]
    guardrail_checks: Dict[str, Any]
    warnings: List[str]
    statistics: Dict[str, Any]


class RAGGenerationResponse(BaseModel):
    """Response model für RAG generation agent"""

    answer: str = Field(description="Generated answer based on the retrieved documents")
    confidence: float = Field(description="Confidence in the answer", ge=0.0, le=1.0)
    source_citations: List[str] = Field(description="Citations to source pages used")


class EnhancedGuardRAGAgent:
    """
    Erweiterte RAG Agent mit modernen Guardrails und deutscher PII-Unterstützung
    """

    def __init__(
        self,
        llm_endpoint: str,
        llm_api_key: str,
        llm_model: str = "qwen2.5:latest",
        colpali_model: str = "vidore/colqwen2.5-v0.2",
        enable_input_guardrails: bool = True,
        enable_output_guardrails: bool = True,
        enable_pii_sanitization: bool = True,
        enable_competitor_detection: bool = True,
        custom_competitors: Optional[List[str]] = None,
        max_retrieval_results: int = 5,
        device: str = "cuda",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ):
        """
        Initialisiert den erweiterten GuardRAG agent

        Args:
            llm_endpoint: LLM API endpoint
            llm_api_key: API key
            llm_model: Model name für generation
            colpali_model: COLPALI model identifier
            enable_input_guardrails: Aktiviert Input-Validierung
            enable_output_guardrails: Aktiviert Output-Validierung
            enable_pii_sanitization: Aktiviert PII-Sanitisierung
            enable_competitor_detection: Aktiviert Konkurrenten-Erkennung
            custom_competitors: Benutzerdefinierte Konkurrenten-Liste
            max_retrieval_results: Maximale Anzahl von Dokumenten für Retrieval
            device: Device für COLPALI inference
            qdrant_host: Qdrant host
            qdrant_port: Qdrant port
            qdrant_url: Qdrant URL (alternativ zu host/port)
            qdrant_api_key: Qdrant API key
        """
        self.llm_endpoint = llm_endpoint
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.enable_input_guardrails = enable_input_guardrails
        self.enable_output_guardrails = enable_output_guardrails
        self.max_retrieval_results = max_retrieval_results

        # Performance tracking
        self.performance_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "blocked_queries": 0,
            "avg_processing_time": 0.0,
            "avg_retrieval_time": 0.0,
            "avg_generation_time": 0.0,
        }

        logger.info("Initializing Enhanced GuardRAG components...")

        # Initialisiere COLPALI processor mit Qdrant
        qdrant_config = QdrantConfig(
            host=qdrant_host,
            port=qdrant_port,
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        self.colpali = COLPALIProcessor(
            model_name=colpali_model,
            device=device,
            qdrant_config=qdrant_config,
        )

        # Initialisiere moderne Input Guardrails
        if self.enable_input_guardrails:
            self.input_guardrail = ModernInputGuardrail(
                enable_profanity=True,
                enable_toxicity=True,
                enable_pii=True,
                enable_competitor=enable_competitor_detection,
                sanitize_pii=enable_pii_sanitization,
                custom_competitors=custom_competitors,
                danger_threshold=5,
            )
            logger.info("Modern input guardrails initialized")

        # Initialisiere Output Guardrails (behalte bestehende Implementierung)
        if self.enable_output_guardrails:
            self.output_guardrail = OutputGuardrail(
                llm_endpoint=llm_endpoint,
                llm_api_key=llm_api_key,
                llm_model=llm_model,
                enable_factual_check=True,
                enable_toxicity_check=True,
                enable_completeness_check=True,
                enable_score_check=True,
                relevance_score_threshold=0.2,
            )
            logger.info("Output guardrails initialized")

        # Initialisiere generation agent
        self._init_generation_agent()

        logger.info("Enhanced GuardRAG agent initialized successfully")

    def _init_generation_agent(self):
        """Initialisiert den Generation Agent"""
        try:
            # Verwende denselben Ansatz wie im rag_agent.py
            provider = OpenAIProvider(
                base_url=self.llm_endpoint, 
                api_key=self.llm_api_key or "dummy"
            )
            model = OpenAIModel(
                provider=provider, 
                model_name=self.llm_model
            )

            self.generation_agent = Agent(
                model=model,
                result_type=RAGGenerationResponse,
                retries=3,
                system_prompt="""
                Du bist ein hilfreicher Assistent, der auf Basis der bereitgestellten Dokumente präzise und sachliche Antworten gibt.
                
                Instruktionen:
                1. Beantworte Fragen nur basierend auf den bereitgestellten Dokumenten
                2. Wenn die Informationen nicht ausreichen, sage das ehrlich
                3. Zitiere relevante Stellen aus den Dokumenten
                4. Halte Antworten konzise aber vollständig
                5. Verwende eine professionelle, sachliche Sprache
                6. Gib eine Confidence-Bewertung ab (0.0-1.0)
                """,
            )
            logger.info("Generation agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize generation agent: {e}")
            raise

    async def process_query(
        self, query: str, collection_name: str
    ) -> EnhancedRAGResponse:
        """
        Verarbeitet eine Benutzeranfrage mit vollständiger Guardrails-Validierung

        Args:
            query: Benutzeranfrage
            collection_name: Name der Qdrant-Collection für Retrieval

        Returns:
            EnhancedRAGResponse mit detaillierten Ergebnissen
        """
        start_time = time.time()
        self.performance_stats["total_queries"] += 1

        warnings = []
        sanitized_query = query

        try:
            # 1. Input Guardrails Validierung
            input_validation = None
            if self.enable_input_guardrails:
                logger.info("Running input guardrails validation...")
                input_validation = self.input_guardrail.validate_query(query)

                if not input_validation.is_valid:
                    self.performance_stats["blocked_queries"] += 1
                    logger.warning(
                        f"Query blocked by input guardrails: {input_validation.blocked_reason}"
                    )

                    return EnhancedRAGResponse(
                        answer="Ihre Anfrage wurde aufgrund von Sicherheitsrichtlinien blockiert.",
                        sanitized_answer=None,
                        sources=[],
                        confidence=0.0,
                        processing_time=time.time() - start_time,
                        input_validation=input_validation,
                        output_validation=None,
                        guardrail_checks={
                            "input_blocked": True,
                            "block_reason": input_validation.blocked_reason,
                            "failed_validators": input_validation.failed_validators,
                        },
                        warnings=[f"Query blocked: {input_validation.blocked_reason}"],
                        statistics=self.get_statistics(),
                    )

                # Verwende sanitisierten Text wenn verfügbar
                if input_validation.sanitized_text:
                    sanitized_query = input_validation.sanitized_text
                    warnings.append(
                        "Query was sanitized to remove sensitive information"
                    )
                    logger.info("Query sanitized by input guardrails")

            # 2. Document Retrieval
            logger.info(f"Retrieving documents for query: {sanitized_query[:100]}...")
            retrieval_start = time.time()

            retrieval_results = await self.colpali.query_documents(
                query=sanitized_query,
                collection_name=collection_name,
                top_k=self.max_retrieval_results,
            )

            retrieval_time = time.time() - retrieval_start
            logger.info(
                f"Retrieved {len(retrieval_results)} documents in {retrieval_time:.2f}s"
            )

            if not retrieval_results:
                return EnhancedRAGResponse(
                    answer="Keine relevanten Dokumente gefunden.",
                    sanitized_answer=None,
                    sources=[],
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    input_validation=input_validation,
                    output_validation=None,
                    guardrail_checks={"no_documents_found": True},
                    warnings=["No relevant documents found"],
                    statistics=self.get_statistics(),
                )

            # 3. Answer Generation
            logger.info("Generating answer...")
            generation_start = time.time()

            # Formatiere Dokumente für den Prompt
            formatted_docs = self._format_documents_for_prompt(retrieval_results)

            # Generiere Antwort
            generation_result = await self.generation_agent.run(
                f"""
                Bitte beantworte diese Frage basierend auf den bereitgestellten Dokumenten:
                
                Frage: {sanitized_query}
                
                Verfügbare Dokumente:
                {formatted_docs}
                
                Gib eine präzise Antwort mit entsprechenden Quellenangaben.
                """
            )

            generation_time = time.time() - generation_start
            answer = generation_result.data.answer
            confidence = generation_result.data.confidence

            logger.info(
                f"Generated answer in {generation_time:.2f}s with confidence {confidence}"
            )

            # 4. Output Guardrails Validierung
            output_validation = None
            sanitized_answer = answer

            if self.enable_output_guardrails:
                logger.info("Running output guardrails validation...")
                output_validation = self.output_guardrail.validate_response(
                    answer, sanitized_query, retrieval_results
                )

                if not output_validation.is_valid:
                    warnings.append(
                        f"Output validation warning: {output_validation.reason}"
                    )
                    logger.warning(
                        f"Output validation failed: {output_validation.reason}"
                    )

                    # Hier könnten wir die Antwort modifizieren oder blockieren
                    if output_validation.danger_level > 7:
                        answer = "Die generierte Antwort wurde aufgrund von Sicherheitsrichtlinien blockiert."
                        confidence = 0.0

            # 5. Finale PII-Sanitisierung der Antwort (optional)
            if self.enable_input_guardrails and hasattr(
                self.input_guardrail, "guardrails_validator"
            ):
                final_validation = (
                    self.input_guardrail.guardrails_validator.validate_input(answer)
                )
                if final_validation.get("sanitized_text"):
                    sanitized_answer = final_validation["sanitized_text"]
                    warnings.append(
                        "Answer was sanitized to remove sensitive information"
                    )

            # Aktualisiere Performance-Statistiken
            total_time = time.time() - start_time
            self._update_performance_stats(total_time, retrieval_time, generation_time)
            self.performance_stats["successful_queries"] += 1

            return EnhancedRAGResponse(
                answer=answer,
                sanitized_answer=sanitized_answer
                if sanitized_answer != answer
                else None,
                sources=retrieval_results,
                confidence=confidence,
                processing_time=total_time,
                input_validation=input_validation,
                output_validation=output_validation,
                guardrail_checks={
                    "input_passed": input_validation.is_valid
                    if input_validation
                    else True,
                    "output_passed": output_validation.is_valid
                    if output_validation
                    else True,
                    "sanitization_applied": sanitized_answer != answer
                    or sanitized_query != query,
                },
                warnings=warnings,
                statistics=self.get_statistics(),
            )

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return EnhancedRAGResponse(
                answer=f"Fehler bei der Verarbeitung: {str(e)}",
                sanitized_answer=None,
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
                input_validation=input_validation,
                output_validation=None,
                guardrail_checks={"system_error": True},
                warnings=[f"System error: {str(e)}"],
                statistics=self.get_statistics(),
            )

    def _format_documents_for_prompt(
        self, retrieval_results: List[RetrievalResult]
    ) -> str:
        """Formatiert Retrieval-Ergebnisse für den Prompt"""
        formatted = []
        for i, result in enumerate(retrieval_results, 1):
            formatted.append(f"""
            Dokument {i} (Score: {result.score:.3f}):
            Quelle: {result.source}
            Seite: {result.page}
            Inhalt: {result.text if hasattr(result, "text") else "Text-Extraktion nicht verfügbar"}
            """)
        return "\n".join(formatted)

    def _update_performance_stats(
        self, total_time: float, retrieval_time: float, generation_time: float
    ):
        """Aktualisiert Performance-Statistiken"""
        total_queries = self.performance_stats["total_queries"]

        # Aktualisiere gleitende Durchschnitte
        self.performance_stats["avg_processing_time"] = (
            self.performance_stats["avg_processing_time"] * (total_queries - 1)
            + total_time
        ) / total_queries
        self.performance_stats["avg_retrieval_time"] = (
            self.performance_stats["avg_retrieval_time"] * (total_queries - 1)
            + retrieval_time
        ) / total_queries
        self.performance_stats["avg_generation_time"] = (
            self.performance_stats["avg_generation_time"] * (total_queries - 1)
            + generation_time
        ) / total_queries

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt detaillierte Statistiken zurück"""
        stats = dict(self.performance_stats)

        # Füge Guardrails-Statistiken hinzu
        if self.enable_input_guardrails:
            stats["input_guardrails"] = self.input_guardrail.get_statistics()

        if self.enable_output_guardrails:
            stats["output_guardrails"] = self.output_guardrail.get_statistics()

        return stats

    def reset_statistics(self):
        """Setzt alle Statistiken zurück"""
        self.performance_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "blocked_queries": 0,
            "avg_processing_time": 0.0,
            "avg_retrieval_time": 0.0,
            "avg_generation_time": 0.0,
        }

        if self.enable_input_guardrails:
            self.input_guardrail.reset_statistics()

        logger.info("All statistics reset")

    def update_competitors(self, competitors: List[str]):
        """Aktualisiert die Konkurrenten-Liste"""
        if self.enable_input_guardrails:
            self.input_guardrail.update_competitors(competitors)
            logger.info(f"Updated competitor list with {len(competitors)} entries")

    async def health_check(self) -> Dict[str, Any]:
        """Überprüft die Gesundheit aller Komponenten"""
        health = {"status": "healthy", "components": {}, "timestamp": time.time()}

        try:
            # Test COLPALI
            health["components"]["colpali"] = (
                "healthy" if self.colpali else "unavailable"
            )

            # Test Guardrails
            if self.enable_input_guardrails:
                test_result = self.input_guardrail.validate_query("test query")
                health["components"]["input_guardrails"] = (
                    "healthy" if test_result else "error"
                )

            if self.enable_output_guardrails:
                health["components"]["output_guardrails"] = "healthy"

            # Test Generation Agent
            health["components"]["generation_agent"] = (
                "healthy" if self.generation_agent else "unavailable"
            )

        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
            logger.error(f"Health check failed: {e}")

        return health


# Export key classes
__all__ = ["EnhancedGuardRAGAgent", "EnhancedRAGResponse", "RAGGenerationResponse"]
