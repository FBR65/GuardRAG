"""
RAG Agent Module
Coordinates the entire RAG pipeline with integrated guardrails.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field

from .colpali_integration import RetrievalResult
from .colpali_manager import COLPALIManager
from .output_guardrail import OutputGuardrail, OutputValidationResult
from .qdrant_integration import QdrantConfig

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Complete RAG response with metadata."""

    answer: str
    sources: List[RetrievalResult]
    confidence: float
    processing_time: float
    guardrail_checks: Dict[str, Any]
    warnings: List[str]


class RAGGenerationResponse(BaseModel):
    """Response model for RAG generation agent."""

    answer: str = Field(description="Generated answer based on the retrieved documents")
    confidence: float = Field(description="Confidence in the answer", ge=0.0, le=1.0)
    source_citations: List[str] = Field(description="Citations to source pages used")


class GuardRAGAgent:
    """
    Main RAG agent with integrated COLPALI and guardrails.
    """

    def __init__(
        self,
        llm_endpoint: str,
        llm_api_key: str,
        llm_model: str = "qwen2.5:latest",
        colpali_model: str = "vidore/colqwen2.5-v0.2",
        enable_output_guardrails: bool = True,
        max_retrieval_results: int = 5,
        device: str = "cuda",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ):
        """
        Initialize GuardRAG agent.

        Args:
            llm_endpoint: LLM API endpoint
            llm_api_key: API key
            llm_model: Model name for generation
            colpali_model: COLPALI model identifier
            enable_output_guardrails: Enable output validation
            max_retrieval_results: Maximum number of documents to retrieve
            device: Device for COLPALI inference
            qdrant_host: Qdrant host
            qdrant_port: Qdrant port
            qdrant_url: Qdrant URL (alternative to host/port)
            qdrant_api_key: Qdrant API key
        """
        self.llm_endpoint = llm_endpoint
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.enable_output_guardrails = enable_output_guardrails
        self.max_retrieval_results = max_retrieval_results

        # Initialize components
        logger.info("Initializing GuardRAG components...")

        # Initialize COLPALI processor with Qdrant (using shared manager)
        qdrant_config = QdrantConfig(
            host=qdrant_host,
            port=qdrant_port,
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        self.colpali = COLPALIManager.get_instance(
            model_name=colpali_model,
            device=device,
            qdrant_config=qdrant_config,
        )

        # Initialize guardrails with pattern-based validation (no external downloads)
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

        # Initialize generation agent
        self._init_generation_agent()

        logger.info("GuardRAG agent initialized successfully")

    def _init_generation_agent(self):
        """Initialize the response generation agent."""
        try:
            provider = OpenAIProvider(
                base_url=self.llm_endpoint, api_key=self.llm_api_key
            )
            model = OpenAIModel(provider=provider, model_name=self.llm_model)

            system_prompt = """Du bist ein KI-Assistent, der wissenschaftliche Dokumente analysiert und präzise Antworten basierend auf den bereitgestellten Quellen gibt.

WICHTIGE REGELN:
1. Antworte NUR basierend auf den bereitgestellten Dokumentenseiten
2. Zitiere immer die Seitenzahlen deiner Quellen
3. Wenn Informationen nicht in den Quellen verfügbar sind, sage das deutlich
4. Sei präzise und faktisch korrekt
5. Verwende eine wissenschaftliche, professionelle Sprache
6. Strukturiere deine Antworten klar und verständlich

ANTWORTFORMAT:
- Beginne mit einer direkten Antwort auf die Frage
- Führe unterstützende Details aus den Quellen an
- Beende mit einer Zusammenfassung oder Schlussfolgerung
- Nutze Quellenangaben in der Form [Seite X]

Beispiel:
"Basierend auf den bereitgestellten Dokumenten zeigt die Studie, dass... [Seite 3]. Die Methodik umfasste... [Seite 5]. Zusammenfassend lässt sich feststellen, dass..."

Gib niemals Informationen aus, die nicht in den bereitgestellten Quellen stehen."""

            self.generation_agent = Agent(
                model=model,
                result_type=RAGGenerationResponse,
                retries=3,
                system_prompt=system_prompt,
            )

            logger.info("Generation agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize generation agent: {e}")
            raise

    async def process_query(self, query: str) -> RAGResponse:
        """
        Process a complete RAG query with guardrails.

        Args:
            query: User query

        Returns:
            RAGResponse with answer and metadata
        """
        import time

        start_time = time.time()

        guardrail_checks = {}
        warnings = []

        try:
            # Step 1: Document retrieval (no input validation for authenticated users)
            logger.info(f"Retrieving documents for query: {query}")
            retrieval_results = self.colpali.search(
                query=query, top_k=self.max_retrieval_results, return_explanations=True
            )

            if not retrieval_results:
                return RAGResponse(
                    answer="Keine relevanten Dokumente gefunden. Bitte stellen Sie sicher, dass Dokumente geladen wurden.",
                    sources=[],
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    guardrail_checks=guardrail_checks,
                    warnings=["Keine Dokumente im Index verfügbar"],
                )

            # Step 2: Response generation
            logger.info("Generating response...")
            response = await self._generate_response(query, retrieval_results)

            # Step 3: Output validation
            if self.enable_output_guardrails:
                logger.info("Running output guardrails...")
                output_result = await self.output_guardrail.validate_response(
                    response=response.answer,
                    query=query,
                    retrieval_results=retrieval_results,
                )

                guardrail_checks["output_validation"] = {
                    "result": output_result.result.value,
                    "reason": output_result.reason,
                    "confidence": output_result.confidence,
                }

                if output_result.result == OutputValidationResult.REJECTED:
                    return RAGResponse(
                        answer="Die generierte Antwort konnte die Sicherheitsprüfungen nicht bestehen. Bitte formulieren Sie Ihre Frage anders.",
                        sources=retrieval_results,
                        confidence=0.0,
                        processing_time=time.time() - start_time,
                        guardrail_checks=guardrail_checks,
                        warnings=[output_result.reason],
                    )

                # Handle revised output
                if output_result.result == OutputValidationResult.REQUIRES_REVISION:
                    warnings.append(f"Antwort überarbeitet: {output_result.reason}")
                    if output_result.revised_response:
                        response.answer = output_result.revised_response

                # Handle warnings
                if output_result.result == OutputValidationResult.WARNING:
                    warnings.append(f"Antwortqualität: {output_result.reason}")

            # Step 4: Return final response
            processing_time = time.time() - start_time

            return RAGResponse(
                answer=response.answer,
                sources=retrieval_results,
                confidence=response.confidence,
                processing_time=processing_time,
                guardrail_checks=guardrail_checks,
                warnings=warnings,
            )

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return RAGResponse(
                answer=f"Ein Fehler ist aufgetreten: {str(e)}",
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
                guardrail_checks=guardrail_checks,
                warnings=[f"Verarbeitungsfehler: {str(e)}"],
            )

    async def _generate_response(
        self, query: str, retrieval_results: List[RetrievalResult]
    ) -> RAGGenerationResponse:
        """Generate response based on retrieved documents."""
        # Prepare context from retrieved documents
        context_parts = []
        for i, result in enumerate(retrieval_results):
            page_text = result.page.text_content or "Kein Text verfügbar"
            context_parts.append(
                f"Dokument {i + 1} (Seite {result.page.page_number}, Relevanz: {result.score:.3f}):\n{page_text[:1000]}..."
            )

        context = "\n\n".join(context_parts)

        generation_prompt = f"""
Frage: {query}

Verfügbare Dokumentenseiten:
{context}

Beantworte die Frage basierend ausschließlich auf den bereitgestellten Dokumentenseiten. 
Zitiere die Seitenzahlen für deine Aussagen.
"""

        try:
            result = await self.generation_agent.run(generation_prompt)
            return result.data

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

    def load_document(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Load a PDF document into the system.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Loading statistics
        """
        try:
            pages = self.colpali.process_document(pdf_path)
            index_info = self.colpali.get_index_info()

            return {
                "success": True,
                "pages_processed": len(pages),
                "document_path": str(pdf_path),
                "total_pages_in_index": index_info.get("num_pages", 0),
            }

        except Exception as e:
            logger.error(f"Error loading document {pdf_path}: {e}")
            return {"success": False, "error": str(e), "document_path": str(pdf_path)}

    def clear_index(self):
        """Clear all loaded documents."""
        self.colpali.clear_index()
        logger.info("Document index cleared")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "colpali_status": self.colpali.get_index_info(),
            "input_guardrails_enabled": False,  # Disabled for authenticated users
            "output_guardrails_enabled": self.enable_output_guardrails,
            "max_retrieval_results": self.max_retrieval_results,
            "llm_model": self.llm_model,
        }

        if self.enable_output_guardrails:
            status["output_guardrail_stats"] = self.output_guardrail.get_stats()

        return status

    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health = {"status": "healthy", "components": {}}

        try:
            # Check COLPALI
            index_info = self.colpali.get_index_info()
            health["components"]["colpali"] = {
                "status": "healthy",
                "pages_indexed": index_info.get("num_pages", 0),
            }
        except Exception as e:
            health["components"]["colpali"] = {"status": "error", "error": str(e)}
            health["status"] = "degraded"

        try:
            # Check generation agent
            await self.generation_agent.run("Test generation")
            health["components"]["generation"] = {"status": "healthy"}
        except Exception as e:
            health["components"]["generation"] = {"status": "error", "error": str(e)}
            health["status"] = "degraded"

        return health
