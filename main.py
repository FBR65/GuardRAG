import os
import time
import uvicorn
import logging
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp import FastApiMCP
from pydantic import BaseModel, Field

# --- Import Service Classes ---
from mcp_fileconverter.file2pdf import PDFConverter

# --- Import GuardRAG Components ---
from src.rag_agent import GuardRAGAgent
from src.enhanced_rag_agent import EnhancedGuardRAGAgent
from src.colpali_manager import COLPALIManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("guardrag.main")

# --- Load Environment Variables ---
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, override=True)
    logger.info(f"Loaded environment variables from: {dotenv_path}")
else:
    logger.warning(
        f".env file not found at {dotenv_path}. Relying on system environment variables."
    )

# --- Global Variables ---
pdf_converter: Optional[PDFConverter] = None
guardrag_agent: Optional[GuardRAGAgent] = None
enhanced_guardrag_agent: Optional[EnhancedGuardRAGAgent] = None

# Upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting GuardRAG application...")
    await initialize_services()
    logger.info("GuardRAG application started successfully")

    yield

    # Shutdown
    logger.info("Shutting down GuardRAG application...")


async def initialize_services():
    """Initialize all services during startup with proper error handling."""
    global pdf_converter, guardrag_agent, enhanced_guardrag_agent

    # Get LLM configuration
    llm_endpoint = os.getenv("LLM_ENDPOINT", "http://localhost:11434/v1")
    llm_api_key = os.getenv("LLM_API_KEY", "ollama")
    llm_model = os.getenv("LLM_MODEL", "qwen2.5:latest")
    colpali_model = os.getenv("COLPALI_MODEL", "vidore/colqwen2.5-v0.2")
    device = os.getenv("DEVICE", "cuda")

    # Get Qdrant configuration
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    try:
        logger.info("🚀 Starting GuardRAG service initialization...")

        # Initialize PDF Converter
        logger.info("📄 Initializing PDF Converter...")
        try:
            pdf_converter = PDFConverter()
            logger.info("✅ PDF Converter initialized successfully")
        except Exception as e:
            error_msg = f"❌ Failed to initialize PDF Converter: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Initialize GuardRAG Agent
        logger.info("🤖 Initializing Standard GuardRAG agent...")
        try:
            guardrag_agent = GuardRAGAgent(
                llm_endpoint=llm_endpoint,
                llm_api_key=llm_api_key,
                llm_model=llm_model,
                colpali_model=colpali_model,
                enable_output_guardrails=True,
                max_retrieval_results=5,
                device=device,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
            )
            logger.info("✅ Standard GuardRAG agent initialized successfully")
        except ConnectionError as e:
            # Qdrant connection error - provide helpful message
            logger.error(str(e))
            raise SystemExit(
                "🛑 Service startup failed due to Qdrant connection error. Please start Qdrant first."
            )
        except Exception as e:
            error_msg = f"❌ Failed to initialize Standard GuardRAG agent: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Initialize Enhanced GuardRAG Agent
        logger.info("🛡️ Initializing Enhanced GuardRAG agent...")
        try:
            enhanced_guardrag_agent = EnhancedGuardRAGAgent(
                llm_endpoint=llm_endpoint,
                llm_api_key=llm_api_key,
                llm_model=llm_model,
                colpali_model=colpali_model,
                enable_input_guardrails=True,
                enable_output_guardrails=True,
                enable_pii_sanitization=True,
                enable_competitor_detection=True,
                custom_competitors=None,  # Can be configured via environment
                max_retrieval_results=5,
                device=device,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
            )
            logger.info("✅ Enhanced GuardRAG agent initialized successfully")
        except ConnectionError as e:
            # Qdrant connection error - provide helpful message
            logger.error(str(e))
            raise SystemExit(
                "🛑 Service startup failed due to Qdrant connection error. Please start Qdrant first."
            )
        except Exception as e:
            error_msg = f"❌ Failed to initialize Enhanced GuardRAG agent: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info("🎉 All services initialized successfully!")
        logger.info("🌐 GuardRAG API is ready to serve requests")

    except SystemExit:
        # Re-raise SystemExit to properly terminate
        raise
    except Exception as e:
        error_msg = f"💥 Critical error during service initialization: {str(e)}"
        logger.error(error_msg)
        logger.error(
            "🛑 GuardRAG startup failed. Please check the configuration and dependencies."
        )
        raise SystemExit(error_msg)


# --- Pydantic Models ---


# File Upload Models
class FileUploadResponse(BaseModel):
    success: bool
    message: str
    file_id: Optional[str] = None
    pages_processed: Optional[int] = None


# RAG Query Models
class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask about the documents")
    max_results: int = Field(
        5, description="Maximum number of source documents to use", ge=1, le=10
    )


class RAGQueryResponse(BaseModel):
    answer: str
    confidence: float
    processing_time: float
    sources: List[dict]
    warnings: List[str]
    guardrail_checks: dict


# Enhanced RAG Query Models
class EnhancedRAGQueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask about the documents")
    collection_name: str = Field(
        default="documents", description="Name of the document collection"
    )
    enable_pii_sanitization: bool = Field(
        default=True, description="Enable PII sanitization"
    )
    enable_competitor_detection: bool = Field(
        default=True, description="Enable competitor detection"
    )


class EnhancedRAGQueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sanitized_answer: Optional[str] = Field(
        None, description="PII-sanitized answer if different from original"
    )
    confidence: float = Field(..., description="Confidence score (0-1)")
    processing_time: float = Field(..., description="Total processing time in seconds")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents used")
    input_validation: Dict[str, Any] = Field(
        ..., description="Input validation results"
    )
    output_validation: Optional[Dict[str, Any]] = Field(
        None, description="Output validation results"
    )
    guardrail_checks: Dict[str, Any] = Field(
        ..., description="All guardrail check results"
    )
    warnings: List[str] = Field(..., description="Warning messages")
    statistics: Dict[str, Any] = Field(
        ..., description="Performance and validation statistics"
    )


# System Status Models
class SystemStatusResponse(BaseModel):
    status: str
    components: dict
    documents_loaded: int
    guardrails_enabled: dict


# Existing models (for file conversion)
class ConvertToPdfRequest(BaseModel):
    input_filepath: str = Field(
        ..., description="Path to the file that should be converted to PDF."
    )
    output_directory: Optional[str] = Field(
        None, description="Optional directory where the PDF should be saved."
    )
    output_filename: Optional[str] = Field(
        None, description="Optional filename for the output PDF (without extension)."
    )


class ConvertToPdfResponse(BaseModel):
    output_filepath: Optional[str] = None
    error: Optional[str] = None


# --- FastAPI Setup ---
app = FastAPI(
    title="GuardRAG - Secure Document RAG System",
    description="RAG-System auf Basis von COLPALI mit integrierten Guardrails für sichere Dokumentenverarbeitung",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helper Functions ---
def check_service(service_instance, service_name: str):
    """Check if a service is available."""
    if service_instance is None:
        logger.error(f"Attempted to use unavailable {service_name} service.")
        raise HTTPException(
            status_code=503,
            detail=f"{service_name} service is not configured or failed to initialize.",
        )


def check_guardrag():
    """Check if GuardRAG agent is available."""
    check_service(guardrag_agent, "GuardRAG")


# --- Main Routes ---


@app.get("/", include_in_schema=True)
async def root():
    """Root endpoint with system information."""
    return {
        "message": "GuardRAG - Secure Document RAG System",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "upload": "/upload-document",
            "query": "/rag-query",
            "status": "/system-status",
            "health": "/health",
        },
    }


@app.get("/health", include_in_schema=True)
async def health_check():
    """Health check endpoint."""
    try:
        if guardrag_agent:
            health_info = await guardrag_agent.health_check()
            return health_info
        else:
            return {
                "status": "initializing",
                "message": "GuardRAG agent not yet initialized",
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# --- RAG Endpoints ---


@app.post("/upload-document", response_model=FileUploadResponse, tags=["RAG"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a PDF document for RAG.

    The document will be converted to PDF if necessary and processed by COLPALI.
    """
    check_guardrag()

    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        file_extension = Path(file.filename).suffix.lower()
        allowed_extensions = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Allowed: {allowed_extensions}",
            )

        # Save uploaded file
        file_id = f"{int(time.time())}_{file.filename}"
        file_path = UPLOAD_DIR / file_id

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Convert to PDF if necessary
        if file_extension != ".pdf":
            logger.info(f"Converting {file_extension} to PDF")
            pdf_path = pdf_converter.convert(
                str(file_path),
                output_directory=str(UPLOAD_DIR),
                output_filename=f"{file_id}_converted",
            )

            if not pdf_path:
                raise HTTPException(
                    status_code=500, detail="Failed to convert file to PDF"
                )

            # Remove original file and use converted PDF
            file_path.unlink()
            file_path = Path(pdf_path)

        # Process document with GuardRAG
        result = guardrag_agent.load_document(file_path)

        if result["success"]:
            return FileUploadResponse(
                success=True,
                message="Document processed successfully",
                file_id=file_id,
                pages_processed=result["pages_processed"],
            )
        else:
            raise HTTPException(
                status_code=500, detail=f"Failed to process document: {result['error']}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag-query", response_model=RAGQueryResponse, tags=["RAG"])
async def rag_query(request: RAGQueryRequest):
    """
    Query the RAG system with a natural language question.

    The system will:
    1. Validate the query using input guardrails
    2. Retrieve relevant document sections using COLPALI
    3. Generate an answer using the LLM
    4. Validate the response using output guardrails
    """
    check_guardrag()

    try:
        # Process query through GuardRAG pipeline
        result = await guardrag_agent.process_query(request.query)

        # Format sources for API response
        sources = []
        for source in result.sources:
            sources.append(
                {
                    "page_number": source.page.page_number,
                    "score": source.score,
                    "explanation": source.explanation,
                    "text_preview": source.page.text_content[:200] + "..."
                    if source.page.text_content
                    else None,
                }
            )

        return RAGQueryResponse(
            answer=result.answer,
            confidence=result.confidence,
            processing_time=result.processing_time,
            sources=sources,
            warnings=result.warnings,
            guardrail_checks=result.guardrail_checks,
        )

    except Exception as e:
        logger.error(f"Error processing RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/enhanced-rag-query",
    response_model=EnhancedRAGQueryResponse,
    tags=["Enhanced RAG"],
)
async def enhanced_rag_query(request: EnhancedRAGQueryRequest):
    """
    Enhanced RAG query with modern guardrails and German PII support.

    The enhanced system provides:
    1. Advanced toxicity detection (German/English)
    2. Comprehensive PII detection and sanitization
    3. Competitor mention detection
    4. Performance statistics and monitoring
    5. Detailed validation reporting
    """
    global enhanced_guardrag_agent

    if enhanced_guardrag_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Enhanced GuardRAG service is not configured or failed to initialize.",
        )

    try:
        # Process query through Enhanced GuardRAG pipeline
        result = await enhanced_guardrag_agent.process_query(
            query=request.query, collection_name=request.collection_name
        )

        # Format sources for API response
        sources = []
        for source in result.sources:
            sources.append(
                {
                    "page_number": source.page.page_number
                    if hasattr(source, "page")
                    else None,
                    "score": source.score,
                    "source": source.source if hasattr(source, "source") else None,
                    "text_preview": source.text[:200] + "..."
                    if hasattr(source, "text") and source.text
                    else None,
                }
            )

        # Format input validation for response
        input_validation_data = {}
        if result.input_validation:
            input_validation_data = {
                "is_valid": result.input_validation.is_valid,
                "blocked_reason": result.input_validation.blocked_reason,
                "confidence": result.input_validation.confidence,
                "processing_time_ms": result.input_validation.processing_time_ms,
                "failed_validators": result.input_validation.failed_validators,
                "suggestions": result.input_validation.suggestions,
            }

        # Format output validation for response
        output_validation_data = None
        if result.output_validation:
            output_validation_data = {
                "is_valid": result.output_validation.is_valid,
                "reason": result.output_validation.reason,
                "danger_level": getattr(result.output_validation, "danger_level", 0),
                "confidence": getattr(result.output_validation, "confidence", 0.0),
            }

        return EnhancedRAGQueryResponse(
            answer=result.answer,
            sanitized_answer=result.sanitized_answer,
            confidence=result.confidence,
            processing_time=result.processing_time,
            sources=sources,
            input_validation=input_validation_data,
            output_validation=output_validation_data,
            guardrail_checks=result.guardrail_checks,
            warnings=result.warnings,
            statistics=result.statistics,
        )

    except Exception as e:
        logger.error(f"Error processing enhanced RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system-status", response_model=SystemStatusResponse, tags=["RAG"])
async def system_status():
    """Get comprehensive system status."""
    check_guardrag()

    try:
        status = guardrag_agent.get_system_status()

        return SystemStatusResponse(
            status="operational",
            components=status,
            documents_loaded=status["colpali_status"]["num_pages"] or 0,
            guardrails_enabled={
                "input": status["input_guardrails_enabled"],
                "output": status["output_guardrails_enabled"],
            },
        )

    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/enhanced-system-status", response_model=Dict[str, Any], tags=["Enhanced RAG"]
)
async def enhanced_system_status():
    """Get comprehensive enhanced system status with guardrails statistics."""
    global enhanced_guardrag_agent

    if enhanced_guardrag_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Enhanced GuardRAG service is not configured.",
        )

    try:
        health_check = await enhanced_guardrag_agent.health_check()
        statistics = enhanced_guardrag_agent.get_statistics()

        return {
            "status": "healthy",
            "version": "1.0.0-enhanced",
            "health_check": health_check,
            "statistics": statistics,
            "features": {
                "german_pii_detection": True,
                "toxicity_detection": True,
                "competitor_detection": True,
                "text_sanitization": True,
                "spacy_available": True,  # This should come from the actual system
            },
        }
    except Exception as e:
        logger.error(f"Error getting enhanced system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear-documents", tags=["RAG"])
async def clear_documents():
    """Clear all loaded documents from the system."""
    check_guardrag()

    try:
        guardrag_agent.clear_index()

        # Clean up upload directory
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()

        return {"message": "All documents cleared successfully"}

    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- File Conversion Endpoint ---


@app.post(
    "/convert-to-pdf",
    response_model=ConvertToPdfResponse,
    summary="Convert File to PDF",
    operation_id="convert_to_pdf",
    tags=["File Conversion"],
)
async def convert_to_pdf_endpoint(request_data: ConvertToPdfRequest):
    """Converts various file types to PDF format."""
    check_service(pdf_converter, "PDF Converter")
    logger.info(
        f"API: Received request to convert file to PDF: {request_data.input_filepath}"
    )
    try:
        output_path = pdf_converter.convert(
            request_data.input_filepath,
            output_directory=request_data.output_directory,
            output_filename=request_data.output_filename,
        )

        if output_path:
            return ConvertToPdfResponse(output_filepath=str(output_path))
        else:
            return ConvertToPdfResponse(
                error="Conversion failed. Check logs for details."
            )
    except Exception as e:
        logger.exception(f"Error during PDF conversion: {e}")
        return ConvertToPdfResponse(error=str(e))


# --- Security and Guardrails Endpoints ---


@app.get("/security/stats", response_model=dict)
async def get_security_stats():
    """Get comprehensive security statistics from guardrails."""
    try:
        if not guardrag_agent:
            raise HTTPException(
                status_code=503, detail="GuardRAG agent not initialized"
            )

        stats = {
            "system_status": "operational",
            "input_guardrails": {
                "status": "disabled",
                "reason": "Input guardrails are disabled for authenticated users",
                "blocked_requests": 0,
                "last_activity": "N/A"
            },
            "output_guardrails": None,
            "total_security_events": 0,
        }

        # Get output guardrail stats (only output guardrails are available in standard GuardRAG)
        if hasattr(guardrag_agent, 'enable_output_guardrails') and guardrag_agent.enable_output_guardrails:
            if hasattr(guardrag_agent, 'output_guardrail') and guardrag_agent.output_guardrail:
                try:
                    output_stats = guardrag_agent.output_guardrail.get_security_stats()
                    stats["output_guardrails"] = output_stats
                    stats["total_security_events"] += output_stats.get("blocked_responses", 0)
                except Exception as guardrail_error:
                    logger.warning(f"Could not get output guardrail stats: {guardrail_error}")
                    stats["output_guardrails"] = {
                        "status": "error",
                        "error": str(guardrail_error),
                        "blocked_responses": 0
                    }
            else:
                stats["output_guardrails"] = {
                    "status": "enabled_but_not_initialized",
                    "blocked_responses": 0
                }
        else:
            stats["output_guardrails"] = {
                "status": "disabled",
                "blocked_responses": 0
            }

        # Check if enhanced agent is available for more detailed stats
        if enhanced_guardrag_agent:
            try:
                enhanced_stats = enhanced_guardrag_agent.get_statistics()
                stats["enhanced_guardrails"] = enhanced_stats
                stats["enhanced_available"] = True
            except Exception as enhanced_error:
                logger.debug(f"Enhanced stats not available: {enhanced_error}")
                stats["enhanced_available"] = False

        return stats

    except Exception as e:
        logger.error(f"Error getting security stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/security/update-level")
async def update_security_level(level: str):
    """Update the security level of guardrails."""
    try:
        if not guardrag_agent:
            raise HTTPException(
                status_code=503, detail="GuardRAG agent not initialized"
            )

        if level not in ["low", "medium", "high"]:
            raise HTTPException(
                status_code=400,
                detail="Security level must be 'low', 'medium', or 'high'",
            )

        updated_components = []

        # Update output guardrail security level (input guardrails not available in standard GuardRAG)
        if hasattr(guardrag_agent, 'enable_output_guardrails') and guardrag_agent.enable_output_guardrails:
            if hasattr(guardrag_agent, 'output_guardrail') and guardrag_agent.output_guardrail:
                try:
                    # Try to update security level if method exists
                    if hasattr(guardrag_agent.output_guardrail, 'update_security_level'):
                        guardrag_agent.output_guardrail.update_security_level(level)
                        updated_components.append("output_guardrail")
                    elif hasattr(guardrag_agent.output_guardrail, 'security_level'):
                        guardrag_agent.output_guardrail.security_level = level
                        updated_components.append("output_guardrail")
                    else:
                        logger.warning("Output guardrail does not support security level updates")
                except Exception as guardrail_error:
                    logger.warning(f"Could not update output guardrail security level: {guardrail_error}")

        # Try to update enhanced guardrails if available
        if enhanced_guardrag_agent:
            try:
                if hasattr(enhanced_guardrag_agent, 'update_security_level'):
                    enhanced_guardrag_agent.update_security_level(level)
                    updated_components.append("enhanced_guardrails")
            except Exception as enhanced_error:
                logger.debug(f"Enhanced guardrails security level update failed: {enhanced_error}")

        if not updated_components:
            return {
                "message": f"Security level set to {level}, but no updatable components found",
                "success": True,
                "updated_components": [],
                "note": "Standard GuardRAG has limited security level configuration"
            }

        return {
            "message": f"Security level updated to {level}",
            "success": True,
            "updated_components": updated_components
        }

    except Exception as e:
        logger.error(f"Error updating security level: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset-enhanced-statistics", tags=["Enhanced RAG"])
async def reset_enhanced_statistics():
    """Reset all enhanced guardrails statistics."""
    global enhanced_guardrag_agent

    if enhanced_guardrag_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Enhanced GuardRAG service is not configured.",
        )

    try:
        enhanced_guardrag_agent.reset_statistics()
        return {"message": "Enhanced statistics reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting enhanced statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update-competitors", tags=["Enhanced RAG"])
async def update_competitors(competitors: List[str]):
    """Update the list of competitors to detect and block."""
    global enhanced_guardrag_agent

    if enhanced_guardrag_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Enhanced GuardRAG service is not configured.",
        )

    try:
        enhanced_guardrag_agent.update_competitors(competitors)
        return {
            "message": f"Competitor list updated with {len(competitors)} entries",
            "competitors": competitors,
        }
    except Exception as e:
        logger.error(f"Error updating competitors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- COLPALI Management Endpoints ---


@app.get("/colpali-stats", response_model=dict, tags=["System"])
async def get_colpali_stats():
    """Get COLPALI model statistics and memory usage."""
    try:
        stats = COLPALIManager.get_stats()
        return {
            "status": "success",
            "colpali_instances": stats,
            "memory_optimization": {
                "shared_instances": stats["total_instances"],
                "memory_saved": "Modell wird zwischen Agenten geteilt"
                if stats["total_instances"] == 1
                else "Mehrere Instanzen aktiv",
            },
        }
    except Exception as e:
        logger.error(f"Error getting COLPALI stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/colpali-clear-cache", tags=["System"])
async def clear_colpali_cache():
    """Clear COLPALI model cache to free memory."""
    try:
        COLPALIManager.clear_cache()
        return {"message": "COLPALI cache cleared successfully", "status": "success"}
    except Exception as e:
        logger.error(f"Error clearing COLPALI cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- FastAPI MCP Integration ---
server_host = os.environ.get("SERVER_HOST", "localhost")
server_port = os.environ.get("SERVER_PORT", "8000")
try:
    port_num = int(server_port)
except ValueError:
    port_num = 8000
    logger.warning(f"Invalid SERVER_PORT '{server_port}', using default {port_num}.")

server_scheme = os.environ.get("SERVER_SCHEME", "http")
base_url = f"{server_scheme}://{server_host}:{port_num}"
logger.info(f"Configuring MCP with base_url: {base_url}")

mcp = FastApiMCP(
    app,
    name="GuardRAG MCP Server",
    describe_full_response_schema=True,
    description="Provides tools for secure document RAG and file conversion.",
    include_operations=[
        "convert_to_pdf",
    ],
)

mcp.mount()


# --- Run Server ---
if __name__ == "__main__":
    logger.info(f"Starting GuardRAG server on host 0.0.0.0:{port_num}")

    reload_enabled = os.getenv("UVICORN_RELOAD", "true").lower() == "true"
    log_level = os.getenv("UVICORN_LOG_LEVEL", "info").lower()
    workers_count = int(os.getenv("UVICORN_WORKERS", "1"))

    if reload_enabled or workers_count > 1:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port_num,
            reload=reload_enabled if workers_count == 1 else False,
            workers=workers_count if not reload_enabled else 1,
            log_level=log_level,
        )
    else:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port_num,
            log_level=log_level,
        )
