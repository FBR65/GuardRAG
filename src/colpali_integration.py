"""
COLPALI Integration Module
Handles document processing and retrieval using COLPALI vision-language models with Qdrant vector storage.
"""

import torch
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from PIL import Image
import pdf2image
import fitz  # PyMuPDF
from dataclasses import dataclass
from tqdm import tqdm
import hashlib
from huggingface_hub import snapshot_download

from colpali_engine.models import ColQwen2_5, ColQwen2Processor

from .qdrant_integration import QdrantVectorStore, QdrantConfig

logger = logging.getLogger(__name__)


@dataclass
class DocumentPage:
    """Represents a single page from a document."""

    page_number: int
    image: Image.Image
    text_content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalResult:
    """Represents a retrieval result with similarity scores."""

    page: DocumentPage
    score: float
    query: str
    explanation: Optional[str] = None


class COLPALIProcessor:
    """
    Processes documents using COLPALI for efficient visual document retrieval with Qdrant storage.
    """

    def __init__(
        self,
        model_name: str = "vidore/colqwen2.5-v0.2",
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        qdrant_config: Optional[QdrantConfig] = None,
    ):
        """
        Initialize COLPALI processor.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on
            torch_dtype: Torch data type for model weights
            qdrant_config: Qdrant configuration
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype

        logger.info(f"Initializing COLPALI with model: {model_name}")
        self._load_model()

        # Initialize Qdrant vector store
        self.qdrant_config = qdrant_config or QdrantConfig()
        self.vector_store = QdrantVectorStore(self.qdrant_config)

        # Document tracking
        self.document_ids: Dict[str, str] = {}  # filepath -> document_id

    def _check_model_cache(self) -> bool:
        """Check if model is fully cached locally."""
        try:
            # Try to determine cache directory
            cache_dir = os.getenv("HF_HOME") or os.path.expanduser(
                "~/.cache/huggingface"
            )
            model_cache_dir = (
                Path(cache_dir)
                / "hub"
                / f"models--{self.model_name.replace('/', '--')}"
            )

            if not model_cache_dir.exists():
                logger.info(f"Model cache directory not found: {model_cache_dir}")
                return False

            # Check for essential model files
            snapshots_dir = model_cache_dir / "snapshots"
            if not snapshots_dir.exists():
                logger.info("No snapshots directory found in cache")
                return False

            # Find latest snapshot
            snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
            if not snapshot_dirs:
                logger.info("No snapshot directories found")
                return False

            latest_snapshot = max(snapshot_dirs, key=lambda x: x.stat().st_mtime)

            # Check for required files
            required_files = [
                "config.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model.safetensors.index.json",
            ]

            missing_files = []
            for file_name in required_files:
                file_path = latest_snapshot / file_name
                if not file_path.exists():
                    missing_files.append(file_name)
                elif (
                    file_name.endswith(".safetensors")
                    and file_path.stat().st_size < 1000
                ):  # Too small
                    missing_files.append(
                        f"{file_name} (too small: {file_path.stat().st_size} bytes)"
                    )

            if missing_files:
                logger.warning(
                    f"Model cache incomplete. Missing files: {missing_files}"
                )
                return False

            logger.info(f"Model fully cached at: {latest_snapshot}")
            return True

        except Exception as e:
            logger.warning(f"Error checking model cache: {e}")
            return False

    def _ensure_model_cached(self):
        """Ensure model is fully downloaded and cached."""
        if self._check_model_cache():
            logger.info("Model already fully cached")
            return

        logger.info(f"Model not fully cached. Downloading: {self.model_name}")
        try:
            # Download model using snapshot_download to ensure complete download
            snapshot_download(
                repo_id=self.model_name,
                cache_dir=None,  # Use default cache
                resume_download=True,
                local_files_only=False,
            )
            logger.info("Model download completed successfully")

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise RuntimeError(
                f"Could not download COLPALI model {self.model_name}: {e}"
            )

    def _load_model(self):
        """Load COLPALI model and processor."""
        try:
            logger.info(f"Loading COLPALI model: {self.model_name}")

            # Ensure model is fully cached first
            self._ensure_model_cached()

            # Load model from cache only
            self.model = ColQwen2_5.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                local_files_only=True,  # Only use cached files
            ).eval()

            self.processor = ColQwen2Processor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=True,  # Only use cached files
            )

            # Move to CPU after loading
            self.model = self.model.to("cpu")
            self.device = "cpu"

            logger.info(f"Successfully loaded COLPALI model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load COLPALI model: {e}")
            raise

    def pdf_to_pages(self, pdf_path: Path, dpi: int = 200) -> List[DocumentPage]:
        """
        Convert PDF to list of DocumentPage objects.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for image conversion

        Returns:
            List of DocumentPage objects
        """
        pages = []

        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(str(pdf_path), dpi=dpi)

            # Extract text using PyMuPDF
            doc = fitz.open(str(pdf_path))

            for i, (image, page) in enumerate(zip(images, doc)):
                text_content = page.get_text()

                page_obj = DocumentPage(
                    page_number=i + 1,
                    image=image,
                    text_content=text_content,
                    metadata={
                        "source_file": str(pdf_path),
                        "dpi": dpi,
                        "image_size": image.size,
                    },
                )
                pages.append(page_obj)

            doc.close()
            logger.info(f"Converted PDF {pdf_path} to {len(pages)} pages")

        except Exception as e:
            logger.error(f"Error converting PDF {pdf_path}: {e}")
            raise

        return pages

    def process_document(self, pdf_path: Path) -> List[DocumentPage]:
        """
        Process a PDF document and generate embeddings.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of processed DocumentPage objects
        """
        # Create document ID from file path hash
        document_id = hashlib.md5(str(pdf_path).encode()).hexdigest()

        # Convert PDF to pages
        pages = self.pdf_to_pages(pdf_path)

        # Generate embeddings for all pages
        images = [page.image for page in pages]
        embeddings = self._generate_embeddings(images)

        # Convert embeddings to numpy arrays
        numpy_embeddings = []
        for embedding in embeddings:
            if isinstance(embedding, torch.Tensor):
                numpy_embeddings.append(embedding.cpu().numpy())
            else:
                numpy_embeddings.append(embedding)

        # Store in Qdrant
        point_ids = self.vector_store.add_embeddings(
            embeddings=numpy_embeddings,
            pages=pages,
            document_id=document_id,
        )

        # Track document
        self.document_ids[str(pdf_path)] = document_id

        logger.info(
            f"Processed document {pdf_path} with {len(pages)} pages, stored {len(point_ids)} embeddings"
        )
        return pages

    def _generate_embeddings(
        self, images: List[Image.Image], batch_size: int = 4
    ) -> List[torch.Tensor]:
        """
        Generate COLPALI embeddings for a list of images.

        Args:
            images: List of PIL images
            batch_size: Batch size for processing

        Returns:
            List of embedding tensors
        """
        embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="Generating embeddings"):
            batch_images = images[i : i + batch_size]

            # Process images
            batch_inputs = self.processor.process_images(batch_images).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                batch_embeddings = self.model(**batch_inputs)
                embeddings.extend(list(torch.unbind(batch_embeddings.to("cpu"))))

        return embeddings

    def search(
        self, query: str, top_k: int = 5, return_explanations: bool = True
    ) -> List[RetrievalResult]:
        """
        Search for relevant pages using a text query.

        Args:
            query: Search query
            top_k: Number of top results to return
            return_explanations: Whether to include explanations

        Returns:
            List of RetrievalResult objects
        """
        # Generate query embedding
        query_inputs = self.processor.process_queries([query]).to(self.device)

        with torch.no_grad():
            query_embedding = self.model(**query_inputs)

        # Convert to numpy array
        if isinstance(query_embedding, torch.Tensor):
            query_np = query_embedding.cpu().numpy()
        else:
            query_np = query_embedding

        # Search in Qdrant
        search_results = self.vector_store.search(
            query_embedding=query_np,
            top_k=top_k,
        )

        # Convert results to RetrievalResult objects
        results = []
        for hit in search_results:
            payload = hit["payload"]

            # Reconstruct DocumentPage
            page = DocumentPage(
                page_number=payload["page_number"],
                image=None,  # Image not stored in Qdrant
                text_content=payload["text_content"],
                metadata=payload.get("metadata", {}),
            )

            result = RetrievalResult(
                page=page,
                score=hit["score"],
                query=query,
                explanation=self._generate_explanation(query, page)
                if return_explanations
                else None,
            )
            results.append(result)

        logger.info(f"Search for '{query}' returned {len(results)} results")
        return results

    def _generate_explanation(self, query: str, page: DocumentPage) -> str:
        """
        Generate explanation for why a page was retrieved.

        Args:
            query: Original search query
            page: Retrieved page

        Returns:
            Explanation string
        """
        # Simple explanation based on text content match
        if page.text_content:
            query_words = set(query.lower().split())
            page_words = set(page.text_content.lower().split())
            common_words = query_words.intersection(page_words)

            if common_words:
                return f"Seite {page.page_number} enthält übereinstimmende Begriffe: {', '.join(common_words)}"

        return f"Seite {page.page_number} wurde basierend auf visueller Ähnlichkeit ausgewählt"

    def clear_index(self):
        """Clear all stored embeddings and pages."""
        self.vector_store.clear_collection()
        self.document_ids.clear()
        logger.info("Document index cleared")

    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the current index."""
        collection_info = self.vector_store.get_collection_info()
        return {
            "num_pages": collection_info.get("vectors_count", 0),
            "model_name": self.model_name,
            "device": self.device,
            "num_documents": len(self.document_ids),
            "collection_info": collection_info,
        }
