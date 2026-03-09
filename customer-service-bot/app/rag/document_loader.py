"""Document loading and chunking for RAG."""

from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_config, DOCUMENTS_DIR
from app.utils.logging import get_logger

logger = get_logger("document_loader")


class DocumentLoader:
    """
    Load and chunk documents from various file formats.
    """
    
    # Supported file extensions and their loaders
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
    
    def __init__(
        self,
        documents_dir: Optional[Path] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the document loader.
        
        Args:
            documents_dir: Directory containing documents to load
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.documents_dir = documents_dir or DOCUMENTS_DIR
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    
    def _load_text_file(self, filepath: Path) -> str:
        """Load a plain text or markdown file."""
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    
    def _load_pdf_file(self, filepath: Path) -> str:
        """Load a PDF file."""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            
            loader = PyPDFLoader(str(filepath))
            pages = loader.load()
            return "\n\n".join(page.page_content for page in pages)
        except ImportError:
            logger.warning("PyPDF not installed. Install with: pip install pypdf")
            return ""
    
    def _load_docx_file(self, filepath: Path) -> str:
        """Load a Word document."""
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            
            loader = Docx2txtLoader(str(filepath))
            docs = loader.load()
            return "\n\n".join(doc.page_content for doc in docs)
        except ImportError:
            logger.warning("docx2txt not installed. Install with: pip install docx2txt")
            return ""
    
    def load_file(self, filepath: Path) -> list[Document]:
        """
        Load and chunk a single file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            List of Document chunks
        """
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return []
        
        extension = filepath.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported file type: {extension}")
            return []
        
        # Load file content based on type
        if extension in {".txt", ".md"}:
            content = self._load_text_file(filepath)
        elif extension == ".pdf":
            content = self._load_pdf_file(filepath)
        elif extension == ".docx":
            content = self._load_docx_file(filepath)
        else:
            return []
        
        if not content.strip():
            return []
        
        # Create document with metadata
        doc = Document(
            page_content=content,
            metadata={
                "source": str(filepath),
                "filename": filepath.name,
                "file_type": extension,
            },
        )
        
        # Split into chunks
        chunks = self.text_splitter.split_documents([doc])
        logger.info(f"Loaded {filepath.name}: {len(chunks)} chunks")
        
        return chunks
    
    def load_directory(self) -> list[Document]:
        """
        Load all documents from the documents directory.
        
        Returns:
            List of all Document chunks
        """
        if not self.documents_dir.exists():
            logger.warning(f"Documents directory not found: {self.documents_dir}")
            return []
        
        all_chunks = []
        
        for filepath in self.documents_dir.iterdir():
            if filepath.is_file() and filepath.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                chunks = self.load_file(filepath)
                all_chunks.extend(chunks)
        
        logger.info(f"Loaded {len(all_chunks)} total chunks from {self.documents_dir}")
        return all_chunks
    
    def load_text(self, text: str, source: str = "manual") -> list[Document]:
        """
        Load and chunk raw text.
        
        Args:
            text: Text content to chunk
            source: Source identifier for metadata
            
        Returns:
            List of Document chunks
        """
        if not text.strip():
            return []
        
        doc = Document(
            page_content=text,
            metadata={"source": source},
        )
        
        return self.text_splitter.split_documents([doc])
