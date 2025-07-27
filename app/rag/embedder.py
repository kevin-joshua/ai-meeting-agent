import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import re
import hashlib

class AudioTranscriptionIngestor:
    def __init__(self, 
                 collection_name: str = "audio_transcriptions",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the ChromaDB ingestion system for audio transcriptions
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            embedding_model: HuggingFace model name for embeddings
        """
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Audio transcription embeddings"}
        )
    
    def preprocess_transcription(self, transcription: str) -> str:
        """
        Clean and preprocess the transcription text
        
        Args:
            transcription: Raw transcription text
            
        Returns:
            Cleaned transcription text
        """
        if not transcription or not isinstance(transcription, str):
            return ""
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', transcription.strip())
        
        # Remove common transcription artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
        text = re.sub(r'\(.*?\)', '', text)  # Remove parenthetical content
        
        # Clean up punctuation
        text = re.sub(r'[^\w\s.,!?;:\-]', '', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for better embedding representation
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                sentence_end = text.rfind('.', start, end)
                if sentence_end == -1:
                    sentence_end = text.rfind('!', start, end)
                if sentence_end == -1:
                    sentence_end = text.rfind('?', start, end)
                
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
        return chunks
    
    def generate_metadata(self, 
                         transcription_data: Dict[str, Any], 
                         chunk_index: int = 0, 
                         total_chunks: int = 1) -> Dict[str, Any]:
        """
        Generate metadata for the transcription chunk
        
        Args:
            transcription_data: Original transcription data with metadata
            chunk_index: Index of current chunk (0-based)
            total_chunks: Total number of chunks for this transcription
            
        Returns:
            Metadata dictionary
        """
        # Extract common metadata fields
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "embedding_model": self.embedding_model_name,
            "content_type": "audio_transcription"
        }
        
        # Add transcription-specific metadata
        if isinstance(transcription_data, dict):
            # Audio file information
            if "audio_file" in transcription_data:
                metadata["audio_file"] = transcription_data["audio_file"]
            if "audio_duration" in transcription_data:
                metadata["audio_duration"] = transcription_data["audio_duration"]
            if "audio_format" in transcription_data:
                metadata["audio_format"] = transcription_data["audio_format"]
                
            # Transcription information
            if "transcription_model" in transcription_data:
                metadata["transcription_model"] = transcription_data["transcription_model"]
            if "confidence_score" in transcription_data:
                metadata["confidence_score"] = transcription_data["confidence_score"]
            if "language" in transcription_data:
                metadata["language"] = transcription_data["language"]
                
            # Speaker information
            if "speaker_id" in transcription_data:
                metadata["speaker_id"] = transcription_data["speaker_id"]
            if "num_speakers" in transcription_data:
                metadata["num_speakers"] = transcription_data["num_speakers"]
                
            # Content analysis
            if "word_count" in transcription_data:
                metadata["word_count"] = transcription_data["word_count"]
            if "sentiment" in transcription_data:
                metadata["sentiment"] = transcription_data["sentiment"]
                
            # Custom metadata
            if "custom_metadata" in transcription_data:
                metadata.update(transcription_data["custom_metadata"])
        
        return metadata
    
    def generate_content_hash(self, content: str) -> str:
        """Generate a hash for content deduplication"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def ingest_transcription(self, transcription_data: Any) -> Dict[str, Any]:
        """
        Main ingestion function to process and store transcription data
        
        Args:
            transcription_data: Can be a string (just text) or dict with text and metadata
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            # Handle different input formats
            if isinstance(transcription_data, str):
                text = transcription_data
                metadata_dict = {}
            elif isinstance(transcription_data, dict):
                # Extract text from various possible keys
                text = (transcription_data.get("text") or 
                       transcription_data.get("transcription") or 
                       transcription_data.get("content") or "")
                metadata_dict = transcription_data
            else:
                raise ValueError("Transcription data must be string or dictionary")
            
            if not text:
                raise ValueError("No text content found in transcription data")
            
            # Preprocess the transcription
            cleaned_text = self.preprocess_transcription(text)
            if not cleaned_text:
                raise ValueError("No valid content after preprocessing")
            
            # Chunk the text
            chunks = self.chunk_text(cleaned_text)
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                # Generate unique ID
                content_hash = self.generate_content_hash(chunk)
                doc_id = f"transcription_{content_hash}_{i}"
                
                # Generate metadata
                metadata = self.generate_metadata(metadata_dict, i, len(chunks))
                metadata["chunk_text_length"] = len(chunk)
                metadata["chunk_word_count"] = len(chunk.split())
                
                documents.append(chunk)
                metadatas.append(metadata)
                ids.append(doc_id)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Store in ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )
            
            result = {
                "status": "success",
                "chunks_processed": len(chunks),
                "total_characters": len(cleaned_text),
                "collection_name": self.collection_name,
                "document_ids": ids,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"Successfully ingested {len(chunks)} chunks into ChromaDB")
            return result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }
            print(f"Error during ingestion: {e}")
            return error_result
    
    def search_similar(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar transcriptions
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Search results
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return {
                "status": "success",
                "results": results,
                "query": query
            }
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "query": query
            }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model_name
            }
        except Exception as e:
            return {
                "error": str(e)
            }

# Example usage function
# def example_usage():
#     """Example of how to use the AudioTranscriptionIngestor"""
    
#     # Initialize the ingestor
#     ingestor = AudioTranscriptionIngestor(
#         collection_name="my_audio_transcriptions",
#         persist_directory="./my_chroma_db"
#     )
    
#     # Example transcription data (this would come from your other function)
#     sample_transcription = {
#         "text": "Hello, this is a sample audio transcription. The speaker discusses various topics including technology, artificial intelligence, and machine learning. The conversation lasted approximately 5 minutes.",
#         "audio_file": "sample_audio.wav",
#         "audio_duration": 300,  # seconds
#         "language": "en",
#         "confidence_score": 0.95,
#         "transcription_model": "whisper-large",
#         "speaker_id": "speaker_001",
#         "custom_metadata": {
#             "session_id": "session_123",
#             "topic": "AI Discussion"
#         }
#     }
    
#     # Ingest the transcription
#     result = ingestor.ingest_transcription(sample_transcription)
#     print("Ingestion result:", result)
    
#     # Search for similar content
#     search_result = ingestor.search_similar("artificial intelligence", n_results=3)
#     print("Search result:", search_result)
    
#     # Get collection statistics
#     stats = ingestor.get_collection_stats()
#     print("Collection stats:", stats)

