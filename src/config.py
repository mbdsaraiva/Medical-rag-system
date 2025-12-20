"""
centralized configuration for the medical rag system project
"""
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CHROMA_DIR = ROOT_DIR / "chroma_db"

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR,EMBEDDINGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# dataset
MEDQUAD_URL = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"
MEDQUAD_ZIP = RAW_DATA_DIR / "medquad.zip"
MEDQUAD_EXTRACTED = RAW_DATA_DIR / "MedQuAD-master"

PROCESSED_CSV = PROCESSED_DATA_DIR / "medquad_clean.csv"
METADATA_JSON = PROCESSED_DATA_DIR / "documents_metadata.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K_RETRIEVAL = 3        
CHUNK_SIZE = 500       
CHUNK_OVERLAP = 50  

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
LLM_TEMPERATURE = 0.3     
LLM_MAX_TOKENS = 500

LOG_LEVEL = "INFO"

print("Configuration loaded")
print(f"Root directory: {ROOT_DIR}")
print(f"Dataset will be saved in: {RAW_DATA_DIR}")