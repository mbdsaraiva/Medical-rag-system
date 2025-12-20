import requests
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

from src.config import (
    MEDQUAD_URL, 
    MEDQUAD_ZIP, 
    MEDQUAD_EXTRACTED,
    RAW_DATA_DIR,
    PROCESSED_CSV
)


def download_medquad() -> None:
    """
    Downloads the MedQuAD dataset from GitHub
    """
    if MEDQUAD_ZIP.exists():
        print(f"Dataset already downloaded at {MEDQUAD_ZIP}")
        return
    
    print(f"Downloading MedQuAD from {MEDQUAD_URL}...")
    
    try:
        response = requests.get(MEDQUAD_URL, stream=True)
        response.raise_for_status()
        
        # Download with progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        with open(MEDQUAD_ZIP, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"Download completed: {MEDQUAD_ZIP}")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise


def extract_medquad() -> None:
    """
    Extracts the MedQuAD ZIP file
    """
    if MEDQUAD_EXTRACTED.exists():
        print(f"Dataset already extracted at {MEDQUAD_EXTRACTED}")
        return
    
    print(f"Extracting {MEDQUAD_ZIP}...")
    
    try:
        with zipfile.ZipFile(MEDQUAD_ZIP, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        
        print(f"Extraction completed: {MEDQUAD_EXTRACTED}")
        
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        raise


def parse_medquad_xml(xml_path: Path) -> List[Dict[str, str]]:
    """
    Parses a MedQuAD XML file
    
    Args:
        xml_path: Path to the XML file
        
    Returns:
        List of dictionaries containing question–answer pairs
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract document metadata
        doc_source = root.get('source', 'Unknown')
        doc_url = root.get('url', '')
        
        qa_pairs = []
        
        # Find all QA pairs
        for qa in root.findall('.//QAPair'):
            question_elem = qa.find('Question')
            answer_elem = qa.find('Answer')
            
            if question_elem is not None and answer_elem is not None:
                question = question_elem.text
                answer = answer_elem.text
                
                # Only add if both exist and are not empty
                if question and answer:
                    qa_pairs.append({
                        'question': question.strip(),
                        'answer': answer.strip(),
                        'source': doc_source,
                        'url': doc_url,
                        'doc_id': qa.get('pid', '')
                    })
        
        return qa_pairs
        
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return []


def process_all_xmls() -> pd.DataFrame:
    """
    Processes all MedQuAD XML files
    
    Returns:
        DataFrame containing all processed QA pairs
    """
    print(f"🔍 Searching for XML files in {MEDQUAD_EXTRACTED}...")
    
    # Find all XML files
    xml_files = list(MEDQUAD_EXTRACTED.rglob("*.xml"))
    print(f"Found {len(xml_files)} XML files")
    
    all_qa_pairs = []
    
    # Process each XML with a progress bar
    print("Processing files...")
    for xml_file in tqdm(xml_files, desc="Processing XMLs"):
        qa_pairs = parse_medquad_xml(xml_file)
        all_qa_pairs.extend(qa_pairs)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_qa_pairs)
    
    # Add unique ID
    df['id'] = df.index
    
    # Statistics
    print(f"\nDataset Statistics:")
    print(f"   • Total QA pairs: {len(df)}")
    print(f"   • Unique sources: {df['source'].nunique()}")
    print(f"   • Distribution by source:")
    print(df['source'].value_counts().head(10))
    
    return df


def save_processed_data(df: pd.DataFrame) -> None:
    """
    Saves the processed dataset to CSV
    
    Args:
        df: DataFrame containing the processed data
    """
    print(f"\nSaving processed data to {PROCESSED_CSV}...")
    df.to_csv(PROCESSED_CSV, index=False)
    print(f"Data saved successfully!")
    print(f"File size: {PROCESSED_CSV.stat().st_size / 1024 / 1024:.2f} MB")


def load_processed_data() -> pd.DataFrame:
    """
    Loads the processed dataset
    
    Returns:
        DataFrame containing the data
    """
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(
            f"File {PROCESSED_CSV} not found. "
            "Run setup_dataset() first."
        )
    
    print(f"Loading data from {PROCESSED_CSV}...")
    df = pd.read_csv(PROCESSED_CSV)
    print(f"Loaded {len(df)} records")
    return df


def setup_dataset() -> pd.DataFrame:
    """
    Full pipeline: download → extract → process → save
    
    Returns:
        DataFrame containing the processed data
    """
    print("Starting MedQuAD dataset setup\n")
    
    download_medquad()
    extract_medquad()
    df = process_all_xmls()
    save_processed_data(df)
    print("\nDataset setup completed successfully!")
    return df


if __name__ == "__main__":
    df = setup_dataset()
    
    print("\nSample data:")
    print(df.head(3))
