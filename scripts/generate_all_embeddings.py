import sys
sys.path.append('..')

from src.data_loader import load_processed_data
from src.embeddings import generate_document_embeddings


def main():
    """
    main function to generate and save all embeddings
    """
    print("="*80)
    print(" Generating embeddings for entire medquad dataset")
    print("="*80)
    
    # load data
    print("\n1️Loading dataset...")
    df = load_processed_data()
    
    # generate embeddings
    print("\n2️Generating embeddings...")
    print("This will take approximately 10 minutes...")
    
    embeddings = generate_document_embeddings(df, text_column='answer')
    
    print("\n" + "="*80)
    print(" Embeddings generated successfully")
    print("="*80)
    print(f"\nTotal embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Total size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
    print(f"\nSaved to: data/embeddings/document_embeddings.pkl")
    

if __name__ == "__main__":
    main()