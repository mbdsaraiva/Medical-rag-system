# 🏥 Medical RAG System

An intelligent Medical Question Answering system using Retrieval-Augmented Generation (RAG)

## 🎯 Objective

Develop an intelligent system that answers medical questions with:
- **High precision** through retrieval of reliable documents
- **Transparency** with source citations
- **Explainability** showing documents used in the response

## 💡 Why RAG?

Traditional language models can "hallucinate" information. RAG solves this by:

1. **Retrieval**: Searches relevant documents in a trusted knowledge base
2. **Augmented**: Enhances LLM context with real information
3. **Generation**: LLM generates answers based ONLY on retrieved documents

**Result**: Accurate, verifiable, and updatable answers without retraining the model.

## 📊 Dataset

- **MedQuAD**: Medical Question Answering Dataset
- **Size**: 47,457 question-answer pairs
- **Sources**: NIH, CDC, GARD, NIDDK, FDA, among others
- **Format**: Structured XML

### Dataset Statistics

- Total documents: 47,457
- Unique sources: 13
- Average answer length: ~1,200 characters
- Average question length: ~80 characters
