# Task 12 - Semantic Document Search using ChromaDB and LangChain

## Overview

This project demonstrates semantic document retrieval using LangChain and ChromaDB. User queries are converted into vector embeddings and matched against healthcare documents stored in a ChromaDB vector database.

## Technologies Used

- Python
- LangChain
- ChromaDB
- Sentence Transformers
- Pandas

## Features

- Semantic document search
- Vector similarity retrieval
- Top-5 document retrieval
- Similarity score calculation
- CSV export of search results
- Integration with ChromaDB

## Project Structure

Semantic_Document_Search/

├── dataset/

│   ├── documents/

│   ├── queries.csv

│   ├── search_results.csv

│   └── similarity_scores.csv

├── chroma_db/

├── notebook/

├── semantic_search.py

├── update_database.py

├── README.md

└── requirements.txt

## Installation

```bash
pip install -r requirements.txt
```

## Run

### Update ChromaDB Collection

```bash
python update_database.py
```

### Run Semantic Search

```bash
python semantic_search.py
```

## Generated Files

- search_results.csv
- similarity_scores.csv

## Features Demonstrated

- Semantic Search
- Vector Embeddings
- ChromaDB Retrieval
- Similarity Ranking
- Healthcare Knowledge Base Search

## Future Enhancements

- Retrieval-Augmented Generation (RAG)
- Conversational Search
- Document Chunking
- PDF Knowledge Bases
- Healthcare Question Answering Systems

## Conclusion

This project demonstrates semantic retrieval using LangChain and ChromaDB. The generated vector database and retrieval pipeline provide the foundation for future RAG-based healthcare applications.