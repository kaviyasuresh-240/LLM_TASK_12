import os

from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="healthcare_documents"
)

folder = "dataset/documents"

documents = []
ids = []
metadata = []

for index in range(11, 21):

    filename = f"document{index}.txt"

    filepath = os.path.join(
        folder,
        filename
    )

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    documents.append(text)

    ids.append(f"new_{index}")

    metadata.append(
        {
            "title": filename,
            "category": "Healthcare",
            "source": filename
        }
    )

embeddings = model.encode(
    documents
)

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadata
)

print("New documents added successfully.")