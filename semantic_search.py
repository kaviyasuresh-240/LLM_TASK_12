import pandas as pd

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# =====================================================
# Load Embedding Model
# =====================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# =====================================================
# Load Existing ChromaDB Collection
# =====================================================

print("Loading ChromaDB...")

db = Chroma(
    collection_name="healthcare_documents",
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# =====================================================
# Check Database
# =====================================================

print("\nChecking Database...")

try:
    count = db._collection.count()
    print("Documents in Collection:", count)
except Exception as e:
    print("Error:", e)

# =====================================================
# Load Queries
# =====================================================

queries_df = pd.read_csv(
    "dataset/queries.csv"
)

search_results = []
similarity_scores = []

print("\nStarting Semantic Search...\n")

# =====================================================
# Search
# =====================================================

for query in queries_df["Query"]:

    print("\n" + "=" * 60)
    print("QUERY:", query)
    print("=" * 60)

    results = db.similarity_search_with_score(
        query=query,
        k=5
    )

    if len(results) == 0:
        print("No documents retrieved.")
        continue

    rank = 1

    for doc, score in results:

        print(f"\nRank {rank}")
        print("Similarity Score:", round(score, 4))
        print(doc.page_content[:200])

        search_results.append(
            {
                "Query": query,
                "Rank": rank,
                "Retrieved_Document": doc.page_content
            }
        )

        similarity_scores.append(
            {
                "Query": query,
                "Rank": rank,
                "Similarity_Score": score,
                "Retrieved_Document": doc.page_content[:150]
            }
        )

        rank += 1

# =====================================================
# Save Search Results
# =====================================================

results_df = pd.DataFrame(
    search_results
)

results_df.to_csv(
    "dataset/search_results.csv",
    index=False
)

print("\nsearch_results.csv saved.")

# =====================================================
# Save Similarity Scores
# =====================================================

scores_df = pd.DataFrame(
    similarity_scores
)

scores_df.to_csv(
    "dataset/similarity_scores.csv",
    index=False
)

print("similarity_scores.csv saved.")

# =====================================================
# Final Summary
# =====================================================

print("\nTotal Retrieved Records:", len(search_results))
print("Semantic Search Completed Successfully.")