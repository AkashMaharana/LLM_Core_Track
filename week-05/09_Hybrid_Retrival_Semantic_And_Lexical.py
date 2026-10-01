from rank_bm25 import BM25Okapi

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name_embedding = "text-embedding-3-small"


def main():

    # ---------------------------------------------------------
    # 1. Load the same source used for the vector index
    # ---------------------------------------------------------

    loader = TextLoader(
        "/Users/in46032455/Documents/My Learnings/LLM_0_To_1/Core_Track/week-05/knowledge_base/employee_policy.txt"
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30
    )

    chunks = splitter.split_documents(
        documents
    )

    # =========================================================
    # BM25 RETRIEVER
    # =========================================================

    # ---------------------------------------------------------
    # 2. Tokenize chunks
    # ---------------------------------------------------------

    tokenized_corpus = [
        chunk.page_content.lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    # =========================================================
    # VECTOR RETRIEVER
    # =========================================================

    embeddings = OpenAIEmbeddings(
        model=deployment_name_embedding,
        base_url=endpoint
    )

    vector_store = Chroma(
        collection_name="employee_policies_langchain",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )

    question = (
        "How much vacation can employees "
        "carry forward?"
    )

    # ---------------------------------------------------------
    # 3. BM25 search
    # ---------------------------------------------------------

    tokenized_query = (
        question.lower().split()
    )

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    bm25_ranked = sorted(
        zip(bm25_scores, chunks),
        key=lambda item: item[0],
        reverse=True
    )[:3]

    # ---------------------------------------------------------
    # 4. Vector search
    # ---------------------------------------------------------

    vector_retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    vector_ranked = vector_retriever.invoke(
        question
    )

    # ---------------------------------------------------------
    # 5. Display independent results
    # ---------------------------------------------------------

    print("\nQUESTION")
    print("=" * 70)
    print(question)

    print("\nBM25 RESULTS")
    print("=" * 70)

    for rank, (score, document) in enumerate(
        bm25_ranked,
        start=1
    ):
        print(
            f"\nRank {rank} | BM25 score: {score:.4f}"
        )
        print(document.page_content)

    print("\nVECTOR RESULTS")
    print("=" * 70)

    for rank, document in enumerate(
        vector_ranked,
        start=1
    ):
        print(
            f"\nRank {rank}"
        )
        print(document.page_content)

    # =========================================================
    # RECIPROCAL RANK FUSION
    # =========================================================

    rrf_scores = {}
    documents_by_text = {}

    rrf_constant = 60

    # ---------------------------------------------------------
    # 6. Add BM25 rankings
    # ---------------------------------------------------------

    for rank, (_, document) in enumerate(
        bm25_ranked,
        start=1
    ):

        text = document.page_content

        documents_by_text[text] = document

        rrf_scores[text] = (
            rrf_scores.get(text, 0)
            + 1 / (rrf_constant + rank)
        )

    # ---------------------------------------------------------
    # 7. Add vector rankings
    # ---------------------------------------------------------

    for rank, document in enumerate(
        vector_ranked,
        start=1
    ):

        text = document.page_content

        documents_by_text[text] = document

        rrf_scores[text] = (
            rrf_scores.get(text, 0)
            + 1 / (rrf_constant + rank)
        )

    # ---------------------------------------------------------
    # 8. Final hybrid ranking
    # ---------------------------------------------------------

    hybrid_results = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    print("\nHYBRID RESULTS")
    print("=" * 70)

    for rank, (text, score) in enumerate(
        hybrid_results[:3],
        start=1
    ):

        print(
            f"\nRank {rank} | RRF score: {score:.6f}"
        )

        print(
            documents_by_text[text].page_content
        )


if __name__ == "__main__":
    main()