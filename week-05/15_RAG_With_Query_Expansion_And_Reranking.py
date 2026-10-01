from pydantic import BaseModel

from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI,
)

from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name_embedding = "text-embedding-3-small"
deployment_name_gpt = "gpt-5.1"

class RankedChunk(BaseModel):
    index: int
    relevance: int


class RankingResult(BaseModel):
    rankings: list[RankedChunk]

# Query Expansion → Retrieval → Reranking → Answer

def main():

    embeddings = OpenAIEmbeddings(
        model=deployment_name_embedding,
        base_url=endpoint
    )

    model = ChatOpenAI(
        model=deployment_name_gpt,
        base_url=endpoint
    )

    vector_store = Chroma(
        collection_name="employee_policies_langchain",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )

    question = (
        "What benefits apply to employees "
        "after five years?"
    )

    # =========================================================
    # 1. QUERY EXPANSION
    # =========================================================

    expansion_prompt = f"""
Generate exactly 3 search queries that may help
retrieve information needed to answer the question.

Return one query per line.
Do not number them.
Do not answer the question.

Question:
{question}
"""

    expansion_response = model.invoke(
        expansion_prompt
    )

    expanded_queries = [
        line.strip()
        for line in expansion_response.content.splitlines()
        if line.strip()
    ]

    queries = [
        question,
        *expanded_queries,
    ]

    print("\nSEARCH QUERIES")
    print("=" * 70)

    for query in queries:
        print("-", query)

    # =========================================================
    # 2. RETRIEVE BROAD CANDIDATE SET
    # =========================================================

    candidate_map = {}

    for query in queries:

        documents = vector_store.similarity_search(
            query,
            k=3,
        )

        for document in documents:

            # Simple deduplication
            candidate_map[
                document.page_content
            ] = document

    candidates = list(
        candidate_map.values()
    )

    print(
        "\nCandidate count:",
        len(candidates)
    )

    # =========================================================
    # 3. PREPARE CANDIDATES FOR RERANKER
    # =========================================================

    candidate_text = "\n\n".join(
        f"""
CHUNK {index}

{document.page_content}
"""
        for index, document in enumerate(
            candidates
        )
    )

    # =========================================================
    # 4. STRUCTURED RERANKER
    # =========================================================

    reranker = model.with_structured_output(
        RankingResult
    )

    ranking = reranker.invoke(
        f"""
Rank the following chunks by relevance
to the user question.

Give every chunk a relevance score from 0 to 10.

10 = directly answers the question.
0 = completely irrelevant.

Return the chunk index and relevance score.

QUESTION:

{question}

CANDIDATES:

{candidate_text}
"""
    )

    # Highest relevance first
    sorted_rankings = sorted(
        ranking.rankings,
        key=lambda item: item.relevance,
        reverse=True,
    )

    # =========================================================
    # 5. SELECT TOP 3
    # =========================================================

    top_rankings = sorted_rankings[:3]

    top_documents = []

    print("\nRERANKED RESULTS")
    print("=" * 70)

    for rank, result in enumerate(
        top_rankings,
        start=1,
    ):

        document = candidates[
            result.index
        ]

        top_documents.append(
            document
        )

        print(
            f"\nRank {rank}"
            f" | Relevance: {result.relevance}/10"
        )

        print(
            document.page_content
        )

    # =========================================================
    # 6. FINAL RAG CONTEXT
    # =========================================================

    context = "\n\n".join(
        document.page_content
        for document in top_documents
    )

    # =========================================================
    # 7. GENERATE FINAL ANSWER
    # =========================================================

    answer = model.invoke(
        f"""
Answer the question using only the context below.

If the answer cannot be determined from the context,
say:

"I don't know based on the provided information."

CONTEXT:

{context}

QUESTION:

{question}
"""
    )

    print("\nFINAL ANSWER")
    print("=" * 70)

    print(
        answer.content
    )


if __name__ == "__main__":
    main()