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
        "What happens after five years?"
    )

    # ---------------------------------------------------------
    # 1. Generate expanded queries
    # ---------------------------------------------------------

    prompt = f"""
Generate exactly 3 search queries that could help
retrieve information needed to answer the question.

Return one query per line.

Do not number the queries.
Do not answer the question.

Question:
{question}
"""

    response = model.invoke(
        prompt
    )

    expanded_queries = [
        line.strip()
        for line in response.content.splitlines()
        if line.strip()
    ]

    # Include original query too
    queries = [
        question,
        *expanded_queries,
    ]

    print("\nSEARCH QUERIES")
    print("=" * 70)

    for query in queries:
        print("-", query)

    # ---------------------------------------------------------
    # 2. Retrieve for every query
    # ---------------------------------------------------------

    retrieved_documents = {}

    for query in queries:

        documents = vector_store.similarity_search(
            query,
            k=3,
        )

        for document in documents:

            # Use text as simple deduplication key
            retrieved_documents[
                document.page_content
            ] = document

    # ---------------------------------------------------------
    # 3. Display merged candidates
    # ---------------------------------------------------------

    print("\nRETRIEVED CANDIDATES")
    print("=" * 70)

    for index, document in enumerate(
        retrieved_documents.values(),
        start=1,
    ):

        print(
            f"\nCandidate {index}"
        )

        print(
            document.page_content
        )


if __name__ == "__main__":
    main()