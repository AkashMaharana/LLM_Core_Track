from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name_embedding = "text-embedding-3-small"


def main():

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
        "How much vacation do employees receive "
        "and what other leave rules apply?"
    )

    # ---------------------------------------------------------
    # 1. Normal similarity retrieval
    # ---------------------------------------------------------

    similarity_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3
        }
    )

    similarity_results = similarity_retriever.invoke(
        question
    )

    print("\nSIMILARITY SEARCH")
    print("=" * 70)

    for index, document in enumerate(
        similarity_results
    ):
        print(f"\nResult {index + 1}")
        print(document.page_content)
        print("Metadata:", document.metadata)

    # ---------------------------------------------------------
    # 2. MMR retrieval
    # ---------------------------------------------------------

    mmr_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,

            # First fetch more possible candidates
            "fetch_k": 6,

            # Balance relevance and diversity
            "lambda_mult": 0.5
        }
    )

    mmr_results = mmr_retriever.invoke(
        question
    )

    print("\n\nMMR SEARCH")
    print("=" * 70)

    for index, document in enumerate(
        mmr_results
    ):
        print(f"\nResult {index + 1}")
        print(document.page_content)
        print("Metadata:", document.metadata)

    # ---------------------------------------------------------
    # 3. Metadata-filtered retrieval
    # ---------------------------------------------------------

    filtered_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,
            "filter": {
                "doctype": "HR_POLICY"
            }
        }
    )

    filtered_results = filtered_retriever.invoke(
        question
    )

    print("\n\nFILTERED SEARCH")
    print("=" * 70)

    for index, document in enumerate(
        filtered_results
    ):
        print(f"\nResult {index + 1}")
        print(document.page_content)
        print("Metadata:", document.metadata)


if __name__ == "__main__":
    main()