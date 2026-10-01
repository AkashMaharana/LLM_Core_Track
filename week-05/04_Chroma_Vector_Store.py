from openai import OpenAI

import chromadb
from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "text-embedding-3-small"

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():

    openai_client = OpenAI(
        base_url=endpoint
    )

    # ---------------------------------------------------------
    # 1. Load document
    # ---------------------------------------------------------

    loader = TextLoader(
        "/Users/in46032455/Documents/My Learnings/LLM_0_To_1/Core_Track/week-05/knowledge_base/employee_policy.txt"
    )

    documents = loader.load()

    documents[0].metadata["doctype"] = "HR_POLICY"

    # ---------------------------------------------------------
    # 2. Split document into chunks
    # ---------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30
    )

    chunks = splitter.split_documents(
        documents
    )

    print("Total chunks:", len(chunks))

    # ---------------------------------------------------------
    # 3. Extract chunk text
    # ---------------------------------------------------------

    chunk_texts = [
        chunk.page_content
        for chunk in chunks
    ]

    # ---------------------------------------------------------
    # 4. Generate embeddings
    # ---------------------------------------------------------

    embedding_response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk_texts
    )

    embeddings = [
        item.embedding
        for item in embedding_response.data
    ]

    # ---------------------------------------------------------
    # 5. Create persistent Chroma database
    # ---------------------------------------------------------

    chroma_client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = chroma_client.get_or_create_collection(
        name="employee_policies"
    )

    # ---------------------------------------------------------
    # 6. Store chunks + embeddings + metadata
    # ---------------------------------------------------------

    ids = [
        f"chunk-{index}"
        for index in range(len(chunks))
    ]

    metadatas = [
        chunk.metadata
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=chunk_texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        "Documents stored in Chroma:",
        collection.count()
    )

    # =========================================================
    # QUERY PHASE
    # =========================================================

    question = "How much vacation do employees receive?"

    # ---------------------------------------------------------
    # 7. Embed the user's question
    # ---------------------------------------------------------

    question_response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    question_embedding = (
        question_response.data[0].embedding
    )

    # ---------------------------------------------------------
    # 8. Vector similarity search
    # ---------------------------------------------------------

    results = collection.query(
        query_embeddings=[
            question_embedding
        ],
        n_results=3
    )

    # ---------------------------------------------------------
    # 9. Display retrieved chunks
    # ---------------------------------------------------------

    print("\nQUESTION")
    print(question)

    print("\nTOP MATCHING CHUNKS")

    for index, document in enumerate(
        results["documents"][0]
    ):

        distance = results["distances"][0][index]
        metadata = results["metadatas"][0][index]

        print("\n" + "=" * 60)

        print(
            f"Result {index + 1}"
        )

        print(
            f"Distance: {distance:.4f}"
        )

        print(
            f"Metadata: {metadata}"
        )

        print("\nContent:")
        print(document)


if __name__ == "__main__":
    main()