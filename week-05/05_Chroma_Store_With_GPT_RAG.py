from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name_embedding = "text-embedding-3-small"
deployment_name_gpt = "gpt-5.1"

# Pre-requisites: Execute 04_Chroma_Vector_Store.py to create the Chroma database before running this script.

def main():

    openai_client = OpenAI(
        base_url=endpoint
    )

    # ---------------------------------------------------------
    # 1. Connect to existing Chroma database
    # ---------------------------------------------------------

    chroma_client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = chroma_client.get_collection(
        name="employee_policies"
    )

    # ---------------------------------------------------------
    # 2. User question
    # ---------------------------------------------------------

    question = """
    How much vacation do employees receive,
    and how many days can they carry forward?
    """

    # ---------------------------------------------------------
    # 3. Convert question → embedding
    # ---------------------------------------------------------

    embedding_response = openai_client.embeddings.create(
        model=deployment_name_embedding,
        input=question
    )

    question_embedding = (
        embedding_response.data[0].embedding
    )

    # ---------------------------------------------------------
    # 4. Retrieve top relevant chunks
    # ---------------------------------------------------------

    results = collection.query(
        query_embeddings=[
            question_embedding
        ],
        n_results=3
    )

    retrieved_documents = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]
    retrieved_distances = results["distances"][0]

    # ---------------------------------------------------------
    # 5. Inspect retrieved chunks
    # ---------------------------------------------------------

    print("\nRETRIEVED CHUNKS")
    print("=" * 70)

    for index, document in enumerate(
        retrieved_documents
    ):

        print(
            f"\nChunk {index + 1}"
        )

        print(
            "Distance:",
            round(
                retrieved_distances[index],
                4
            )
        )

        print(
            "Source:",
            retrieved_metadata[index]
        )

        print("\nContent:")
        print(document)

    # ---------------------------------------------------------
    # 6. Build context for GPT
    # ---------------------------------------------------------

    context = "\n\n".join(
        retrieved_documents
    )

    # ---------------------------------------------------------
    # 7. Ask GPT using retrieved knowledge
    # ---------------------------------------------------------

    prompt = f"""
Use only the context below to answer the question.

If the answer cannot be determined from the context,
say:

"I don't know based on the provided information."

Do not invent facts.

CONTEXT:

{context}

QUESTION:

{question}
"""

    response = openai_client.responses.create(
        model=deployment_name_gpt,
        input=prompt
    )

    # ---------------------------------------------------------
    # 8. Final grounded answer
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(
        response.output_text
    )

    # ---------------------------------------------------------
    # 9. Sources
    # ---------------------------------------------------------

    print("\nSOURCES")
    print("=" * 70)

    unique_sources = set()

    for metadata in retrieved_metadata:

        source = metadata.get(
            "source",
            "Unknown"
        )

        unique_sources.add(source)

    for source in unique_sources:
        print("-", source)


if __name__ == "__main__":
    main()