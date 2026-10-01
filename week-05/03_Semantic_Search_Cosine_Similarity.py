from openai import OpenAI
import math
from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "text-embedding-3-small"

def main():

    client = OpenAI(
        base_url=endpoint
    )

    documents = [
        "Employees receive 25 days of annual leave each year.",
        "Employees may work remotely up to three days per week.",
        "Expenses above £500 require manager approval."
    ]

    question = "How much vacation do employees get?"

    # ---------------------------------------------------------
    # 1. Create embeddings for documents
    # ---------------------------------------------------------

    document_response = client.embeddings.create(
        model=deployment_name,
        input=documents
    )

    document_embeddings = [
        item.embedding
        for item in document_response.data
    ]

    # ---------------------------------------------------------
    # 2. Create embedding for question
    # ---------------------------------------------------------

    question_response = client.embeddings.create(
        model=deployment_name,
        input=question
    )

    question_embedding = (
        question_response.data[0].embedding
    )

    # ---------------------------------------------------------
    # 3. Cosine similarity
    # ---------------------------------------------------------

    def cosine_similarity(vector_a, vector_b):

        dot_product = sum(
            a * b
            for a, b in zip(vector_a, vector_b)
        )

        magnitude_a = math.sqrt(
            sum(a * a for a in vector_a)
        )

        magnitude_b = math.sqrt(
            sum(b * b for b in vector_b)
        )

        return (
            dot_product
            /
            (magnitude_a * magnitude_b)
        )

    # ---------------------------------------------------------
    # 4. Compare question with every document
    # ---------------------------------------------------------

    results = []

    for document, embedding in zip(
        documents,
        document_embeddings
    ):

        score = cosine_similarity(
            question_embedding,
            embedding
        )

        results.append(
            (score, document)
        )

    # Highest similarity first
    results.sort(
        reverse=True,
        key=lambda item: item[0]
    )

    # ---------------------------------------------------------
    # 5. Display result
    # ---------------------------------------------------------

    print("\nQUESTION")
    print(question)

    print("\nSEARCH RESULTS")

    for score, document in results:

        print(
            f"\nScore: {score:.4f}"
        )

        print(document)


if __name__ == "__main__":
    main()