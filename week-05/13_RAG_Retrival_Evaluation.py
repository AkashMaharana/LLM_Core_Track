from langchain_openai import OpenAIEmbeddings
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

    vector_store = Chroma(
        collection_name="employee_policies_langchain",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )

    # ---------------------------------------------------------
    # Small golden evaluation dataset
    # ---------------------------------------------------------

    eval_cases = [
        {
            "question":
                "How much vacation do employees receive?",

            "expected":
                "25 days",
        },
        {
            "question":
                "How much unused leave can be carried forward?",

            "expected":
                "5 days",
        },
        {
            "question":
                "What happens after five years of service?",

            "expected":
                "additional five days",
        },
        {
            "question":
                "How many days can employees work remotely?",

            "expected":
                "three days",
        },
        {
            "question":
                "When is manager approval required for expenses?",

            "expected":
                "above £500",
        },
    ]

    k = 3
    hits = 0

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    for case in eval_cases:

        question = case["question"]
        expected = case["expected"]

        documents = vector_store.similarity_search(
            question,
            k=k,
        )

        found = any(
            expected.lower()
            in document.page_content.lower()
            for document in documents
        )

        if found:
            hits += 1

        print("\n" + "=" * 70)

        print("QUESTION:")
        print(question)

        print("\nEXPECTED EVIDENCE:")
        print(expected)

        print(
            "\nRESULT:",
            "PASS" if found else "FAIL"
        )

        print("\nRETRIEVED:")

        for document in documents:
            print(
                "-",
                document.page_content
            )

    # ---------------------------------------------------------
    # Final score
    # ---------------------------------------------------------

    score = hits / len(eval_cases)

    print("\n" + "=" * 70)

    print(
        f"Hit@{k}: "
        f"{hits}/{len(eval_cases)} "
        f"= {score:.2%}"
    )


if __name__ == "__main__":
    main()