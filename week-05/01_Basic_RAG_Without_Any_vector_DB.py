from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "gpt-5.1"

def main():
    client = OpenAI(
        base_url=endpoint
    )

    # ---------------------------------------------------------
    # 1. Tiny local knowledge base
    # ---------------------------------------------------------

    knowledge_base = {
        "leave_policy": """
        Employees receive 25 days of annual leave each year.
        Unused leave can be carried forward up to 5 days.
        """,

        "remote_work": """
        Employees may work remotely up to three days per week.
        Employees must attend the office at least two days per week.
        """,

        "expense_policy": """
        Expenses below £500 can be submitted directly.
        Expenses above £500 require manager approval.
        """
    }

    question = "How many annual leave days do employees receive?"

    # "What is the maternity leave policy?"

    # ---------------------------------------------------------
    # 2. Very simple keyword retrieval
    # ---------------------------------------------------------

    question_words = set(
        question.lower()
        .replace("?", "")
        .split()
    )

    best_document = None
    best_score = 0

    for document_name, document_text in knowledge_base.items():

        document_words = set(
            document_text.lower()
            .replace(".", "")
            .split()
        )

        # Number of words shared by question + document
        score = len(
            question_words.intersection(document_words)
        )

        print(
            f"{document_name}: retrieval score = {score}"
        )

        if score > best_score:
            best_score = score
            best_document = document_text

    # ---------------------------------------------------------
    # 3. Retrieved context
    # ---------------------------------------------------------

    print("\nRETRIEVED CONTEXT")
    print("=" * 60)
    print(best_document)

    # ---------------------------------------------------------
    # 4. Augment the LLM prompt with retrieved knowledge
    # ---------------------------------------------------------

    prompt = f"""
Use only the context below to answer the question.

If the answer is not present in the context,
say: "I don't know based on the provided information."

CONTEXT:

{best_document}

QUESTION:

{question}
"""

    # ---------------------------------------------------------
    # 5. Generate grounded answer
    # ---------------------------------------------------------

    response = client.responses.create(
        model=deployment_name,
        input=prompt
    )

    print("\nANSWER")
    print("=" * 60)
    print(response.output_text)


if __name__ == "__main__":
    main()