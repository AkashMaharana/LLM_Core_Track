from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "gpt-5.1"

def main():
    client = OpenAI(
        base_url=endpoint
    )

    problem = """
    Design a review-sentiment pipeline that processes incoming customer
    reviews and maps them against five predefined sentiments per product.
    """

    # Step 1: Architect proposes a design
    architect_response = client.responses.create(
        model=deployment_name,
        instructions="""
        You are a software architect.

        Propose a concise architecture for the given problem.
        Focus on the LLM workflow and data flow.
        """,
        input=problem
    )

    architecture = architect_response.output_text

    print("\n--- ARCHITECTURE ---\n")
    print(architecture)

    # Step 2: Reviewer critiques the design
    reviewer_response = client.responses.create(
        model=deployment_name,
        instructions="""
        You are a critical Staff Engineer reviewing an architecture.

        Identify the three most important problems or risks.
        Do not rewrite the architecture.
        """,
        input=architecture
    )

    review = reviewer_response.output_text

    print("\n--- REVIEW ---\n")
    print(review)

    # Step 3: Architect revises the design using the review
    revised_response = client.responses.create(
        model=deployment_name,
        instructions="""
        You are the original software architect.

        Revise the architecture using the review comments.
        Keep the final design concise.
        """,
        input=f"""
        Original architecture:
        {architecture}

        Review comments:
        {review}
        """
    )

    print("\n--- REVISED ARCHITECTURE ---\n")
    print(revised_response.output_text)


if __name__ == "__main__":
    main()