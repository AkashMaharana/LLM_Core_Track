from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "gpt-5.1"

def main():
    client = OpenAI(
        base_url=endpoint
    )

    short_context = """
    Analyze the sentiment of this review:

    Great screen quality but poor battery life.
    """

    long_context = """
    You are a retail review analyst.

    Product: Smartphone

    Allowed sentiments:
    - Screen
    - Battery
    - Camera
    - Build Quality
    - Performance

    Rules:
    - Only use the provided sentiments.
    - Do not infer characteristics not mentioned.
    - Ignore delivery feedback.
    - Ignore packaging feedback.
    - Return a concise answer.

    Review:
    Great screen quality but poor battery life.
    """

    for context in [short_context, long_context]:

        response = client.responses.create(
            model = deployment_name,
            input=context
        )

        print("\nResponse:")
        print(response.output_text)

        print("Input tokens :", response.usage.input_tokens)
        print("Output tokens:", response.usage.output_tokens)


if __name__ == "__main__":
    main()