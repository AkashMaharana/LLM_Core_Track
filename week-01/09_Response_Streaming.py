from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "gpt-5.1"

def main():
    client = OpenAI(
        base_url=endpoint
    )

    stream = client.responses.create(
                model = deployment_name,
                input="Explain Kafka consumer groups in 5 short points.",
                stream=True
            )

    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)

    print()


if __name__ == "__main__":
    main()