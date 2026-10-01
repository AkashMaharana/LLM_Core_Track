from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "gpt-5.1"

def main():
    client = OpenAI(
        base_url=endpoint
    )

    messages = [
        {
            "role": "user",
            "content": "My favourite database is PostgreSQL."
        }
    ]

    response = client.responses.create(
                model = deployment_name,
                input = messages
            )

    print("Assistant:")
    print(response.output_text)

    messages.append({
        "role": "assistant",
        "content": response.output_text
    })

    # Adding user messages to context
    messages.append({
        "role": "user",
        "content": "What is my favourite database?"
    })

    response = client.responses.create(
                    model = deployment_name,
                    input = messages
                )

    print("\nAssistant:")
    print(response.output_text)


    print("\nInput tokens:", response.usage.input_tokens)


if __name__ == "__main__":
    main()