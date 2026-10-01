from openai import OpenAI
import gradio as gr
from dotenv import load_dotenv

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "gpt-5.1"

load_dotenv()


def main():
    client = OpenAI(
        base_url=endpoint
    )

    def chat(message, history):

        messages = []

        # Add previous conversation
        for item in history:
            content = item["content"]

            # Gradio 6 represents text as content blocks
            if isinstance(content, list):
                content = " ".join(
                    block["text"]
                    for block in content
                    if block.get("type") == "text"
                )

            messages.append({
                "role": item["role"],
                "content": content
            })

        # Add current user message
        messages.append({
            "role": "user",
            "content": message
        })

        response = client.responses.create(
            model=deployment_name,
            instructions="""
            You are an airline customer-service assistant.

            You can help customers with general questions about:
            - travel
            - destinations
            - baggage
            - airline services

            If the user asks for current ticket prices,
            flight availability, seat availability, or booking information,
            explain that you do not currently have access to live booking data.

            Do not invent prices or availability.

            Keep responses concise.
            """,
            input=messages
        )

        return response.output_text

    demo = gr.ChatInterface(
        fn=chat,
        title="Airline Customer Service"
    )

    demo.launch(share=True)


if __name__ == "__main__":
    main()