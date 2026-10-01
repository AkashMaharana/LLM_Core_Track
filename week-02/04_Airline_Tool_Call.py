from openai import OpenAI
import gradio as gr
import json
from dotenv import load_dotenv

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "gpt-5.1"

load_dotenv()

def main():
    client = OpenAI(
        base_url=endpoint
    )

    prices = {
        "paris": "$149",
        "rome": "$179",
        "berlin": "$129"
    }

    def get_ticket_price(destination):
        return prices.get(destination.lower(), "Price not available")

    price_tool = {
        "type": "function",
        "name": "get_ticket_price",
        "description": "Get the current airline ticket price for a destination city.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": "Destination city, for example Paris"
                }
            },
            "required": ["destination"],
            "additionalProperties": False
        },
        "strict": True
    }

    def chat(message, history):
        messages = []

        # Add previous conversation
        for item in history:
            content = item["content"]

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

        # Add current message
        messages.append({
            "role": "user",
            "content": message
        })

        # First LLM call
        response = client.responses.create(
            model=deployment_name,
            instructions="""
            You are an airline customer-service assistant.

            Help customers with travel and destination questions.

            When the user asks for a ticket price,
            use the get_ticket_price tool.

            Do not invent ticket prices.
            """,
            input=messages,
            tools=[price_tool]
        )

        # Keep the model output in the conversation
        messages += response.output

        # Check whether the model requested a tool
        for item in response.output:

            if item.type == "function_call":

                arguments = json.loads(item.arguments)

                destination = arguments["destination"]

                price = get_ticket_price(destination)

                print(
                    f"Tool called: get_ticket_price({destination}) -> {price}"
                )

                # Send tool result back to the model
                messages.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": price
                })

        # If a tool was called, ask the model to produce final answer
        if any(item.type == "function_call" for item in response.output):

            response = client.responses.create(
                model="gpt-5.1",
                instructions="""
                You are an airline customer-service assistant.

                Use the tool result to answer the customer's question.
                Keep the response concise.
                """,
                input=messages,
                tools=[price_tool]
            )

        return response.output_text

    demo = gr.ChatInterface(
        fn=chat,
        title="Airline Customer Service"
    )

    demo.launch()


if __name__ == "__main__":
    main()