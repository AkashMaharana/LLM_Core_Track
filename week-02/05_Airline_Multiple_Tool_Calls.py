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

    flight_status = {
        "BA249": "On Time",
        "AI101": "Delayed by 30 minutes"
    }

    baggage_allowance = {
        "economy": "23 kg",
        "business": "32 kg"
    }

    def get_ticket_price(destination):
        return prices.get(
            destination.lower(),
            "Price not available"
        )

    def get_flight_status(flight_number):
        return flight_status.get(
            flight_number.upper(),
            "Flight status not available"
        )

    def get_baggage_allowance(ticket_class):
        return baggage_allowance.get(
            ticket_class.lower(),
            "Baggage information not available"
        )

    tools = [
        {
            "type": "function",
            "name": "get_ticket_price",
            "description": "Get the current ticket price for a destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string"
                    }
                },
                "required": ["destination"],
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "get_flight_status",
            "description": "Get the current status of a flight.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_number": {
                        "type": "string"
                    }
                },
                "required": ["flight_number"],
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "get_baggage_allowance",
            "description": "Get baggage allowance for a ticket class.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_class": {
                        "type": "string"
                    }
                },
                "required": ["ticket_class"],
                "additionalProperties": False
            },
            "strict": True
        }
    ]

    instructions = """
    You are an airline customer-service assistant.

    Use the available tools whenever current airline
    information is required.

    You can:
    - get ticket prices
    - get flight status
    - get baggage allowance

    Do not invent airline information.
    Keep responses concise.
    """

    def chat(message, history):

        # This contains ONLY user-visible conversation history.
        messages = []

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

        messages.append({
            "role": "user",
            "content": message
        })

        # First model call for this user turn.
        response = client.responses.create(
            model=deployment_name,
            instructions=instructions,
            input=messages,
            tools=tools
        )

        while True:

            tool_calls = [
                item
                for item in response.output
                if item.type == "function_call"
            ]

            # No tool call -> model has produced the final answer.
            if not tool_calls:
                return response.output_text

            tool_outputs = []

            for tool_call in tool_calls:

                args = json.loads(tool_call.arguments)

                if tool_call.name == "get_ticket_price":

                    result = get_ticket_price(
                        args["destination"]
                    )

                elif tool_call.name == "get_flight_status":

                    result = get_flight_status(
                        args["flight_number"]
                    )

                elif tool_call.name == "get_baggage_allowance":

                    result = get_baggage_allowance(
                        args["ticket_class"]
                    )

                else:
                    result = "Unknown tool"

                print(
                    f"Tool: {tool_call.name} "
                    f"Args: {args} "
                    f"Result: {result}"
                )

                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": result
                })

            # Continue ONLY the current tool-call chain.
            response = client.responses.create(
                model="gpt-5.1",
                instructions=instructions,
                previous_response_id=response.id,
                input=tool_outputs,
                tools=tools
            )

    demo = gr.ChatInterface(
        fn=chat,
        title="Airline Customer Service"
    )

    demo.launch()


if __name__ == "__main__":
    main()