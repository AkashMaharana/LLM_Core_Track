from openai import OpenAI
import gradio as gr
import json
import base64
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
        return prices.get(
            destination.lower(),
            "Price not available"
        )

    price_tool = {
        "type": "function",
        "name": "get_ticket_price",
        "description": "Get the current ticket price for a destination city.",
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
    }

    instructions = """
    You are an airline customer-service assistant.

    Help customers with travel and destination questions.

    When the customer asks for ticket prices,
    use the get_ticket_price tool.

    Do not invent ticket prices.

    Keep responses concise.
    """

    def chat(message, history):

        # -----------------------------------------
        # Build conversation context
        # -----------------------------------------

        messages = []

        for item in history:
            messages.append({
                "role": item["role"],
                "content": item["content"]
            })

        messages.append({
            "role": "user",
            "content": message
        })

        # -----------------------------------------
        # GPT-5.1 reasoning + tool calling
        # -----------------------------------------

        response = client.responses.create(
            model=deployment_name,
            instructions=instructions,
            input=messages,
            tools=[price_tool]
        )

        while True:

            tool_calls = [
                item
                for item in response.output
                if item.type == "function_call"
            ]

            # No tool call means we have the final response
            if not tool_calls:
                break

            tool_outputs = []

            for tool_call in tool_calls:

                args = json.loads(tool_call.arguments)

                if tool_call.name == "get_ticket_price":

                    result = get_ticket_price(
                        args["destination"]
                    )

                    print(
                        f"Tool: get_ticket_price "
                        f"Destination: {args['destination']} "
                        f"Result: {result}"
                    )

                    tool_outputs.append({
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": result
                    })

            # Continue only the current tool-call execution
            response = client.responses.create(
                model=deployment_name,
                instructions=instructions,
                previous_response_id=response.id,
                input=tool_outputs,
                tools=[price_tool]
            )

        assistant_response = response.output_text

        # -----------------------------------------
        # Generate destination image
        # -----------------------------------------

        image_response = client.images.generate(
            model="gpt-image-2",
            prompt=f"""
            Create an attractive travel illustration inspired by
            this airline assistant response:

            {assistant_response}

            Do not include text in the image.
            """
        )

        image_data = base64.b64decode(
            image_response.data[0].b64_json
        )

        image_path = "destination.png"

        with open(image_path, "wb") as file:
            file.write(image_data)

        # -----------------------------------------
        # Generate audio response
        # -----------------------------------------

        audio_path = "response.mp3"

        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=assistant_response
        ) as audio_response:

            audio_response.stream_to_file(audio_path)

        # -----------------------------------------
        # Update visible Gradio chat history
        # -----------------------------------------

        history.append({
            "role": "user",
            "content": message
        })

        history.append({
            "role": "assistant",
            "content": assistant_response
        })

        return history, "", image_path, audio_path

    # -----------------------------------------
    # Gradio UI
    # -----------------------------------------

    with gr.Blocks() as demo:

        gr.Markdown("# Airline AI Assistant")

        chatbot = gr.Chatbot()

        message = gr.Textbox(
            label="Ask something"
        )

        send_button = gr.Button("Send")

        with gr.Row():

            image = gr.Image(
                label="Destination"
            )

            audio = gr.Audio(
                label="Audio Response"
            )

        send_button.click(
            fn=chat,
            inputs=[
                message,
                chatbot
            ],
            outputs=[
                chatbot,
                message,
                image,
                audio
            ]
        )

    demo.launch()


if __name__ == "__main__":
    main()